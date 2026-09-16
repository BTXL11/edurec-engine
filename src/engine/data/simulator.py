from __future__ import annotations
import numpy as np
from .schema import DataBundle, User, Resource, Behavior, Rating

# 模拟数据的「语义素材」：让资源带有可被文本编码器区分的内容，
# 这是语义召回（CourseHub 式）能跑通的前提——否则标题只是「模拟资源0」这类占位符，
# 任何文本嵌入都无从编码。
CATEGORY_NAMES: list[str] = [
    "编程与软件开发", "数据科学与分析", "人工智能与机器学习", "前端与移动开发",
    "数学与统计", "产品与设计", "商业与创业", "语言学习",
    "考试与升学", "职业发展与求职", "艺术与人文", "健康与生活",
]

_CATEGORY_TOPICS: list[list[str]] = [
    ["Python 编程基础", "Java 面向对象设计", "Git 版本控制", "数据结构与算法"],
    ["SQL 数据库查询", "数据清洗与可视化", "统计学基础", "A/B 实验设计"],
    ["机器学习入门", "深度学习与神经网络", "自然语言处理", "模型调参与评估"],
    ["HTML 与 CSS 布局", "JavaScript 异步编程", "React 组件开发", "移动端适配"],
    ["线性代数", "概率论与数理统计", "微积分入门", "离散数学"],
    ["产品需求分析", "用户体验设计", "原型与交互稿", "用户研究方法"],
    ["商业模式画布", "创业融资入门", "市场调研方法", "增长与运营"],
    ["英语口语训练", "学术写作规范", "日语五十音入门", "雅思阅读技巧"],
    ["高考数学冲刺", "考研英语长难句", "公考行测解题", "雅思写作批改"],
    ["简历撰写与优化", "面试常见问题准备", "职业方向探索", "职场沟通技巧"],
    ["西方美术史", "摄影构图基础", "音乐理论基础", "写作与叙事"],
    ["健身训练计划", "营养与膳食搭配", "睡眠与压力管理", "常见运动损伤预防"],
]

_KIND_WORDS = ("课程", "专题文章", "视频课")
_LEVELS = ("零基础", "入门", "进阶")
_FOCUS = ("核心概念讲解", "动手实战演练", "常见误区梳理", "配套练习与答疑")
_GOALS = ("帮助零基础学习者建立完整知识框架",
          "通过案例驱动掌握实际应用能力",
          "为进阶学习与项目实践打好基础",
          "配合练习与测验巩固关键知识点")

_NAME_POOL_SIZE = 32
_LEVEL_POOL = 64


def _category_name(i: int) -> str:
    """超出素材表时继续给出可读名称（当前默认 12 个类目全在表内）。"""
    if i < len(CATEGORY_NAMES):
        return CATEGORY_NAMES[i]
    return f"扩展类目{i + 1}"


def _category_topics(i: int) -> list[str]:
    if i < len(_CATEGORY_TOPICS):
        return _CATEGORY_TOPICS[i]
    return [f"{_category_name(i)}主题{k + 1}" for k in range(4)]


def generate(config) -> DataBundle:
    rng = np.random.default_rng(config.seed)
    n_u, n_r = config.sim_n_users, config.sim_n_resources
    n_cat, n_tag = config.sim_n_categories, config.sim_n_tags
    n_sent = max(1, int(getattr(config, "sim_description_sentences", 3)))

    # --- 隐藏结构参数 ---
    # 用户：类目偏好(狄利克雷)、类型偏好、活跃度
    user_cat = rng.dirichlet(np.ones(n_cat), size=n_u)          # (n_u, n_cat)
    user_type = rng.dirichlet(np.ones(3), size=n_u)             # course/article/video
    user_act = rng.gamma(2.0, 2.0, size=n_u)                    # 活跃度

    # 资源：类目、标签(1-4个)、质量、热度、冷门度
    res_cat = rng.integers(0, n_cat, size=n_r)
    res_tags = [tuple(rng.choice(n_tag, size=min(int(rng.integers(1, 5)), max(1, n_tag)), replace=False).astype(int).tolist())
                for _ in range(n_r)]
    res_quality = rng.beta(2.0, 2.0, size=n_r)
    res_pop = rng.lognormal(0.0, 1.0, size=n_r)

    resources = []
    for i in range(n_r):
        cat = int(res_cat[i])
        cname = _category_name(cat)
        topics = _category_topics(cat)
        topic = topics[int(rng.integers(len(topics)))]
        kind = _KIND_WORDS[int(rng.integers(len(_KIND_WORDS)))]
        level = _LEVELS[int(rng.integers(len(_LEVELS)))]
        title = f"{topic}{kind}（{level}）{int(rng.integers(_NAME_POOL_SIZE)):02d}"
        parts = [f"这是一门面向{level}学习者的{cname}方向的{kind}，主题是{topic}。"]
        picks = rng.choice(len(_FOCUS), size=min(n_sent, len(_FOCUS)), replace=False)
        for k in picks:
            parts.append(f"侧重{_FOCUS[int(k)]}。")
        parts.append(_GOALS[int(rng.integers(len(_GOALS)))] + "。")
        resources.append(Resource(
            resource_id=i,
            type=("course", "article", "video")[int(rng.integers(3))],
            category_id=cat,
            tags=tuple(str(t) for t in res_tags[i]),
            metadata={"title": title},
            description="".join(parts),
        ))
    users = [User(user_id=i) for i in range(n_u)]

    # --- 采样交互（自适应循环，行为量达标） ---
    behaviors: list[Behavior] = []
    ratings: list[Rating] = []
    p_user = user_act / user_act.sum()
    ts = 1_700_000_000
    attempts = 0
    while len(behaviors) < config.sim_n_interactions and attempts < config.sim_n_interactions * 10:
        attempts += 1
        u = int(rng.choice(n_u, p=p_user))
        i = int(rng.integers(n_r))
        match = user_cat[u, res_cat[i]]                          # 类目匹配度
        p = float(match * res_quality[i] * res_pop[i] + rng.normal(0, 0.1))
        if rng.uniform() > min(1.0, 3.0 * p):                    # 保留一部分交互率
            continue
        action = rng.choice(("view", "click", "favorite"),
                            p=(0.5, 0.3, 0.2))
        behaviors.append(Behavior(user_id=u, resource_id=i, action=str(action), ts=ts))
        if action == "favorite" and rng.uniform() < 0.5:
            score = int(np.clip(round(2.5 + 2.5 * match + rng.normal(0, 0.3)), 1, 5))
            ratings.append(Rating(user_id=u, resource_id=i, score=score, ts=ts))
        ts += 1

    return DataBundle(users=users, resources=resources,
                      behaviors=behaviors, ratings=ratings)
