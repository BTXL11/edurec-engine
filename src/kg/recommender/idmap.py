"""课程 ID 映射：图文件名用的安全名 <-> 原始 course id。

MOOCCube 的原始 id 含 `:` `/`（如 `C_course-v1:ACCA+FA1_X+sp`），而 Windows 下
生成图文件名时被 safe_filename 替换成了 `_`（`C_course-v1_ACCA+FA1_X+sp`）。
热度、选课、概念等关系数据用的都是原始 id，所以要能双向对上。
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommender.stats import load_courses  # noqa: E402

_ILLEGAL = r'[\\/:*?"<>|]'


def safe_filename(s):
    """与 notebook 中一致的 Windows 安全文件名规则。"""
    return re.sub(_ILLEGAL, "_", str(s))


def build_reverse_map():
    """safe 名 -> 原始 id。"""
    rev = {}
    for c in load_courses():
        rev.setdefault(safe_filename(c["id"]), c["id"])
    return rev


def candidate_raw_ids(cid, max_underscores=12):
    """把 Cid 中的 `_` 还原成 `:` 或 `/` 的所有可能组合。

    safe_filename 抹掉了 `:` 与 `/` 的区别，单靠字符串无法唯一还原，
    所以列出候选再拿去别的数据源里验证。下划线过多时放弃（组合爆炸），
    此时依赖反向表即可。
    """
    idxs = [i for i, ch in enumerate(cid) if ch == "_"]
    if len(idxs) > max_underscores:
        return []
    out = []
    for mask in range(1 << len(idxs)):
        chars = list(cid)
        for b, i in enumerate(idxs):
            chars[i] = ":" if (mask >> b) & 1 else "/"
        out.append("".join(chars))
    return out


def resolve(cid, known_ids, reverse=None):
    """把一个 Cid 解析成原始 id；失败返回 None。

    known_ids: 可用的原始 id 集合（例如热度数据的 key），用于验证候选。
    """
    if reverse is None:
        reverse = build_reverse_map()
    raw = reverse.get(cid)
    if raw is not None:
        return raw
    for cand in candidate_raw_ids(cid):
        if cand in known_ids:
            return cand
    return None


def cid_from_filename(path):
    """从图文件路径取 Cid（去掉目录与扩展名）。"""
    return os.path.basename(path).rsplit(".", 1)[0]
