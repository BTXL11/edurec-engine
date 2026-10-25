"""先修关系：概念级依赖图 + 课程→概念映射。

数据源：
    relations/prerequisite-dependency.json   概念 -> 后继概念（2 列 TSV，1027 行）
    relations/course-concept.json            课程 -> 概念（2 列 TSV，167,751 行）
    relations/concept-field.json             概念 -> 领域（可选）

概念 id 形如 `K_计算机科学_计算机科学技术`（K_概念名_领域）。

先修判断思路（这是原型，规则明确可解释）：
    课程 A 是课程 B 的先修，当且仅当 A 的概念集合 ∩ B 的概念先修闭包 ≠ ∅。
    prerequisite_score(user, course) = 用户已学课程覆盖了该课程先修概念的多少比例。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommender.stats import PROCESSED, load_cache, read_tsv_pairs, save_cache  # noqa: E402

CACHE_NAME = "prereq_graph.json"


def build_graph(verbose=True):
    """构建先修依赖图与课程-概念映射，并缓存。"""
    cached = load_cache(CACHE_NAME)
    if cached:
        if verbose:
            print(f"  使用缓存先修图（{cached['n_edges']} 条依赖, "
                  f"{len(cached['course_concepts'])} 门课程有概念）")
        return cached

    edges = read_tsv_pairs("relations/prerequisite-dependency.json")
    # 注意文件方向：第一列是"前置概念"，第二列是"依赖它的概念"
    prereq_of = {}     # concept -> set(它的先修概念)
    successor_of = {}  # concept -> set(后继概念)
    for a, b in edges:
        prereq_of.setdefault(b, set()).add(a)
        successor_of.setdefault(a, set()).add(b)

    course_concepts = {}
    for c, k in read_tsv_pairs("relations/course-concept.json"):
        course_concepts.setdefault(c, set()).add(k)

    try:
        field_of = {}
        for k, f in read_tsv_pairs("relations/concept-field.json"):
            field_of.setdefault(k, set()).add(f)
    except FileNotFoundError:
        field_of = {}

    graph = {
        "n_edges": len(edges),
        "prereq_of": {k: sorted(v) for k, v in prereq_of.items()},
        "successor_of": {k: sorted(v) for k, v in successor_of.items()},
        "course_concepts": {k: sorted(v) for k, v in course_concepts.items()},
        "field_of": {k: sorted(v) for k, v in field_of.items()},
    }
    save_cache(CACHE_NAME, graph)
    if verbose:
        print(f"  先修依赖 {len(edges)} 条，涉及 {len(prereq_of)} 个被依赖概念")
        print(f"  课程-概念映射: {len(course_concepts)} 门课程")
    return graph


def prereq_closure(graph, concepts, max_depth=5):
    """求一组概念的先修闭包（含自身）。带深度限制，防止环导致无限展开。"""
    seen = set()
    frontier = set(concepts)
    depth = 0
    prereq_of = graph["prereq_of"]
    while frontier and depth < max_depth:
        nxt = set()
        for c in frontier:
            if c in seen:
                continue
            seen.add(c)
            nxt.update(prereq_of.get(c, ()))
        frontier = nxt - seen
        depth += 1
    return seen


_PREREQ_CACHE = {}      # (id(graph), course_id, max_depth) -> frozenset(先修概念)


def course_prereq_concepts(graph, course_id, max_depth=5):
    """某门课程的"先修概念集合"（不含课程自身概念）。

    带缓存：一次评测里同一门课会被反复询问（每用户每候选一次），
    而闭包涉及上百个概念的集合运算，不加缓存会白算上百万次。
    """
    key = (id(graph), course_id, max_depth)
    hit = _PREREQ_CACHE.get(key)
    if hit is not None:
        return hit
    own = set(graph["course_concepts"].get(course_id, ()))
    if not own:
        _PREREQ_CACHE[key] = frozenset()
        return _PREREQ_CACHE[key]
    closure = prereq_closure(graph, own, max_depth=max_depth)
    _PREREQ_CACHE[key] = frozenset(closure - own)
    return _PREREQ_CACHE[key]


def prerequisite_score(graph, learned_concepts, course_id, max_depth=5):
    """用户对候选课程先修要求的满足度 ∈ [0,1]。

    learned_concepts: 用户已学课程涉及的概念集合。
    返回 (score, 缺失的必须先修概念数, 总先修概念数)。
    """
    need = course_prereq_concepts(graph, course_id, max_depth=max_depth)
    if not need:
        # 该课程没有先修要求（或没有概念映射）——视为无约束，给中性分
        return None, 0, 0
    have = need & set(learned_concepts)
    return len(have) / len(need), len(need) - len(have), len(need)


if __name__ == "__main__":
    print("构建先修图 ...")
    g = build_graph()
    print()

    # 规模验证
    sizes = sorted((len(v) for v in g["course_concepts"].values()), reverse=True)
    print(f"课程概念数: 最多 {sizes[0]}, 中位 {sizes[len(sizes)//2]}, 最少 {sizes[-1]}")
    dep_sizes = sorted((len(v) for v in g["prereq_of"].values()), reverse=True)
    print(f"单概念先修数: 最多 {dep_sizes[0]}, 中位 {dep_sizes[len(dep_sizes)//2]}")

    # 找一个概念较多的课程看先修闭包
    top = max(g["course_concepts"].items(), key=lambda kv: len(kv[1]))
    cid, concepts = top
    own = set(concepts)
    need = course_prereq_concepts(g, cid)
    print(f"\n样例课程: {cid}")
    print(f"  自身概念数: {len(own)}")
    print(f"  先修概念数: {len(need)}")
    print(f"  自身概念样例: {sorted(own)[:3]}")
    print(f"  先修概念样例: {sorted(need)[:3]}")

    # 刚才生成的 20 门课里有多少有概念/先修
    import json

    from recommender.idmap import build_reverse_map, resolve

    root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "model")
    codes = json.load(open(os.path.join(root, "output", "graphs", "v01", "_codes.json"),
                           encoding="utf-8"))
    reverse = build_reverse_map()

    n_concept = n_prereq = 0
    for c in codes:
        raw = resolve(c, g["course_concepts"], reverse=reverse)
        if raw and raw in g["course_concepts"]:
            n_concept += 1
            if course_prereq_concepts(g, raw):
                n_prereq += 1
    print(f"\n20 门课中: 有概念映射 {n_concept}, 其中有先修要求 {n_prereq}")

