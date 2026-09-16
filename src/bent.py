"""一榀刚架 + 纵向拉伸 + 建模算子。

**这是对"可实现的刚架种类太少"的正面回答。**

原来只有两个生成器——正交矩形网格和坡屋面门式——中间到"逐个列节点坐标"
之间什么都没有。而模板是加法：K 个模板给 K 种结构。这里换成乘法：

    一榀（平面内任意折线） → 纵向拉伸成多榀 → 算子逐步改

三层配起来，绝大多数刚架都出得来，而且**每一层都是一句自然语言能说清的**，
正好也是 Agent 的落点。

---

**为什么"一榀"要用折线定义。**

钢结构工程师真实的建模顺序是：先画一榀横向刚架，再沿纵向复制，
最后用连系梁和支撑连起来。而"跨 × 开间 × 层"三个数组只能生成正交长方体。

这里把**顶层轮廓写成一条折线** `profile`，柱从折线上对应的高度落下来。
于是：

    平屋面   profile = [(0, 7.5), (24, 7.5)]
    双坡     profile = [(0, 7.5), (12, 8.7), (24, 7.5)]
    单坡     profile = [(0, 6.0), (24, 9.0)]
    带悬挑   profile 的两端伸到柱子外面
    错层     柱脚标高逐个给

同一套参数覆盖掉了原来要写四个生成器的东西。
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from generator import (DEFAULT_BEAM, DEFAULT_COLUMN, DEFAULT_MATERIAL,
                       GeneratorError, _check_positive, _merge_fix,
                       _OUT_OF_PLANE)

TOL = 1e-9


def _profile_z(profile: Sequence[tuple[float, float]], x: float) -> float:
    """折线在 x 处的高度，线性插值。

    x 落在折线之外时报错而不是外推——**外推出来的柱顶标高看起来是对的，
    但它不在结构上**，用户很难发现。
    """
    pts = sorted(profile)
    if x < pts[0][0] - TOL or x > pts[-1][0] + TOL:
        raise GeneratorError(
            f"柱的位置 x={x:g} 落在屋面折线 "
            f"[{pts[0][0]:g}, {pts[-1][0]:g}] 之外。"
            "若要做悬挑，请把折线两端延伸出去，而不是把柱移出去。")
    for (x0, z0), (x1, z1) in zip(pts, pts[1:], strict=False):
        if x0 - TOL <= x <= x1 + TOL:
            if abs(x1 - x0) < TOL:
                return max(z0, z1)
            t = (x - x0) / (x1 - x0)
            return z0 + t * (z1 - z0)
    return pts[-1][1]


def _check_profile(profile) -> list[tuple[float, float]]:
    try:
        pts = [(float(x), float(z)) for x, z in profile]
    except (TypeError, ValueError):
        raise GeneratorError("屋面折线必须由有限数字坐标 (x, z) 组成") from None
    if len(pts) < 2:
        raise GeneratorError("屋面折线至少要两个点")
    if not all(math.isfinite(value) for point in pts for value in point):
        raise GeneratorError("屋面折线坐标必须是有限数，不能使用 NaN 或 Infinity")
    xs = [p[0] for p in pts]
    if sorted(xs) != xs:
        raise GeneratorError("屋面折线的点必须按 x 从小到大给出")
    if len(set(xs)) != len(xs):
        raise GeneratorError("屋面折线里有两个点的 x 相同，无法插值")
    return pts


def generate_bent(
    profile: Sequence[Sequence[float]],
    columns: Sequence[float],
    *,
    levels: Sequence[float] = (),
    base_levels: Sequence[float] | None = None,
    column_section: str = DEFAULT_COLUMN,
    beam_section: str = DEFAULT_BEAM,
    material: str = DEFAULT_MATERIAL,
    base: str = "fixed",
    plane_restraints: bool = True,
) -> tuple[dict[str, Any], dict[str, list[int]]]:
    """生成一榀平面刚架。

    参数
    ----
    profile : 顶层折线 [(x, z), ...]，按 x 递增。屋面形状全靠它
    columns : 落柱的 x 坐标。柱顶标高由折线插值决定
    levels : 中间楼层标高，自下而上。多层框架用它
    base_levels : 各柱柱脚标高；留空则全为 0。**错层和台阶靠它**
    base : "free" 只建几何 / "fixed" 柱底固接 / "pinned" 柱底铰接
    plane_restraints : 自动补面外约束。单榀平面刚架在三维求解器里
        不补就是机构——二维程序只是把这件事隐含掉了

    返回 `(model, ids)`：模型字典本身符合 MODEL_SCHEMA，**编号清单单独给**。

    分开返回是必须的：schema 里 `additionalProperties: False`，把
    `column_member_ids` 这种辅助信息塞进模型会被直接拒掉。第一版就是
    混在一起返回的，几何和约束全对，却卡在校验上。
    """
    pts = _check_profile(profile)
    try:
        xs = [float(x) for x in columns]
    except (TypeError, ValueError):
        raise GeneratorError("柱的 x 坐标必须是有限数字") from None
    if not xs:
        raise GeneratorError("至少要有一根柱")
    if not all(math.isfinite(value) for value in xs):
        raise GeneratorError("柱的 x 坐标必须是有限数，不能使用 NaN 或 Infinity")
    if sorted(xs) != xs or len(set(xs)) != len(xs):
        raise GeneratorError("柱的 x 坐标必须严格递增且不重复")
    try:
        levels = [float(value) for value in levels]
    except (TypeError, ValueError):
        raise GeneratorError("中间楼层标高必须是有限数字") from None
    if not all(math.isfinite(value) for value in levels):
        raise GeneratorError("中间楼层标高必须是有限数，不能使用 NaN 或 Infinity")
    if sorted(levels) != levels or len(set(levels)) != len(levels):
        raise GeneratorError("中间楼层标高必须自下而上严格递增且不重复")
    if base_levels is None:
        bases = [0.0] * len(xs)
    else:
        try:
            bases = [float(z) for z in base_levels]
        except (TypeError, ValueError):
            raise GeneratorError("柱脚标高必须是有限数字") from None
        if len(bases) != len(xs):
            raise GeneratorError(
                f"base_levels 有 {len(bases)} 项，但有 {len(xs)} 根柱，数量要一致")
        if not all(math.isfinite(value) for value in bases):
            raise GeneratorError("柱脚标高必须是有限数，不能使用 NaN 或 Infinity")
    if base not in {"free", "fixed", "pinned"}:
        raise GeneratorError(
            f'base 只能是 "free"、"fixed" 或 "pinned"，收到 {base!r}')

    tops = [_profile_z(pts, x) for x in xs]
    for x, z0, z1 in zip(xs, bases, tops):
        if z1 <= z0 + TOL:
            raise GeneratorError(
                f"x={x:g} 处的柱顶标高 {z1:g} 不高于柱脚标高 {z0:g}，柱长为零或为负")

    nodes: list[dict[str, Any]] = []
    index: dict[tuple[float, float], int] = {}

    def node_at(x: float, z: float) -> int:
        """按坐标取节点，没有就新建。**去重必须按坐标**——
        折线拐点和柱顶常常重合，各建一个就会出现两个位置相同的节点，
        求解不报错，但那根杆等于没连上。"""
        key = (round(x, 9), round(z, 9))
        if key not in index:
            index[key] = len(nodes) + 1
            nodes.append({"id": index[key], "x": x, "y": 0.0, "z": z})
        return index[key]

    members: list[dict[str, Any]] = []

    def add(i: int, j: int, section: str) -> int:
        members.append({"id": len(members) + 1, "i": i, "j": j,
                        "section": section, "material": material})
        return members[-1]["id"]

    # --- 柱：从柱脚往上，中间楼层处断开成段 ---
    column_ids: list[int] = []
    base_nodes: list[int] = []
    level_nodes: dict[float, list[int]] = {z: [] for z in levels}
    top_nodes: list[int] = []

    for x, z0, z1 in zip(xs, bases, tops):
        cuts = [z0] + [z for z in levels if z0 + TOL < z < z1 - TOL] + [z1]
        ids = [node_at(x, z) for z in cuts]
        base_nodes.append(ids[0])
        top_nodes.append(ids[-1])
        for z, nid in zip(cuts, ids):
            if z in level_nodes:
                level_nodes[z].append(nid)
        for a, b in zip(ids, ids[1:], strict=False):
            column_ids.append(add(a, b, column_section))

    # --- 楼层梁：同一标高上相邻两柱之间 ---
    beam_ids: list[int] = []
    for z in levels:
        ids = level_nodes[z]
        for a, b in zip(ids, ids[1:], strict=False):
            beam_ids.append(add(a, b, beam_section))

    # --- 顶层：沿折线逐段相连，折线拐点自然成为节点 ---
    # 柱顶也要在折线上取到，否则柱和屋面接不上
    breaks = sorted({p[0] for p in pts} | set(xs))
    roof_nodes = [node_at(x, _profile_z(pts, x)) for x in breaks]
    rafter_ids = [add(a, b, beam_section)
                  for a, b in zip(roof_nodes, roof_nodes[1:], strict=False)]

    supports: list[dict[str, Any]] = []
    fixed_at: dict[int, list[int]] = {}
    if base != "free":
        fix = ([1, 1, 1, 1, 1, 1] if base == "fixed"
               else [1, 1, 1, 0, 0, 0])
        fixed_at = {nid: list(fix) for nid in base_nodes}
    if plane_restraints:
        # 平面刚架在三维求解器里必须补面外约束，否则整体绕 X 转是刚体位移
        for node in nodes:
            here = fixed_at.get(node["id"], [0, 0, 0, 0, 0, 0])
            fixed_at[node["id"]] = _merge_fix(here, _OUT_OF_PLANE)
    for nid in sorted(fixed_at):
        supports.append({"node": nid, "fix": fixed_at[nid]})

    model = {"nodes": nodes, "members": members, "supports": supports}
    ids = {"column_member_ids": column_ids,
           "beam_member_ids": beam_ids,
           "rafter_member_ids": rafter_ids,
           "top_node_ids": top_nodes}
    return model, ids


# =================================================================== 纵向拉伸

def extrude_bents(model: dict[str, Any], bays: Sequence[float], *,
                  tie_section: str | None = None,
                  material: str | None = None) -> tuple[dict, dict]:
    """把一榀沿 Y 方向复制成多榀，并用连系梁把柱顶连起来。

    `bays` 是**各开间宽度**，不是开间数量——不等间距是常态（山墙开间
    往往小一些），只给一个间距和一个数量就做不出来。

    连系梁只连**同标高的对应节点**：一榀里 x 相同、z 相同的点，
    在相邻两榀之间连一根。不按"所有节点两两相连"来，那会连出一堆
    穿过室内空间的斜杆。
    """
    if not bays:
        raise GeneratorError("至少要有一个开间；单榀刚架不需要拉伸")
    widths = _check_positive("bays", bays)
    src_nodes = model["nodes"]
    if not src_nodes:
        raise GeneratorError("当前一榀没有节点")
    source_y = float(src_nodes[0]["y"])
    if any(abs(float(node["y"]) - source_y) > TOL for node in src_nodes):
        raise GeneratorError(
            "当前模型已经包含多个 Y 平面，不能再次按一榀刚架拉伸。"
            "请回到单榀模型，或使用复制/阵列工具。")
    occupied = [name for name in ("nodal_loads", "member_loads", "member_spans",
                                   "settlements", "load_cases", "combos", "sets")
                if model.get(name)]
    if occupied:
        raise GeneratorError(
            "拉伸应在施加载荷和建立集合之前完成；当前模型已有 "
            + "、".join(occupied) + "，为避免编号变化造成数据错挂，已取消拉伸")

    ys = [source_y]
    for w in widths:
        ys.append(ys[-1] + w)

    src_members = model["members"]
    src_supports = {s["node"]: s["fix"] for s in model.get("supports") or []}

    nodes, members, supports = [], [], []
    ids: dict[int, list[int]] = {}          # 原节点号 → 各榀的节点号

    for y in ys:
        for node in src_nodes:
            new_id = len(nodes) + 1
            nodes.append({"id": new_id, "x": node["x"], "y": y, "z": node["z"]})
            ids.setdefault(int(node["id"]), []).append(new_id)

    for k in range(len(ys)):
        for m in src_members:
            members.append({**m, "id": len(members) + 1,
                            "i": ids[int(m["i"])][k],
                            "j": ids[int(m["j"])][k]})

    # --- 连系梁 ---
    tie = tie_section or (src_members[0]["section"] if src_members else DEFAULT_BEAM)
    mat = material or (src_members[0].get("material") if src_members
                       else DEFAULT_MATERIAL)
    tie_ids: list[int] = []
    # 只连柱顶与楼层节点：柱脚在地上，连系梁连柱脚没有意义
    base_nodes = {int(nid) for nid, fix in src_supports.items()
                  if len(fix) == 6 and fix[0] and fix[2]}
    connect = {int(node["id"]) for node in src_nodes
               if int(node["id"]) not in base_nodes}
    for original in sorted(connect):
        chain = ids[original]
        for a, b in zip(chain, chain[1:], strict=False):
            members.append({"id": len(members) + 1, "i": a, "j": b,
                            "section": tie, "material": mat})
            tie_ids.append(members[-1]["id"])

    # --- 约束 ---
    # **拉伸之后不再是平面刚架，面外约束必须去掉。**
    # 留着的话整个结构在 Y 方向被钉死，横向刚度算出来是假的——
    # 而且不报错，只是结果偏刚。这是拉伸这一步最容易出的错。
    for original, fix in src_supports.items():
        if int(original) not in base_nodes:
            continue                        # 非柱脚的面外约束整条丢掉
        real = list(fix)
        # 铰柱脚的 rx/rz 来自平面模型的人工面外约束；空间化后应释放。
        # 固定柱脚 ry=1，三个转角则仍全部保留。
        if not real[4]:
            real[3] = 0
            real[5] = 0
        for new_id in ids[int(original)]:
            supports.append({"node": new_id, "fix": real})

    out = {"nodes": nodes, "members": members, "supports": supports}
    for key in ("materials", "sections", "units"):
        if key in model:
            out[key] = model[key]
    return out, {"tie_member_ids": tie_ids, "bent_count": len(ys),
                 "node_map": ids}


# =================================================================== 算子

def _member_length(model: dict, m: dict) -> float:
    by_id = {n["id"]: n for n in model["nodes"]}
    a, b = by_id[m["i"]], by_id[m["j"]]
    return math.dist((a["x"], a["y"], a["z"]), (b["x"], b["y"], b["z"]))


def select_members(model: dict[str, Any], *,
                   section: str | None = None,
                   orientation: str | None = None,
                   x_range: Sequence[float] | None = None,
                   y_range: Sequence[float] | None = None,
                   z_range: Sequence[float] | None = None) -> list[int]:
    """按条件挑杆件。**算子的输入靠它，而不是让用户报编号**——
    编号规则是生成器定的，用户并不知道 27 号在哪。

    orientation : "vertical" 竖杆（柱）/ "horizontal" 水平杆 / "inclined" 斜杆
    x/y/z_range : 杆件**中点**落在区间内。用中点而不是两端，
        是为了让"这一跨的梁"这种说法有确定的含义
    """
    by_id = {n["id"]: n for n in model["nodes"]}
    out = []
    for m in model.get("members") or []:
        if section and m.get("section") != section:
            continue
        a, b = by_id[m["i"]], by_id[m["j"]]
        dx, dy, dz = b["x"] - a["x"], b["y"] - a["y"], b["z"] - a["z"]
        horizontal = math.hypot(dx, dy)
        if orientation == "vertical" and horizontal > 1e-6:
            continue
        if orientation == "horizontal" and abs(dz) > 1e-6:
            continue
        if orientation == "inclined" and (horizontal < 1e-6 or abs(dz) < 1e-6):
            continue
        mid = ((a["x"] + b["x"]) / 2, (a["y"] + b["y"]) / 2,
               (a["z"] + b["z"]) / 2)
        for rng, value in ((x_range, mid[0]), (y_range, mid[1]),
                           (z_range, mid[2])):
            if rng is not None and not (rng[0] - TOL <= value <= rng[1] + TOL):
                break
        else:
            out.append(m["id"])
    return out


def add_bracing(model: dict[str, Any], *,
                kind: str = "X",
                x_range: Sequence[float] | None = None,
                y_range: Sequence[float] | None = None,
                z_range: Sequence[float] | None = None,
                section: str | None = None,
                material: str | None = None,
                pinned: bool = True) -> tuple[dict, dict]:
    """在竖向框格里加支撑。

    支撑是钢框架里最有效的抗侧手段——**加一道 X 撑对侧移的改善，
    往往比把所有柱子加大一号还明显**，而且钢材用量少得多。
    这也是最值得让用户"改一个参数看看"的地方。

    kind : "X" 交叉 / "V" 人字（上弦中点向两侧柱脚）/ "single" 单斜

    支撑默认两端铰接（`pinned=True`）：支撑靠轴力工作，
    按刚接建模会高估它对弯矩的贡献。
    """
    if kind not in {"X", "V", "single"}:
        raise GeneratorError(f'kind 只能是 "X" / "V" / "single"，收到 {kind!r}')

    nodes = {n["id"]: n for n in model["nodes"]}
    members = list(model.get("members") or [])
    columns = select_members(model, orientation="vertical",
                             x_range=x_range, y_range=y_range, z_range=z_range)
    if len(columns) < 2:
        raise GeneratorError(
            "指定范围内找不到两根柱，无法组成框格。"
            "检查 x_range / y_range / z_range 是否框住了要加撑的那一跨。")

    # 把柱按 (y, z 段) 分组，同组里 x 相邻的两根组成一个框格
    grid: dict[tuple, list[tuple[float, dict]]] = {}
    for mid in columns:
        m = next(x for x in members if x["id"] == mid)
        a, b = nodes[m["i"]], nodes[m["j"]]
        low, high = (a, b) if a["z"] <= b["z"] else (b, a)
        key = (round(low["y"], 6), round(low["z"], 6), round(high["z"], 6))
        grid.setdefault(key, []).append((low["x"], {"low": low, "high": high}))

    sec = section or (members[0]["section"] if members else DEFAULT_BEAM)
    mat = material or (members[0].get("material") if members
                       else DEFAULT_MATERIAL)
    release = {"i": ["ry", "rz"], "j": ["ry", "rz"]} if pinned else None

    # 删除过杆件后编号会有空洞，len(members)+1 可能撞上仍存在的较大编号。
    # 例如 1..41 中删掉 5、6 后长度 39，再加撑就从 40 起，覆盖 40、41。
    next_member_id = (max((int(member["id"]) for member in members), default=0) + 1)

    def add(i: int, j: int) -> int:
        nonlocal next_member_id
        entry = {"id": next_member_id, "i": i, "j": j,
                 "section": sec, "material": mat}
        if release:
            entry["releases"] = release
        members.append(entry)
        next_member_id += 1
        return entry["id"]

    brace_ids: list[int] = []
    panels = 0
    for key in sorted(grid):
        cols = sorted(grid[key])
        for (x0, left), (x1, right) in zip(cols, cols[1:], strict=False):
            panels += 1
            if kind == "single":
                brace_ids.append(add(left["low"]["id"], right["high"]["id"]))
            elif kind == "X":
                brace_ids.append(add(left["low"]["id"], right["high"]["id"]))
                brace_ids.append(add(right["low"]["id"], left["high"]["id"]))
            else:                           # V：需要上弦中点，没有就新建
                mid_x = (x0 + x1) / 2
                top_z = left["high"]["z"]
                found = next((n["id"] for n in model["nodes"]
                              if abs(n["x"] - mid_x) < 1e-6
                              and abs(n["z"] - top_z) < 1e-6
                              and abs(n["y"] - left["low"]["y"]) < 1e-6), None)
                if found is None:
                    raise GeneratorError(
                        f"人字撑需要在 x={mid_x:g}、z={top_z:g} 处有一个节点，"
                        "但那里没有。请先在该处打断上弦，或改用 X 撑。")
                brace_ids.append(add(left["low"]["id"], found))
                brace_ids.append(add(right["low"]["id"], found))

    if not brace_ids:
        raise GeneratorError("指定范围内没有可加撑的框格")
    out = {**model, "members": members}
    return out, {"brace_member_ids": brace_ids, "panels": panels}


def remove_members_by(model: dict[str, Any], member_ids: Sequence[int]
                      ) -> tuple[dict, dict]:
    """删杆件——抽柱、开洞都靠它。

    **同时清掉变成孤立点的节点**：留着孤立节点，刚度矩阵就有一整行零，
    求解直接奇异。用户以为自己只是"删了一根柱"，却得到一句"矩阵奇异"，
    完全对不上。
    """
    drop = set(member_ids)
    kept = [m for m in model.get("members") or [] if m["id"] not in drop]
    if len(kept) == len(model.get("members") or []):
        raise GeneratorError(f"没有找到要删除的杆件 {sorted(drop)}")
    used = {m["i"] for m in kept} | {m["j"] for m in kept}
    nodes = [n for n in model["nodes"] if n["id"] in used]
    orphans = [n["id"] for n in model["nodes"] if n["id"] not in used]
    supports = [s for s in model.get("supports") or [] if s["node"] in used]
    out = {**model, "nodes": nodes, "members": kept, "supports": supports}
    return out, {"removed_members": sorted(drop),
                 "removed_orphan_nodes": orphans}


def raise_nodes(model: dict[str, Any], *, dz: float,
                x_range: Sequence[float] | None = None,
                y_range: Sequence[float] | None = None,
                z_range: Sequence[float] | None = None) -> tuple[dict, dict]:
    """把一片区域里的节点整体抬高 dz。

    这正是"把 X=10~20m、Y=5~15m 范围内的上弦节点提高 1m"那类指令的落点：
    抬高中央区域能增加空间刚度、降低跨中位移，是很典型的一个 what-if。
    """
    moved = []
    nodes = []
    for n in model["nodes"]:
        hit = True
        for rng, value in ((x_range, n["x"]), (y_range, n["y"]),
                           (z_range, n["z"])):
            if rng is not None and not (rng[0] - TOL <= value <= rng[1] + TOL):
                hit = False
                break
        if hit:
            nodes.append({**n, "z": n["z"] + dz})
            moved.append(n["id"])
        else:
            nodes.append(dict(n))
    if not moved:
        raise GeneratorError("指定范围内没有节点。检查 x/y/z_range 是否框对了位置")
    return {**model, "nodes": nodes}, {"moved_nodes": moved, "dz": dz}


def retaper(model: dict[str, Any], member_ids: Sequence[int], section: str
            ) -> tuple[dict, dict]:
    """给一批杆件换截面。变截面、局部加强都用它。"""
    ids = set(member_ids)
    members, changed = [], []
    for m in model.get("members") or []:
        if m["id"] in ids:
            members.append({**m, "section": section})
            changed.append(m["id"])
        else:
            members.append(dict(m))
    if not changed:
        raise GeneratorError(f"没有找到要改的杆件 {sorted(ids)}")
    return {**model, "members": members}, {"changed_members": changed,
                                           "section": section}
