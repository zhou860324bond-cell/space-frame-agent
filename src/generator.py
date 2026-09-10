"""参数化模型生成器。

**大模型负责「意图 → 参数」，这里负责「参数 → 拓扑」。**

让模型逐个列节点坐标是这类系统最常见的失败方式：多层多跨时它一定会数错、漏杆、
把柱接到错误的楼层。所以工具层只暴露 `generate_frame(spans, bays, storeys, ...)`
这样的参数入口，节点编号与杆件连接由确定性代码展开。
"""

from __future__ import annotations

from typing import Any, Sequence

DEFAULT_COLUMN = "COLUMN"
DEFAULT_BEAM = "BEAM"
DEFAULT_MATERIAL = "STEEL"


class GeneratorError(ValueError):
    """参数不合法。消息面向大模型，应当足够具体以便它自己改正。"""


# 面外自由度：uy 平动、绕 x 转、绕 z 转。留 ry 自由，那是平面内弯曲用的。
_OUT_OF_PLANE = (0, 1, 0, 1, 0, 1)


def _merge_fix(a, b):
    return [1 if (x or y) else 0 for x, y in zip(a, b)]


def _add_plane_restraints(supports: list[dict[str, Any]],
                          node_ids: Sequence[int]) -> list[dict[str, Any]]:
    """给单榀平面刚架补面外约束。

    一榀平面刚架放进三维求解器时，面外平动与绕杆轴的转动全无约束——尤其柱底铰接时，
    整榀可以绕两个柱脚的连线自由转动，刚度矩阵必然奇异。二维刚架程序隐含地
    锁住了这些自由度，三维求解器必须显式补上。
    """
    by_node = {int(s["node"]): list(s["fix"]) for s in supports}
    for nid in node_ids:
        nid = int(nid)
        by_node[nid] = _merge_fix(by_node.get(nid, [0] * 6), _OUT_OF_PLANE)
    return [{"node": nid, "fix": by_node[nid]} for nid in sorted(by_node)]


def _check_positive(name: str, values: Sequence[float]) -> list[float]:
    if not values:
        raise GeneratorError(f"{name} 至少要有一项")
    out = []
    for k, v in enumerate(values):
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise GeneratorError(f"{name} 第 {k + 1} 项 {v!r} 不是数字") from None
        if not (f > 0.0):
            raise GeneratorError(f"{name} 第 {k + 1} 项必须大于 0，收到 {f}")
        out.append(f)
    return out


# 柱子的截面朝向必须**写进模型**，不能靠 local_axes 的兜底。
#
# local_axes 对竖直杆有一条兜底分支：参考向量退化时改用全局 X。它是对的，
# 但**恰好竖直和略微倾斜会给出反向的局部 y**——实测倾斜 0.01° 时局部 y 是
# +X，0.1° 时就变成 −X。双轴对称截面刚度不受影响，但局部 z 跟着翻，
# **报出来的 My/Mz/Vy/Vz 全部反号**。
#
# 于是同一榀框架里，一根恰好竖直的柱和一根偏零点几度的柱（图纸识别出来的
# 坐标、抬升节点之后、带坡度的柱）会给出符号相反的弯矩，内力图上有一根柱
# 画在错误的一侧。而这个不连续是本质的：竖直杆没有"向上投影"这个方向，
# 极限依赖于从哪一侧趋近，兜底只能任选一个。
#
# 所以生成器显式写下这个选择：柱子的参考向量取全局 X，与兜底一致（跨方向
# 用强轴，这是常规做法），但从此它记录在模型里、看得见、改得动，也不会
# 因为坐标动了零点几度就翻号。
_COLUMN_REF = (1.0, 0.0, 0.0)


def _is_upright(a: dict, b: dict, tol: float = 1e-9) -> bool:
    """两节点连线是不是竖直（只差 z）。"""
    return abs(a["x"] - b["x"]) <= tol and abs(a["y"] - b["y"]) <= tol


def _cumulative(widths: Sequence[float]) -> list[float]:
    coords, total = [0.0], 0.0
    for w in widths:
        total += w
        coords.append(total)
    return coords


def generate_frame(
    spans: Sequence[float],
    storeys: Sequence[float],
    bays: Sequence[float] = (),
    *,
    column_section: str = DEFAULT_COLUMN,
    beam_section: str = DEFAULT_BEAM,
    material: str = DEFAULT_MATERIAL,
    base: str = "fixed",
    beam_release: bool = False,
    beam_load: float | None = None,
    load_case: str | None = None,
    plane_restraints: bool = True,
) -> dict[str, Any]:
    """生成规则空间刚架的节点与杆件。

    参数
    ----
    spans : 沿 X 方向各跨跨度，例如 [6.0, 6.0, 6.0] 表示三跨
    storeys : 各层层高，自下而上，例如 [3.6, 3.6] 表示两层
    bays : 沿 Y 方向各开间宽度；留空表示平面刚架（单榀）
    base : "free" 只建几何 / "fixed" 柱底固接 / "pinned" 柱底铰接
    plane_restraints : 平面刚架（bays 留空）时自动补面外约束。关掉会让柱底铰接的
        单榀刚架成为机构——三维求解器里这是必然的，二维程序只是把它隐含掉了
    beam_release : 梁两端释放绕局部 z 的弯矩（铰接梁）
    beam_load : 施加在所有梁上的均布线荷载，沿全局 -Z，单位 N/m（正值表示向下）
    load_case : 荷载放进哪个工况；留空则放进默认工况

    返回符合 MODEL_SCHEMA 的模型字典（不含 materials / sections，由调用方补齐）。
    """
    spans = _check_positive("spans", spans)
    storeys = _check_positive("storeys", storeys)
    bays = _check_positive("bays", bays) if bays else []
    if base not in {"free", "fixed", "pinned"}:
        raise GeneratorError(
            f'base 只能是 "free"、"fixed" 或 "pinned"，收到 {base!r}')

    xs = _cumulative(spans)
    ys = _cumulative(bays) if bays else [0.0]
    zs = _cumulative(storeys)
    nx, ny, nz = len(xs), len(ys), len(zs)

    # 节点编号：先 X 后 Y 后 Z，从 1 开始，便于人工核对
    def nid(ix: int, iy: int, iz: int) -> int:
        return iz * nx * ny + iy * nx + ix + 1

    nodes = [
        {"id": nid(ix, iy, iz), "x": xs[ix], "y": ys[iy], "z": zs[iz]}
        for iz in range(nz) for iy in range(ny) for ix in range(nx)
    ]

    members: list[dict[str, Any]] = []
    beam_ids: list[int] = []

    by_id = {n["id"]: n for n in nodes}

    def add(i: int, j: int, section: str, releases: dict | None = None) -> int:
        m: dict[str, Any] = {"id": len(members) + 1, "i": i, "j": j,
                             "section": section, "material": material}
        if _is_upright(by_id[i], by_id[j]):
            m["ref_vector"] = list(_COLUMN_REF)
        if releases:
            m["releases"] = releases
        members.append(m)
        return m["id"]

    # 柱：每个平面网格点，逐层向上
    for iz in range(nz - 1):
        for iy in range(ny):
            for ix in range(nx):
                add(nid(ix, iy, iz), nid(ix, iy, iz + 1), column_section)

    # 梁：每个楼层（不含基底），X 向与 Y 向各一组
    rel = {"i": ["rz"], "j": ["rz"]} if beam_release else None
    for iz in range(1, nz):
        for iy in range(ny):
            for ix in range(nx - 1):
                beam_ids.append(add(nid(ix, iy, iz), nid(ix + 1, iy, iz), beam_section, rel))
        for iy in range(ny - 1):
            for ix in range(nx):
                beam_ids.append(add(nid(ix, iy, iz), nid(ix, iy + 1, iz), beam_section, rel))

    supports: list[dict[str, Any]] = []
    if base != "free":
        fix = ([1, 1, 1, 1, 1, 1] if base == "fixed"
               else [1, 1, 1, 0, 0, 0])
        supports = [{"node": nid(ix, iy, 0), "fix": list(fix)}
                    for iy in range(ny) for ix in range(nx)]
    if plane_restraints and not bays:
        supports = _add_plane_restraints(supports, [n["id"] for n in nodes])

    model: dict[str, Any] = {
        "units": "N-m-Pa",
        "nodes": nodes,
        "members": members,
        "supports": supports,
    }

    if beam_load:
        loads = [{"member": mid, "w": [0.0, 0.0, -abs(float(beam_load))]} for mid in beam_ids]
        if load_case:
            model["load_cases"] = [{"name": load_case, "member_loads": loads}]
        else:
            model["member_loads"] = loads

    return model


def generate_portal_frame(
    spans: Sequence[float],
    eave_height: float,
    ridge_rise: float,
    bays: Sequence[float] = (),
    *,
    column_section: str = DEFAULT_COLUMN,
    rafter_section: str = DEFAULT_BEAM,
    tie_section: str | None = None,
    material: str = DEFAULT_MATERIAL,
    base: str = "pinned",
    rafter_load: float | None = None,
    load_case: str | None = None,
    plane_restraints: bool = True,
) -> dict[str, Any]:
    """生成坡屋面门式刚架 —— 国内钢结构最常见的形式，规整框架生成器做不了。

    每一跨是一榀双坡：两根柱 + 两根斜梁，跨中起脊。多跨则跨与跨之间共用中柱。
    Y 向给了 bays 就沿进深重复布置，并在檐口和屋脊加纵向系杆。

    参数
    ----
    spans : X 向各跨跨度，例如 [24.0] 单跨 24 米，[18.0, 18.0] 两跨
    eave_height : 檐口高度（柱高）
    ridge_rise : 屋脊相对檐口的升高。坡度 i = ridge_rise / (跨度/2)
    bays : Y 向各开间宽度；留空则只生成一榀平面刚架
    base : "free" 只建几何，或柱底 "pinned" 铰接 / "fixed" 固接
    tie_section : 纵向系杆截面；留空则用 rafter_section
    rafter_load : 施加在所有斜梁上的均布线荷载，沿全局 -Z，N/m（正值向下）

    返回符合 MODEL_SCHEMA 的模型字典（不含 materials / sections，由调用方补齐）。
    """
    spans = _check_positive("spans", spans)
    bays = _check_positive("bays", bays) if bays else []
    if not (float(eave_height) > 0.0):
        raise GeneratorError(f"eave_height 必须大于 0，收到 {eave_height}")
    if float(ridge_rise) <= 0.0:
        raise GeneratorError(
            f"ridge_rise 必须大于 0，收到 {ridge_rise}。"
            "要平屋面的规整框架请改用 generate_frame")
    if base not in {"free", "fixed", "pinned"}:
        raise GeneratorError(
            f'base 只能是 "free"、"fixed" 或 "pinned"，收到 {base!r}')

    eave_height = float(eave_height)
    ridge_rise = float(ridge_rise)
    ys = _cumulative(bays) if bays else [0.0]

    # 每一跨在 X 上贡献「柱脚 - 屋脊」两个站位，最后补一个柱脚
    xs: list[float] = [0.0]
    is_ridge: list[bool] = [False]
    total = 0.0
    for width in spans:
        xs.append(total + width / 2.0); is_ridge.append(True)
        total += width
        xs.append(total); is_ridge.append(False)
    nx, ny = len(xs), len(ys)

    # 编号用查表而不是公式：屋脊站位不立柱，那里没有柱脚节点，
    # 用公式算会凭空多出一批悬空节点（第一版就栽在这里，check_model 能查出来，
    # 但求解会先一步报奇异）。
    nodes: list[dict[str, Any]] = []
    index: dict[tuple[int, int, bool], int] = {}

    def _put(ix: int, iy: int, top: bool, z: float) -> None:
        index[(ix, iy, top)] = len(nodes) + 1
        nodes.append({"id": len(nodes) + 1, "x": xs[ix], "y": ys[iy], "z": z})

    for iy in range(ny):                # 柱脚层：只在立柱的站位生成
        for ix in range(nx):
            if not is_ridge[ix]:
                _put(ix, iy, False, 0.0)
    for iy in range(ny):                # 屋面层：每个站位都有，脊点抬高
        for ix in range(nx):
            _put(ix, iy, True, eave_height + (ridge_rise if is_ridge[ix] else 0.0))

    def nid(ix: int, iy: int, top: bool) -> int:
        return index[(ix, iy, top)]

    members: list[dict[str, Any]] = []
    rafter_ids: list[int] = []

    by_id = {n["id"]: n for n in nodes}

    def add(i: int, j: int, section: str) -> int:
        entry: dict[str, Any] = {"id": len(members) + 1, "i": i, "j": j,
                                 "section": section, "material": material}
        if _is_upright(by_id[i], by_id[j]):
            entry["ref_vector"] = list(_COLUMN_REF)
        members.append(entry)
        return members[-1]["id"]

    for iy in range(ny):                # 柱：只在非脊站位立柱
        for ix in range(nx):
            if not is_ridge[ix]:
                add(nid(ix, iy, False), nid(ix, iy, True), column_section)

    for iy in range(ny):                # 斜梁：沿 X 逐段相连，自然形成双坡
        for ix in range(nx - 1):
            rafter_ids.append(add(nid(ix, iy, True), nid(ix + 1, iy, True),
                                  rafter_section))

    if ny > 1:                          # 纵向系杆：檐口与屋脊各一道
        tie = tie_section or rafter_section
        for iy in range(ny - 1):
            for ix in range(nx):
                add(nid(ix, iy, True), nid(ix, iy + 1, True), tie)

    supports: list[dict[str, Any]] = []
    if base != "free":
        fix = ([1, 1, 1, 1, 1, 1] if base == "fixed"
               else [1, 1, 1, 0, 0, 0])
        supports = [{"node": nid(ix, iy, False), "fix": list(fix)}
                    for iy in range(ny) for ix in range(nx) if not is_ridge[ix]]
    if plane_restraints and not bays:
        supports = _add_plane_restraints(supports, [n["id"] for n in nodes])

    model: dict[str, Any] = {
        "units": "N-m-Pa",
        "nodes": nodes,
        "members": members,
        "supports": supports,
    }
    if rafter_load:
        loads = [{"member": mid, "w": [0.0, 0.0, -abs(float(rafter_load))]}
                 for mid in rafter_ids]
        if load_case:
            model["load_cases"] = [{"name": load_case, "member_loads": loads}]
        else:
            model["member_loads"] = loads
    return model


def rafter_member_ids(model: dict[str, Any]) -> list[int]:
    """挑出斜梁编号，供分工况加载时引用。

    **按几何判别，不按截面名。** 纵向系杆默认与斜梁同截面，按名字筛会把系杆一并
    返回，拿去加载就会连系杆一起压上屋面荷载——早期版本正是这么错的。
    斜梁的判据：两端都在屋面层（z > 0）且 x 坐标不同；系杆则是 x 相同、y 不同。
    """
    nodes = {int(n["id"]): n for n in model.get("nodes", [])}
    out = []
    for m in model.get("members", []):
        a, b = nodes.get(int(m["i"])), nodes.get(int(m["j"]))
        if a is None or b is None:
            continue
        if min(a["z"], b["z"]) <= 0.0:
            continue                       # 柱
        if abs(a["x"] - b["x"]) > 1e-9:
            out.append(int(m["id"]))
    return out


def describe(model: dict[str, Any]) -> dict[str, int]:
    """给大模型看的规模摘要——它不该去数节点，让它读这个。"""
    return {
        "nodes": len(model.get("nodes", [])),
        "members": len(model.get("members", [])),
        "supports": len(model.get("supports", [])),
        "load_cases": len(model.get("load_cases", [])) or (1 if model.get("member_loads") or model.get("nodal_loads") else 0),
    }


def beam_member_ids(model: dict[str, Any], beam_section: str = DEFAULT_BEAM) -> list[int]:
    """按截面名挑出梁的编号，供分工况加载时引用。"""
    return [int(m["id"]) for m in model.get("members", [])
            if m.get("section") == beam_section]
