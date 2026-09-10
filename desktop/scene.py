"""把刚架模型变成 PyVista 网格。

**这里一行 Qt 都没有。** 和 Streamlit 那版同一个原则：真正会出错的是
"几何和标量对不对"，不是"控件摆在哪"。把建网格抽成纯函数，就能在无头环境里
逐条验——管子有没有沿杆件走、云图标量是不是每根杆的真实轴力、变形放大了多少倍。
Qt 那层只负责把这些网格塞进渲染器，薄到不需要测。

三种网格：

* **杆件管** —— 每根杆一段折线，`tube()` 成细圆杆。圆杆只表达分析梁的中心线
  与连续实体感，不表达真实截面；同时保留光照，便于在深色视口下辨认空间关系。
* **符号** —— 支座与荷载，几何和 Streamlit 那版同源（`viz_symbols` 里的分类），
  但这里画成实体而不是线，因为 VTK 的实体在三维里更好辨认。
* **标量** —— 轴力/弯矩挂在管子上做云图。**标量是逐点的**，
  所以沿杆长变化的弯矩能画出真实的渐变，而不是一根杆一个色块。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from internal_forces import (max_centerline_displacement,
                             member_diagram, member_displacement)  # noqa: E402
from frame3d import (Frame, Node, Member, local_axes,
                     member_endpoints)                             # noqa: E402
from modal import member_mode_displacement                        # noqa: E402
from viz_symbols import classify_support, model_size            # noqa: E402

# 分析梁只表达中心线与连续实体感，不冒充真实截面。细长比例接近 CAE 梁单元，
# 避免高饱和粗圆管产生“塑料玩具”观感；云图仍有足够表面积显示颜色。
TUBE_RATIO = 0.0032
# 支座符号比管子大，但不该大到反客为主。原值 0.020 是管半径的五倍，
# 加上近白的颜色，支座成了画面上最抢眼的东西。
SYMBOL_RATIO = 0.0145   # 配合细圆杆，支座可辨认但不压过结构主体
# 每根杆件沿长度取几个点。云图和变形都靠它，取太少弯矩渐变会变成折线
STATIONS = 21
# 挠度积分的分辨率。画形状 21 个点就够，但做两次梯形积分不够——
# 分开取，画得快、算得准
DEFLECTION_STATIONS = 201


def preferred_view(frame) -> str:
    """平面结构返回其正投影视图，空间结构返回等轴测。"""
    if frame is None or not frame.nodes:
        return "isometric"
    coords = np.array([node.xyz for node in frame.nodes.values()], dtype=float)
    extents = np.ptp(coords, axis=0)
    span = float(extents.max())
    if span <= 1e-12:
        return "isometric"
    flat_axis = int(np.argmin(extents))
    if extents[flat_axis] > span * 1e-8:
        return "isometric"
    return ("yz", "xz", "xy")[flat_axis]


def auto_deformation_scale(frame, solution, case: str,
                           target_ratio: float = 0.06) -> float:
    """让最大真实位移约占模型尺寸的 ``target_ratio``。

    “真实”包括单元内部挠度，不仅是节点位移。这样单根单元的简支梁也能自动
    放大出跨中曲线；同时保留节点位移峰值，轴向伸长和整体侧移不会被漏掉。
    """
    peak = float(max_centerline_displacement(
        frame, solution, case, stations=DEFLECTION_STATIONS)["value"])
    return target_ratio * model_size(frame) / peak if peak > 0.0 else 1.0


def auto_mode_scale(frame, shapes: np.ndarray, mode: int,
                    target_ratio: float = 0.12) -> float:
    """按恢复后的中心线平移定振型比例，不能把弧度转角和长度直接比。"""
    peak = 0.0
    for mid in frame.members:
        _, displacement = member_mode_displacement(
            frame, shapes, mode, mid, DEFLECTION_STATIONS)
        peak = max(peak, float(np.linalg.norm(displacement, axis=1).max(initial=0.0)))
    return target_ratio * model_size(frame) / peak if peak > 0.0 else 1.0


def _polyline(points: np.ndarray) -> pv.PolyData:
    """一串点连成折线。"""
    n = len(points)
    mesh = pv.PolyData()
    mesh.points = points
    mesh.lines = np.hstack([[n], np.arange(n)])
    return mesh


def analysis_mesh_polylines(frame, preview: dict) -> pv.PolyData:
    """把 ``preview_analysis_mesh`` 的分析单元转换为独立线段。"""
    coordinates = {int(node_id): np.asarray(node.xyz, dtype=float)
                   for node_id, node in frame.nodes.items()}
    for item in preview.get("split_node_details", ()):
        coordinates[int(item["node"])] = np.asarray(item["coordinates"], dtype=float)

    blocks: list[pv.PolyData] = []
    for item in preview.get("analysis_element_details", ()):
        i, j = int(item["i"]), int(item["j"])
        if i not in coordinates or j not in coordinates:
            raise ValueError(f"分析单元 {item.get('element')} 引用了没有坐标的节点")
        block = _polyline(np.vstack([coordinates[i], coordinates[j]]))
        block.cell_data["analysis_element"] = [int(item["element"])]
        block.cell_data["physical_member"] = [int(item["physical_member"])]
        blocks.append(block)
    if not blocks:
        return pv.PolyData()
    return blocks[0].merge(blocks[1:], merge_points=False)


def analysis_split_points(preview: dict) -> pv.PolyData:
    """自动生成或复用的内部切分节点。"""
    details = list(preview.get("split_node_details", ()))
    points = np.asarray([item["coordinates"] for item in details], dtype=float)
    if not len(points):
        points = np.empty((0, 3), dtype=float)
    mesh = pv.PolyData(points)
    if details:
        mesh.point_data["node"] = [int(item["node"]) for item in details]
        mesh.point_data["generated"] = [
            item.get("origin") == "generated_for_point_load" for item in details]
    return mesh


def member_polylines(frame, solution=None, case: str | None = None,
                     scale: float = 0.0, stations: int = STATIONS,
                     scalars: str | None = None) -> pv.PolyData:
    """所有杆件连成一个 PolyData，每根杆 `stations` 个点。

    scale > 0 时点按变形后位置放，**而且用的是单元内解析挠度**——
    只按节点位移画的话，单跨一个单元的梁会画成一条直线，跨中那个鼓看不见。
    这一点和内力图那边是同一个道理。

    scalars 给了分量名（"N"/"Mz"…）就把该分量逐点挂上去，供云图着色。
    """
    blocks: list[pv.PolyData] = []
    name = case
    if solution is not None and name is None:
        name = solution.primary

    for mid in sorted(frame.members):
        member = frame.members[mid]
        pi, pj = member_endpoints(frame, member)
        diagram = None
        if scalars and solution is not None:
            diagram = member_diagram(frame, solution, mid, name,
                                     stations=stations)
            # 剪力/轴力在跨内集中力处有真实跳跃。若先把结果重采样回等距点，
            # 跳跃会被抹成一段斜坡，云图读出来就像荷载在一小段内分布。
            # 直接沿内力恢复点建线，保留荷载点两侧的站位。
            station_x = diagram.x
            t = (station_x / diagram.length)[:, None]
        else:
            station_x = np.linspace(0.0, float(np.linalg.norm(pj - pi)),
                                    stations)
            t = np.linspace(0.0, 1.0, stations)[:, None]
        pts = pi + t * (pj - pi)

        if scale and solution is not None:
            # 挠度用**高分辨率**积分再插到显示点上。两件事的分辨率要求不同：
            # 21 个点足够画出一条光滑的曲线，但拿 21 个点做两次梯形积分，
            # 跨中挠度会差 0.4%——形状看着没问题，数是错的
            x, recovered = member_displacement(
                frame, solution, mid, name, DEFLECTION_STATIONS)
            displacement = np.column_stack([
                np.interp(station_x, x, recovered[:, k]) for k in range(3)])
            pts = pts + scale * displacement

        block = _polyline(pts)
        if diagram is not None:
            block[scalars] = diagram.component(scalars)
        block["member"] = np.full(len(pts), mid)
        blocks.append(block)

    # 人工建模早期可能只有节点、还没有杆件：此时 blocks 为空，
    # 返回一个空 PolyData 而不是 blocks[0] 越界崩溃
    if not blocks:
        return pv.PolyData()
    # 杆端截面力属于各杆件的局部结果，节点处允许跳变（不同局部轴、端释放、
    # 节点集中荷载都会造成差异）。PyVista 默认合并重合点，会把相邻杆的杆端
    # 标量覆盖/混合成一个值。Abaqus 的 beam section forces 也是按单元端分别
    # 存储，因此这里必须保留每根杆自己的端点。
    merged = (blocks[0].merge(blocks[1:], merge_points=False)
              if len(blocks) > 1 else blocks[0])
    return merged


def member_tubes(frame, solution=None, case: str | None = None,
                 scale: float = 0.0, scalars: str | None = None,
                 radius: float | None = None) -> pv.PolyData:
    """杆件管。radius 留空按模型尺寸取。"""
    line = member_polylines(frame, solution, case, scale, scalars=scalars)
    r = radius if radius is not None else TUBE_RATIO * model_size(frame)
    return line.tube(radius=r, n_sides=32, capping=True)


def support_glyphs(frame, size: float | None = None) -> dict[str, pv.PolyData]:
    """支座符号，按类型分组。固接用立方体、铰接用锥、部分约束用小球。

    和 Streamlit 版同一套分类（`classify_support`），也同一条取舍：
    **只画约束住两个及以上平动方向的节点**——平面刚架的面外约束每个节点都有，
    全画出来会把真正的柱脚淹掉。
    """
    r = SYMBOL_RATIO * (size or model_size(frame))
    groups: dict[str, list[pv.PolyData]] = {}
    for nid, mask in frame.supports.items():
        if sum(mask[:3]) < 2:
            continue
        p = np.asarray(frame.nodes[nid].xyz, dtype=float)
        kind = classify_support(mask)
        if kind == "固接":
            g = pv.Cube(center=p - [0, 0, r], x_length=2.2 * r,
                        y_length=2.2 * r, z_length=2 * r)
        elif kind == "铰接":
            g = pv.Cone(center=p - [0, 0, r], direction=[0, 0, 1],
                        height=2 * r, radius=1.2 * r, resolution=32)
        else:
            g = pv.Sphere(radius=0.9 * r, center=p,
                          theta_resolution=24, phi_resolution=24)
        groups.setdefault(kind, []).append(g)
    return {k: (v[0].merge(v[1:]) if len(v) > 1 else v[0])
            for k, v in groups.items()}


def support_labels(frame, size: float | None = None) -> tuple[list, list[str]]:
    """返回主支座的精确约束自由度标注。

    锥体、方块只能表达支座的概略类别，不能表达 ``U1/U2/U3/UR1/UR2/UR3``
    的真实组合。标签与 :func:`support_glyphs` 使用同一可见性规则，避免把平面
    模型每个节点上的面外稳定约束全部铺到画面上；被画出的主支座则逐项写清。
    """
    offset = 0.024 * (size or model_size(frame))
    dofs = ("U1", "U2", "U3", "UR1", "UR2", "UR3")
    points: list = []
    texts: list[str] = []
    for nid, mask in frame.supports.items():
        if sum(mask[:3]) < 2 or nid not in frame.nodes:
            continue
        fixed = " ".join(name for name, restrained in zip(dofs, mask)
                         if restrained)
        if not fixed:
            continue
        p = np.asarray(frame.nodes[nid].xyz, dtype=float)
        points.append(p + np.array([offset, offset, offset]))
        texts.append(f"N{nid} BC: {fixed}")
    return points, texts


def member_local_axis_glyphs(frame, member_id: int,
                             size: float | None = None) -> dict[str, pv.PolyData]:
    """在杆件中点生成局部 x/y/z 右手坐标箭头。"""
    member = frame.members.get(member_id)
    if member is None:
        return {}
    pi, pj = member_endpoints(frame, member)
    _, rotation = local_axes(pi, pj, member.ref_vector)
    origin = 0.5 * (pi + pj)
    length = 0.075 * (size or model_size(frame))
    return {
        axis: pv.Arrow(start=origin, direction=direction, scale=length,
                       tip_length=0.24, tip_radius=0.075,
                       shaft_radius=0.018, tip_resolution=24,
                       shaft_resolution=16)
        for axis, direction in zip(("x", "y", "z"), rotation)
    }


def member_local_axis_labels(frame, member_id: int,
                             size: float | None = None) -> tuple[list, list[str]]:
    """局部轴箭头端点和标签；与 ``member_local_axis_glyphs`` 共用尺度。"""
    member = frame.members.get(member_id)
    if member is None:
        return [], []
    pi, pj = member_endpoints(frame, member)
    _, rotation = local_axes(pi, pj, member.ref_vector)
    origin = 0.5 * (pi + pj)
    length = 0.088 * (size or model_size(frame))
    return [origin + length * direction for direction in rotation], [
        "local x (i->j)", "local y", "local z"]


def load_arrows(frame, case: str, size: float | None = None,
                per_member: int = 7) -> dict[str, pv.PolyData]:
    """荷载与分析步给定位移箭头，按物理类型分组。

    **各组分别归一**，与 Streamlit 版同一条理由：物理量量级常跨几个数量级，
    共用一把尺子会让小的那类彻底消失，而"看不见"和"不存在"在图上分不出来。
    """
    from frame3d import span_loads_of
    from span_loads import POINT

    load_case = frame.load_cases.get(case)
    if load_case is None:
        return {}
    span = size or model_size(frame)

    def arrows(points, vectors, groups=None) -> pv.PolyData | None:
        if not points:
            return None
        mag = np.linalg.norm(np.array(vectors), axis=1)
        peak = float(mag.max()) or 1.0
        made, tails = [], []
        for p, v, m in zip(points, vectors, mag):
            # 反对称梯形荷载可能恰好在某个显示站点过零。零向量不能传给
            # pv.Arrow；VTK 会归一化它并产生 NaN，严重时在渲染线程原生崩溃。
            if m <= 1e-15 * peak:
                continue
            length = span * 0.06 * (0.45 + 0.55 * m / peak)
            direction = np.asarray(v, dtype=float) / m
            # 箭头尾端接在作用点上，箭头指向荷载方向
            tail = np.asarray(p) - direction * length
            tails.append(tail)
            made.append(pv.Arrow(start=tail,
                                 direction=direction, scale=length,
                                 tip_length=0.26, tip_radius=0.06,
                                 shaft_radius=0.014,
                                 tip_resolution=24, shaft_resolution=16))
        if groups is not None and len(tails) == len(groups):
            group_values = np.asarray(groups)
            tail_values = np.asarray(tails)
            for group in dict.fromkeys(groups):
                rail = tail_values[group_values == group]
                if len(rail) >= 2:
                    made.append(_polyline(rail).tube(
                        radius=span * 0.0011, n_sides=20, capping=True))
        if not made:
            return None
        return made[0].merge(made[1:]) if len(made) > 1 else made[0]

    def circular_arrows(points, vectors) -> pv.PolyData | None:
        """按右手定则画力矩/转角；轴向量决定旋向，不再冒充直线力。"""
        if not points:
            return None
        magnitudes = np.linalg.norm(np.asarray(vectors, dtype=float), axis=1)
        peak = float(magnitudes.max()) or 1.0
        made: list[pv.PolyData] = []
        for point, vector, magnitude in zip(points, vectors, magnitudes):
            if magnitude <= 1e-15 * peak:
                continue
            axis = np.asarray(vector, dtype=float) / magnitude
            reference = (np.array([0.0, 0.0, 1.0])
                         if abs(axis[2]) < 0.85 else np.array([0.0, 1.0, 0.0]))
            radial = np.cross(reference, axis)
            radial /= np.linalg.norm(radial)
            tangent_basis = np.cross(axis, radial)
            radius = span * 0.034 * (0.72 + 0.28 * magnitude / peak)
            theta = np.linspace(-0.78 * np.pi, 0.78 * np.pi, 34)
            center = np.asarray(point, dtype=float)
            arc_points = center + radius * (
                np.cos(theta)[:, None] * radial
                + np.sin(theta)[:, None] * tangent_basis)
            arc = _polyline(arc_points).tube(
                radius=span * 0.00125, n_sides=16, capping=True)
            end = arc_points[-1]
            tangent = (-np.sin(theta[-1]) * radial
                       + np.cos(theta[-1]) * tangent_basis)
            head_height = radius * 0.36
            head = pv.Cone(center=end + 0.5 * head_height * tangent,
                           direction=tangent, height=head_height,
                           radius=radius * 0.105, resolution=24)
            made.append(arc.merge(head))
        if not made:
            return None
        return made[0].merge(made[1:]) if len(made) > 1 else made[0]

    nodal_p, nodal_v, moment_p, moment_v = [], [], [], []
    span_p, span_v, span_groups = [], [], []
    given_p, given_v, rotation_p, rotation_v = [], [], [], []
    for nid, load in load_case.nodal_loads.items():
        f = np.asarray(load[:3], dtype=float)
        if np.linalg.norm(f) > 0 and nid in frame.nodes:
            nodal_p.append(frame.nodes[nid].xyz)
            nodal_v.append(f)
        moment = np.asarray(load[3:], dtype=float)
        if np.linalg.norm(moment) > 0 and nid in frame.nodes:
            moment_p.append(frame.nodes[nid].xyz)
            moment_v.append(moment)
    for nid, values in load_case.settlements.items():
        if nid not in frame.nodes:
            continue
        translation = np.asarray(values[:3], dtype=float)
        if np.linalg.norm(translation) > 0:
            given_p.append(frame.nodes[nid].xyz)
            given_v.append(translation)
        rotation = np.asarray(values[3:], dtype=float)
        if np.linalg.norm(rotation) > 0:
            rotation_p.append(frame.nodes[nid].xyz)
            rotation_v.append(rotation)
    group = 0
    for mid, member in frame.members.items():
        pi, pj = member_endpoints(frame, member)
        for item in span_loads_of(load_case, mid):
            if item.kind == POINT:
                L = float(np.linalg.norm(pj - pi))
                span_p.append(pi + (item.a / L if L else 0.0) * (pj - pi))
                span_v.append(np.asarray(item.w1, dtype=float))
                span_groups.append(None)
                continue
            group += 1
            v1, v2 = item.ends()
            for t in np.linspace(0.15, 0.85, per_member):
                span_p.append(pi + t * (pj - pi))
                span_v.append((1 - t) * v1 + t * v2)
                span_groups.append(group)

    out = {}
    for label, pts, vec in (("节点荷载", nodal_p, nodal_v),
                            ("节点力矩", moment_p, moment_v),
                            ("杆间荷载", span_p, span_v),
                            ("给定位移", given_p, given_v),
                            ("给定转角", rotation_p, rotation_v)):
        groups = span_groups if label == "杆间荷载" else None
        # 集中杆荷载没有连续顶线；用 None 将它们从分布荷载组中排除。
        if groups is not None and any(value is None for value in groups):
            groups = [value if value is not None else f"point-{index}"
                      for index, value in enumerate(groups)]
        got = (circular_arrows(pts, vec)
               if label in {"节点力矩", "给定转角"}
               else arrows(pts, vec, groups))
        if got is not None:
            out[label] = got
    return out


# 云图色标的默认分位上限。
#
# 刚架的内力分布是重尾的：实测一个两层两跨框架的合弯矩，**55% 的样点落在
# 色带最底下的 20%**，而色带上半段只服务约 1% 的结构（p95 只有峰值的 46%）。
# 满量程画出来，绝大多数构件是同一个颜色，云图等于没有信息。
#
# 所以默认按分位裁剪。**这是对显示的一次人为压缩，必须写在图上**——
# 调用方拿 clim_is_clipped() 判断后，把"上限取 pXX，超出饱和"标进色标标题。
# 峰值本身不受影响，仍由标题、查询和 Result DB 如实给出。
# 传 percentile=None 恢复满量程。
CONTOUR_PERCENTILE = 95.0


def _bound(values: np.ndarray, percentile: float | None) -> float:
    magnitude = np.abs(np.asarray(values, dtype=float))
    if magnitude.size == 0:
        return 0.0
    if percentile is None:
        return float(magnitude.max())
    return float(np.percentile(magnitude, percentile))


def clim_is_clipped(mesh: pv.PolyData, name: str,
                    clim: tuple[float, float]) -> bool:
    """色标上限是否低于真实峰值——即图上是否有被饱和掉的部分。"""
    values = np.abs(np.asarray(mesh[name], dtype=float))
    if not values.size:
        return False
    return float(values.max()) > max(abs(clim[0]), abs(clim[1])) * (1.0 + 1e-9)


def contour_caption(component: str, unit: str, clipped: bool = False,
                    sign_filter: str = "all") -> str:
    """梁中心线结果的自描述文字；明确它不是截面实体应力云图。"""
    convention = ("magnitude (nonnegative)" if component in {"M", "V"}
                  else "signed in member local axes")
    suffix = f" | display clipped at p{CONTOUR_PERCENTILE:.0f}" if clipped else ""
    if sign_filter != "all" and component not in {"M", "V"}:
        suffix += f" | {sign_filter} values only"
    return ("BEAM CENTERLINE INTERNAL-FORCE RESULT\n"
            f"{component} [{unit}] | {convention}{suffix}")


def sign_filtered(values, mode: str) -> np.ndarray:
    """保留结果数组尺寸，仅把未选符号压到零色，底层结果不变。"""
    data = np.asarray(values, dtype=float).copy()
    if mode == "positive":
        data[data < 0.0] = 0.0
    elif mode == "negative":
        data[data > 0.0] = 0.0
    elif mode != "all":
        raise ValueError("符号筛选只能是 all / positive / negative")
    return data


def symmetric_clim(mesh: pv.PolyData, name: str,
                   percentile: float | None = CONTOUR_PERCENTILE
                   ) -> tuple[float, float]:
    """发散色标的取值范围必须**关于零对称**。

    否则全受压的结构里，最大值（最轻的压力，接近 0）会落在红色端——
    而红色在我们的约定里是受拉。看图的人会把一根轻微受压的杆读成受拉。
    这不是"不好看"，是读错。

    全零时给一个 ±1 的兜底范围，免得 VTK 拿到零跨度的 clim。
    """
    peak = _bound(mesh[name], percentile)
    return (-peak, peak) if peak > 0 else (-1.0, 1.0)


def contour_clim(mesh: pv.PolyData, name: str,
                 percentile: float | None = CONTOUR_PERCENTILE
                 ) -> tuple[float, float]:
    """按结果物理含义给色标范围。

    N/Vy/Vz/T/My/Mz 是有符号局部分量，要用关于零对称的发散色标；
    V/M 是合量，天然非负，范围应从 0 开始。把合量也画成 ±peak 会浪费
    一半色带，还暗示一个并不存在的正负号。

    上限默认取分位数而不是峰值，理由见 CONTOUR_PERCENTILE。
    """
    if name not in {"V", "M"}:
        return symmetric_clim(mesh, name, percentile)
    peak = _bound(mesh[name], percentile)
    return (0.0, peak) if peak > 0 else (0.0, 1.0)


def node_points(frame) -> pv.PolyData:
    """全部节点的点云。**节点编号跟着点走**——拾取时要靠它回答"点中的是几号"。"""
    ids = list(frame.order())
    cloud = pv.PolyData(np.array([frame.nodes[n].xyz for n in ids], dtype=float))
    cloud["node"] = np.array(ids, dtype=int)
    return cloud


def displaced_node_points(frame, vector: np.ndarray, scale: float) -> pv.PolyData:
    """按全局结果向量移动后的共享节点点云，用小球遮住管端的渲染接缝。"""
    ids = list(frame.order())
    pts = np.array([frame.nodes[n].xyz
                    + scale * vector[frame.node_dofs(n)[:3]] for n in ids])
    cloud = pv.PolyData(pts)
    cloud["node"] = np.asarray(ids, dtype=int)
    return cloud


def rigid_link_polylines(frame, solution=None, case: str | None = None,
                         scale: float = 0.0, mode_vector: np.ndarray | None = None
                         ) -> pv.PolyData:
    """节点与偏移梁端之间的刚域连线；没有偏移时返回空网格。"""
    blocks = []
    for mid in sorted(frame.members):
        member = frame.members[mid]
        end_points = member_endpoints(frame, member)
        recovered = None
        if solution is not None and scale:
            _, recovered = member_displacement(
                frame, solution, mid, case, DEFLECTION_STATIONS)
        elif mode_vector is not None and scale:
            shapes = np.asarray(mode_vector, dtype=float).reshape(-1, 1)
            _, recovered = member_mode_displacement(frame, shapes, 0, mid, 2)
        for end, nid in enumerate((member.i, member.j)):
            p_node = frame.nodes[nid].xyz.copy()
            p_end = end_points[end].copy()
            if np.linalg.norm(p_end - p_node) <= 1e-14:
                continue
            if scale:
                vector = (solution[case or solution.primary].U
                          if solution is not None else mode_vector)
                p_node += scale * vector[frame.node_dofs(nid)[:3]]
                p_end += scale * recovered[0 if end == 0 else -1]
            blocks.append(_polyline(np.vstack([p_node, p_end])))
    if not blocks:
        return pv.PolyData()
    return (blocks[0].merge(blocks[1:], merge_points=False)
            if len(blocks) > 1 else blocks[0])


def hinge_glyphs(frame, scale: float | None = None) -> pv.PolyData:
    """杆端转动释放符号。球心位于可变形梁端，避免与支座铰混淆。"""
    radius = 0.009 * (scale or model_size(frame))
    made = []
    for member in frame.members.values():
        pi, pj = member_endpoints(frame, member)
        if any(name in {"rx", "ry", "rz"} for name in member.releases_i):
            made.append(pv.Sphere(radius=radius, center=pi, theta_resolution=12,
                                  phi_resolution=12))
        if any(name in {"rx", "ry", "rz"} for name in member.releases_j):
            made.append(pv.Sphere(radius=radius, center=pj, theta_resolution=12,
                                  phi_resolution=12))
    if not made:
        return pv.PolyData()
    return made[0].merge(made[1:]) if len(made) > 1 else made[0]


def node_labels(frame) -> tuple[np.ndarray, list[str]]:
    """节点编号标注：位置 + 文字。

    没有编号标注，报错里那句"节点 17 缺少约束"在三维视图里就是**无法定位的**。
    这是"这个软件回答不了在哪"最直接的一处。
    """
    ids = list(frame.order())
    pts = np.array([frame.nodes[n].xyz for n in ids], dtype=float)
    return pts, [str(n) for n in ids]


def member_labels(frame) -> tuple[np.ndarray, list[str]]:
    """杆件编号标注，挂在杆件中点。"""
    ids = sorted(frame.members)
    pts = np.array([(frame.nodes[frame.members[m].i].xyz
                     + frame.nodes[frame.members[m].j].xyz) / 2.0
                    for m in ids], dtype=float)
    return pts, [str(m) for m in ids]


def _point_to_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """点到线段的距离。**是线段不是直线**——用直线的话，
    远处一根杆的延长线从你点的位置附近穿过，就会被误判成最近的。"""
    ab = b - a
    denom = float(ab @ ab)
    if denom <= 0.0:
        return float(np.linalg.norm(p - a))
    tt = float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + tt * ab)))


def nearest(frame, point: np.ndarray, kind: str,
            tolerance: float = 0.1) -> int | None:
    """离拾取点最近的节点或杆件编号。

    **用几何最近，不用 VTK 的 cell id。** 管子是三角化过的，一根杆有几十个
    三角面，cell id 和杆件编号没有任何对应关系；而"离我点的地方最近"这个判据
    在任何网格上都成立，也正好是用户心里想的那件事。

    `tolerance` 是模型尺寸的比例。点得太偏就返回 None——**宁可没选中，
    也不要选中一个用户没想选的东西**，后者会让人以为软件在乱跳。
    """
    size = model_size(frame) or 1.0
    limit = tolerance * size
    best, best_d = None, float("inf")
    if kind == "node":
        for nid in frame.order():
            d = float(np.linalg.norm(point - frame.nodes[nid].xyz))
            if d < best_d:
                best, best_d = nid, d
    else:
        for mid in sorted(frame.members):
            m = frame.members[mid]
            pi, pj = member_endpoints(frame, m)
            d = _point_to_segment(point, pi, pj)
            if d < best_d:
                best, best_d = mid, d
    return best if best_d <= limit else None


def draft_frame(model: dict) -> Frame:
    """从原始 payload **宽松**装配一个只用于显示与拾取的几何 Frame。

    正式的 preview_frame 要先过 schema 校验、要求材料/截面/杆件齐全；
    人工建模的最初阶段往往只点了几个节点，那一整套还没建——若因此清空视口，
    用户就看不到自己刚点的节点、也没法点中它们去连杆件。这里跳过一切求解
    相关校验，能装多少装多少：有节点就显示节点，端点齐全的杆件才显示。
    """
    f = Frame(units=str(model.get("units", "N-m-Pa")))
    for n in model.get("nodes") or []:
        try:
            nid = int(n["id"])
            f.nodes[nid] = Node(nid, float(n["x"]), float(n["y"]), float(n["z"]))
        except (KeyError, TypeError, ValueError):
            continue
    for m in model.get("members") or []:
        try:
            mid, i, j = int(m["id"]), int(m["i"]), int(m["j"])
        except (KeyError, TypeError, ValueError):
            continue
        if i not in f.nodes or j not in f.nodes:
            continue  # 端点还没建出来，先不画这根杆
        f.members[mid] = Member(mid, i, j,
                                str(m.get("section", "")),
                                str(m.get("material", "")))
    for s in model.get("supports") or []:
        try:
            f.supports[int(s["node"])] = tuple(int(v) for v in s["fix"])
        except (KeyError, TypeError, ValueError):
            continue
    return f


def highlight_members(frame, member_ids, radius_scale: float = 1.6) -> pv.PolyData:
    """把选中的杆件加粗画一遍，套在原杆件外面。

    **用加粗而不是换颜色**：云图模式下颜色是有含义的（受拉受压），
    选中再占一个颜色，图就没法读了。加粗在任何显示模式下都不冲突。
    """
    ids = [m for m in member_ids if m in frame.members]
    if not ids:
        return pv.PolyData()
    r = radius_scale * TUBE_RATIO * model_size(frame)
    blocks = []
    for mid in ids:
        m = frame.members[mid]
        blocks.append(_polyline(np.vstack(member_endpoints(frame, m))))
    merged = blocks[0].merge(blocks[1:]) if len(blocks) > 1 else blocks[0]
    return merged.tube(radius=r, n_sides=32, capping=True)


def highlight_nodes(frame, node_ids) -> pv.PolyData:
    ids = [n for n in node_ids if n in frame.nodes]
    if not ids:
        return pv.PolyData()
    return pv.PolyData(np.array([frame.nodes[n].xyz for n in ids], dtype=float))


def mode_shape_tubes(frame, shapes: np.ndarray, mode: int, scale: float,
                     radius: float | None = None) -> pv.PolyData:
    """振型或失稳模态管；使用梁形函数恢复单元内曲线与释放端转角。"""
    blocks = []
    r = radius if radius is not None else TUBE_RATIO * model_size(frame)
    for mid in sorted(frame.members):
        points, displacement = member_mode_displacement(
            frame, shapes, mode, mid, STATIONS)
        blocks.append(_polyline(points + scale * displacement))
    merged = (blocks[0].merge(blocks[1:], merge_points=False)
              if len(blocks) > 1 else blocks[0])
    return merged.tube(radius=r, n_sides=32, capping=True)


def load_labels(frame, case: str, size: float | None = None) -> tuple[list, list]:
    """每个荷载的标注点与文字（含带符号全局分量、数值和单位）。

    **类型不该只靠颜色分。** 四类荷载用四种颜色，跑 validate_palette.js 的
    all-pairs 最严档必然不合格（文档写明超过三槽就该换编码方式）；而且颜色
    只回答"这是哪一类"，回答不了"多大"——工程师真正要看的是后者。
    直接把数值标在旁边，类型和大小一次说清，比任何配色方案都直接。

    文字用 ASCII：VTK 有自己的字体引擎，默认字体没有中文字形。
    """
    from frame3d import span_loads_of
    from span_loads import POINT, TRAPEZOID
    from units import of as unit_system

    load_case = frame.load_cases.get(case)
    if load_case is None:
        return [], []
    system = unit_system(frame)
    span = 0.045 * (size or model_size(frame))
    points: list = []
    texts: list[str] = []

    def vector_text(prefix: str, vector, scale: float, unit: str) -> str:
        values = np.asarray(vector, dtype=float) * scale
        terms = [f"{prefix}{axis}={value:+.3g}"
                 for axis, value in zip("xyz", values) if value != 0.0]
        return f"{', '.join(terms)} {unit} [global]" if terms else ""

    def add(point, text: str) -> None:
        if not text:
            return
        points.append(np.asarray(point, dtype=float) + np.array([0.0, 0.0, span]))
        texts.append(text)

    for node_id, load in load_case.nodal_loads.items():
        if node_id not in frame.nodes:
            continue
        p = frame.nodes[node_id].xyz
        add(p, vector_text("F", load[:3], system.force_scale,
                           system.force_unit))
        add(p, vector_text("M", load[3:], system.moment_scale,
                           system.moment_unit))

    for member_id, member in frame.members.items():
        pi, pj = member_endpoints(frame, member)
        for load in span_loads_of(load_case, member_id):
            if load.kind == POINT:
                length = float(np.linalg.norm(pj - pi)) or 1.0
                where = pi + (float(load.a) / length) * (pj - pi)
                add(where, vector_text("P", load.w1, system.force_scale,
                                       system.force_unit))
            elif load.kind == TRAPEZOID:
                first = vector_text("w1", load.w1, system.line_load_scale,
                                    system.line_load_unit)
                second = vector_text("w2", load.w2, system.line_load_scale,
                                     system.line_load_unit)
                # 单位与坐标系只写一次，避免长标签重复两遍。
                first = first.removesuffix(
                    f" {system.line_load_unit} [global]")
                add(0.5 * (pi + pj), f"{first} -> {second}")
            else:
                add(0.5 * (pi + pj), vector_text(
                    "w", load.w1, system.line_load_scale,
                    system.line_load_unit))
    return points, texts
