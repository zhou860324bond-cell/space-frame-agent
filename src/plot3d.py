"""三维刚架绘图：变形图与内力图。

**图给人看，数给模型看。** 绘图函数只负责出图并返回文件路径加一份数值摘要；
Agent 的结论必须来自 query_results 的数字，不能从图里读。这条界线是刻意的——
让多模态模型去"看"云图再下结论，等于把可验证的数值换成了像素猜测。

中文字体按机器可用情况自动挑选，挑不到就退回英文标签，避免出现方块。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")                     # 无界面环境也能出图
import matplotlib.pyplot as plt           # noqa: E402
import numpy as np                        # noqa: E402
from matplotlib import font_manager       # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: E402

import viz_theme as T                     # noqa: E402  报告图与界面共用一套配色
from units import of as unit_system       # noqa: E402

_CJK_CANDIDATES = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Noto Sans CJK JP",
                   "Source Han Sans SC", "WenQuanYi Zen Hei", "PingFang SC", "Heiti SC")


def _pick_font() -> str | None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CJK_CANDIDATES:
        if name in available:
            return name
    return None


_FONT = _pick_font()
if _FONT:
    plt.rcParams["font.sans-serif"] = [_FONT]
    plt.rcParams["axes.unicode_minus"] = False


def _t(zh: str, en: str) -> str:
    """有中文字体就用中文，没有就退回英文，别画出一排方块。"""
    return zh if _FONT else en


def _segments(frame, displacement: np.ndarray | None = None, scale: float = 0.0):
    """返回每根杆件两端的坐标，displacement 给定时叠加放大后的位移。"""
    segs = []
    for m in frame.members.values():
        pts = []
        for nid in (m.i, m.j):
            p = frame.nodes[nid].xyz
            if displacement is not None and scale:
                d = frame.node_dofs(nid)
                p = p + scale * displacement[d[:3]]
            pts.append(p)
        segs.append(pts)
    return segs


def _deformed_segments(frame, solution, case: str, scale: float,
                       stations: int = 41):
    """按杆件中心线恢复结果画变形，不把每根梁画成两节点直线。"""
    from frame3d import local_axes, member_endpoints
    from internal_forces import member_displacement

    segs = []
    for mid in sorted(frame.members):
        member = frame.members[mid]
        x, displacement = member_displacement(
            frame, solution, mid, case, stations=stations)
        pi, pj = member_endpoints(frame, member)
        _, rot = local_axes(pi, pj, member.ref_vector)
        base = pi[None, :] + x[:, None] * rot[0][None, :]
        segs.append(base + scale * displacement)
    return segs


def _autoscale(frame, solution, case: str) -> float:
    """把杆件中心线最大位移放大到结构尺寸的 5%。"""
    from internal_forces import max_centerline_displacement

    size = 0.0
    coords = np.array([n.xyz for n in frame.nodes.values()])
    if len(coords):
        size = float(np.max(coords.max(axis=0) - coords.min(axis=0)))
    peak = float(max_centerline_displacement(
        frame, solution, case, stations=201)["value"])
    # 1.0 是"不放大"，和 safe_scale 的 0.0（压平）不同：变形图没有可见
    # 位移时按原尺寸画整个结构才对，压平会把模型也压没。
    return T.safe_scale(size, peak, 0.05) or 1.0


def _physical_location(frame, mapping, element_id: int,
                       element_x: float) -> tuple[int, float]:
    """把分析单元局部位置换回物理构件位置。"""
    if mapping is None:
        return element_id, element_x
    from internal_forces import member_length
    physical_id = mapping.physical_member_id(element_id)
    offset = 0.0
    for candidate in mapping.element_ids(physical_id):
        if candidate == element_id:
            break
        offset += member_length(frame, candidate)
    return physical_id, offset + element_x


def _model_size(frame) -> float:
    coords = np.array([n.xyz for n in frame.nodes.values()])
    if not len(coords):
        return 1.0
    return float(np.max(coords.max(axis=0) - coords.min(axis=0))) or 1.0


def _equalize(ax, frame) -> None:
    """按真实比例设视图。三个方向各自取数据范围，再用 box_aspect 保住比例，
    否则一个 18x6x7.2 的框架会被塞进立方体里，看着像另一个结构。

    **平面模型要特判**：所有节点 y 相同时该方向跨度为零，matplotlib 会在一个
    退化区间里硬塞十几个刻度，标签互相压成一团墨。这类模型占绝大多数，
    所以给退化方向一个对称的小范围并**关掉它的刻度**——那个方向本来也没有信息。
    """
    coords = np.array([n.xyz for n in frame.nodes.values()])
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    span = hi - lo
    reach = float(span.max()) or 1.0
    flat = span <= 1e-9 * reach                     # 退化（平面/直线模型）
    pad = 0.06 * reach
    lo = np.where(flat, lo - 0.05 * reach, lo - pad)
    hi = np.where(flat, hi + 0.05 * reach, hi + pad)

    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(tuple(np.where(flat, 0.12 * reach, hi - lo)))

    length = unit_system(frame).length
    for index, (axis, name) in enumerate(
            ((ax.xaxis, "X"), (ax.yaxis, "Y"), (ax.zaxis, "Z"))):
        if flat[index]:
            axis.set_ticks([])
            axis.set_label_text("")
        else:
            axis.set_label_text(f"{name} ({length})")
        axis.label.set_color(T.INK_SECONDARY)
        axis.label.set_fontsize(9)
        axis.set_tick_params(colors=T.INK_MUTED, labelsize=8)
        axis.line.set_color(T.AXIS)
        axis._axinfo["grid"].update(color=T.GRID, linewidth=0.6)
        axis.pane.set_facecolor(T.SURFACE)
        axis.pane.set_edgecolor(T.GRID)
        axis.pane.set_alpha(1.0)
    ax.view_init(elev=20, azim=-60)


# --------------------------------------------------------------- 符号

def _classify_member(frame, member) -> str:
    """按几何方向分梁/柱/斜杆。**只影响线宽和图例，不参与任何计算。**"""
    from frame3d import member_endpoints
    pi, pj = member_endpoints(frame, member)
    direction = pj - pi
    length = float(np.linalg.norm(direction))
    if length <= 0.0:
        return "梁"
    vertical = abs(float(direction[2])) / length
    if vertical > 0.94:
        return "柱"
    if vertical < 0.06:
        return "梁"
    return "斜杆"


def _support_kind(mask) -> str | None:
    """支座分类。**只画约束住两个及以上平动方向的节点**——平面刚架的面外约束
    每个节点都有，全画出来会把真正的柱脚淹掉。这条与桌面端同一取舍。
    """
    translations = sum(mask[:3])
    if translations < 2:
        return None
    if translations == 3:
        return "固接" if sum(mask[3:]) == 3 else "铰接"
    return "滚动"


_SUPPORT_MARKER = {"固接": "s", "铰接": "^", "滚动": "o"}


def _draw_supports(ax, frame) -> dict[str, Any]:
    """支座符号：形状分类型，统一用支座色。返回出现过的类型，供图例用。"""
    size = _model_size(frame)
    seen: dict[str, Any] = {}
    for nid, mask in frame.supports.items():
        kind = _support_kind(mask)
        if kind is None or nid not in frame.nodes:
            continue
        p = np.asarray(frame.nodes[nid].xyz, dtype=float)
        p = p - np.array([0.0, 0.0, 0.018 * size])       # 落在节点下方
        ax.scatter(*p, marker=_SUPPORT_MARKER[kind], s=52,
                   c=T.SUPPORT, edgecolors="none", zorder=6)
        seen[kind] = _SUPPORT_MARKER[kind]
    return seen


def _draw_hinges(ax, frame) -> bool:
    """杆端释放：在释放端画一个空心圈，工程习惯就是这么标铰。"""
    from frame3d import member_endpoints
    size = _model_size(frame)
    drawn = False
    for member in frame.members.values():
        pi, pj = member_endpoints(frame, member)
        direction = pj - pi
        length = float(np.linalg.norm(direction)) or 1.0
        unit = direction / length
        inset = min(0.035 * size, 0.18 * length)
        for releases, anchor, sign in ((member.releases_i, pi, 1.0),
                                       (member.releases_j, pj, -1.0)):
            if not releases:
                continue
            p = anchor + sign * inset * unit
            ax.scatter(*p, marker="o", s=34, facecolors="none",
                       edgecolors=T.HINGE, linewidths=1.2, zorder=6)
            drawn = True
    return drawn


def _draw_loads(ax, frame, case: str) -> dict[str, str]:
    """荷载符号。集中力一支箭头，力矩一个双圈标记，杆间分布一排短箭头。

    **各类分别归一**：物理量量级常跨几个数量级，共用一把尺子会让小的那类
    彻底消失，而"看不见"和"不存在"在图上分不出来。这条与桌面端一致。
    """
    from frame3d import member_endpoints, span_loads_of
    load_case = frame.load_cases.get(case)
    if load_case is None:
        return {}
    size = _model_size(frame)
    span = 0.10 * size
    seen: dict[str, str] = {}

    def arrows(points, vectors, width):
        if not points:
            return
        magnitude = np.linalg.norm(vectors, axis=1)
        peak = float(magnitude.max()) or 1.0
        for p, v, m in zip(points, vectors, magnitude, strict=True):
            if m <= 1e-12 * peak:
                continue
            step = span * (0.45 + 0.55 * m / peak) * np.asarray(v) / m
            ax.quiver(*(np.asarray(p) - step), *step, color=T.LOAD,
                      linewidth=width, arrow_length_ratio=0.32, zorder=5)

    points, vectors, moments = [], [], []
    for nid, load in load_case.nodal_loads.items():
        if nid not in frame.nodes:
            continue
        p = frame.nodes[nid].xyz
        force = np.asarray(load[:3], dtype=float)
        if float(np.linalg.norm(force)) > 0:
            points.append(p)
            vectors.append(force)
        if float(np.linalg.norm(np.asarray(load[3:], dtype=float))) > 0:
            moments.append(p)
    if points:
        arrows(points, np.array(vectors), 1.8)
        seen["节点集中力"] = "arrow"
    for p in moments:
        ax.scatter(*p, marker="P", s=64, c=T.LOAD, edgecolors="none", zorder=6)
        seen["节点力矩"] = "P"

    uniform_points, uniform_vectors = [], []
    point_points, point_vectors = [], []
    for mid, member in frame.members.items():
        pi, pj = member_endpoints(frame, member)
        w = load_case.member_loads.get(mid)
        if w is not None and float(np.linalg.norm(np.asarray(w, dtype=float))) > 0:
            for ratio in np.linspace(0.12, 0.88, 4):
                uniform_points.append(pi + ratio * (pj - pi))
                uniform_vectors.append(np.asarray(w, dtype=float))
        for load in span_loads_of(load_case, mid):        # 杆间集中力
            if getattr(load, "kind", "") != "point":
                continue
            length = float(np.linalg.norm(pj - pi)) or 1.0
            point_points.append(pi + (float(load.a) / length) * (pj - pi))
            point_vectors.append(np.asarray(load.w1[:3], dtype=float))
    if uniform_points:
        arrows(uniform_points, np.array(uniform_vectors), 1.0)
        seen["杆间分布荷载"] = "arrow"
    if point_points:
        arrows(point_points, np.array(point_vectors), 1.8)
        seen["杆间集中力"] = "arrow"
    return seen


def _legend(ax, frame, supports: dict, loads: dict, hinges: bool,
            extra: list | None = None) -> None:
    """图例。**符号靠形状表意，颜色只是加强**——青色荷载在浅底上只有 2.74:1，
    低于 3:1，按 relief 规则必须有可见文字，图例就是那份文字。
    """
    from matplotlib.lines import Line2D
    handles = list(extra or [])
    for kind, marker in supports.items():
        handles.append(Line2D([], [], linestyle="none", marker=marker,
                              markersize=6, color=T.SUPPORT,
                              label=_t(f"支座·{kind}", f"support/{kind}")))
    if hinges:
        handles.append(Line2D([], [], linestyle="none", marker="o", markersize=6,
                              markerfacecolor="none", markeredgecolor=T.HINGE,
                              label=_t("杆端铰", "hinge")))
    for label, marker in loads.items():
        handles.append(Line2D(
            [], [], linestyle="none",
            marker=("$\\rightarrow$" if marker == "arrow" else marker),
            markersize=8, color=T.LOAD, label=_t(label, label)))
    if not handles:
        return
    legend = ax.legend(handles=handles, loc="upper right", frameon=False,
                       fontsize=8.5, labelspacing=0.62, handletextpad=0.7,
                       bbox_to_anchor=(-0.02, 0.98),
                       bbox_transform=ax.transAxes)
    legend.get_frame().set_facecolor(T.SURFACE)
    legend.get_frame().set_edgecolor(T.GRID)
    for text in legend.get_texts():
        text.set_color(T.INK_SECONDARY)


def _member_segments(frame, by_kind: bool = True):
    """未变形杆件，按梁/柱/斜杆分组，供分线宽绘制。"""
    groups: dict[str, list] = {}
    for member in frame.members.values():
        kind = _classify_member(frame, member) if by_kind else "梁"
        groups.setdefault(kind, []).append(
            [frame.nodes[member.i].xyz, frame.nodes[member.j].xyz])
    return groups


def _figure(gutter: float = 0.15):
    """统一画布。3D 轴默认四周留白很大，显式压掉，把结构画大一点。

    ``gutter`` 是左侧留给图例的宽度。图例压在结构上会挡住最该看的东西——
    上一版就把左上角的柱顶和节点荷载遮掉了——所以给它一条独立的栏。
    """
    fig = plt.figure(figsize=(10.4, 6.6), facecolor=T.SURFACE)
    ax = fig.add_subplot(111, projection="3d", facecolor=T.SURFACE)
    ax.set_position([gutter, 0.02, 0.98 - gutter, 0.90])
    # **关掉自动定范围。** 范围最后一律由 _equalize 按节点坐标显式设定；
    # 每加一组线/面 matplotlib 却都要先按数据重算一遍——而数据里解析为零的
    # 坐标常常算成非规格化浮点数（实测 1.24e-311）。3D 轴拿它算边距，下界
    # 变成 -1.8e308、再乘边距系数就溢出成 inf，抛 "Axis limits cannot be NaN
    # or Inf"。这是 test_plot_results_covers_every_kind 在 safe_scale 修好之后
    # 仍然偶发失败的第二条路径（约每 15 次一次）。那次自动计算本来就是多余的。
    # 不用 add_collection3d(autolim=False)：那个参数 3.9 才有，项目声明支持 3.8。
    ax.set_autoscale_on(False)
    return fig, ax


def _title(ax, text: str) -> None:
    ax.set_title(text, color=T.INK, fontsize=11.5, pad=6)


def _callout(ax, frame, point, text: str) -> None:
    """峰值标注：点 + 一小段引线 + 文字。

    直接把文字压在点上会盖住结构线，读者分不清标的是哪一处。引线把文字
    挪到构件外侧，代价只是几个像素。
    """
    size = _model_size(frame)
    point = np.asarray(point, dtype=float)
    tip = point + np.array([0.0, 0.0, 0.055 * size])
    ax.plot(*zip(point, tip, strict=True), color=T.HIGHLIGHT, linewidth=0.9, zorder=6)
    ax.scatter(*point, color=T.HIGHLIGHT, s=34, zorder=7)
    ax.text(tip[0], tip[1], tip[2], f" {text}", color=T.HIGHLIGHT,
            fontsize=9.5, fontweight="bold", zorder=7)


def _structure_handles(frame) -> list:
    """构件类型图例。线宽表意，不占色槽。"""
    from matplotlib.lines import Line2D
    kinds = {_classify_member(frame, m) for m in frame.members.values()}
    order = [k for k in ("柱", "梁", "斜杆") if k in kinds]
    if len(order) < 2:                    # 只有一类时图例没有信息
        return []
    return [Line2D([], [], color=T.REFERENCE, linewidth=0.75 * T.MEMBER_WIDTH[k],
                   label=_t(k, k)) for k in order]


def plot_deformed(frame, solution, case: str | None = None, path: str | Path = "deformed.png",
                  scale: float | None = None, dpi: int = 150,
                  mapping=None) -> dict[str, Any]:
    """画变形图：灰色为原始位置，彩色为放大后的变形。"""
    name = case or solution.primary
    res = solution[name]
    factor = scale if scale is not None else _autoscale(frame, solution, name)
    units = unit_system(frame)

    fig, ax = _figure()
    for kind, segs in _member_segments(frame).items():
        ax.add_collection3d(Line3DCollection(
            segs, colors=T.REFERENCE, linestyles="--", alpha=0.55,
            linewidths=0.75 * T.MEMBER_WIDTH[kind]))
    ax.add_collection3d(Line3DCollection(
        _deformed_segments(frame, solution, name, factor),
        colors=T.ACCENT, linewidths=2.1))

    peak_node, peak = 0, -1.0
    for nid in frame.order():
        mag = float(np.linalg.norm(res.U[frame.node_dofs(nid)[:3]]))
        if mag > peak:
            peak, peak_node = mag, nid
    from frame3d import local_axes, member_endpoints
    from internal_forces import max_centerline_displacement
    worst = max_centerline_displacement(frame, solution, name, stations=201)
    member = frame.members[worst["member"]]
    pi, pj = member_endpoints(frame, member)
    _, rot = local_axes(pi, pj, member.ref_vector)
    p = (pi + worst["x"] * rot[0]
         + factor * np.asarray(worst["vector"]))
    _callout(ax, frame, p,
             f"{worst['value'] * units.disp_scale:.3f} {units.disp_unit}")

    supports = _draw_supports(ax, frame)
    hinges = _draw_hinges(ax, frame)
    loads = _draw_loads(ax, frame, name)
    _equalize(ax, frame)
    _legend(ax, frame, supports, loads, hinges, extra=_structure_handles(frame))
    _title(ax, _t(f"变形图   工况 {name}   位移放大 {factor:.0f} 倍",
                  f"Deformed shape - case {name} - scale x{factor:.0f}"))
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    physical_member, physical_x = _physical_location(
        frame, mapping, worst["member"], worst["x"])
    return {"path": str(path), "case": name, "scale": round(factor, 3),
            "max_displacement_mm": round(peak * units.disp_scale, 6),
            "at_node": peak_node,
            "max_centerline_displacement_mm": round(
                worst["value"] * units.disp_scale, 6),
            "at_member": physical_member,
            "at_x_m": round(physical_x * units.length_to_m, 6)}


def plot_axial(frame, solution, case: str | None = None, path: str | Path = "axial.png",
               dpi: int = 150, mapping=None) -> dict[str, Any]:
    """画轴力图：受拉为暖色、受压为冷色，粗细表示大小。"""
    name = case or solution.primary
    res = solution[name]
    units = unit_system(frame)
    # 以拉为正的轴力是 j 端分量 f[6]；用 f[0] 会把受压的柱子画成受拉
    forces = {mid: float(f[6]) for mid, f in res.member_forces.items()}
    peak = max((abs(v) for v in forces.values()), default=1.0) or 1.0

    fig, ax = _figure()
    # 右侧要放色标：先把三维轴收窄，否则 Z 轴标签会和色标标题叠在一起。
    # 色标必须用**显式的 cax**——交给 fig.colorbar 自己挤地方的话，
    # 它会重算主轴位置，把上面这句收窄直接覆盖掉。
    ax.set_position([0.11, 0.02, 0.68, 0.90])
    # 发散色标：受压 ← 中性灰 → 受拉。不用彩虹，中点不上色相。
    # **分级**而不是连续渐变：连续色带上读不出"这根到底是多少"，只读得出
    # "这根比那根红"；分成有限级之后，每一级对应色标上一个可读的区间。
    # 级数与界面上的云图取自同一个常量（viz_theme.CONTOUR_LEVELS）。
    from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
    levels = T.CONTOUR_LEVELS
    cmap = LinearSegmentedColormap.from_list(
        "axial", [T.DIVERGING_LOW, T.DIVERGING_MID, T.DIVERGING_HIGH],
        N=levels)
    bounds = np.linspace(-peak, peak, levels + 1)
    norm = BoundaryNorm(bounds, cmap.N)
    segs, colors, widths = [], [], []
    for m in frame.members.values():
        segs.append([frame.nodes[m.i].xyz, frame.nodes[m.j].xyz])
        n = forces.get(m.id, 0.0)
        colors.append(cmap(norm(n)))
        widths.append(1.6 + 4.0 * abs(n) / peak)
    ax.add_collection3d(Line3DCollection(segs, colors=colors, linewidths=widths))

    supports = _draw_supports(ax, frame)
    hinges = _draw_hinges(ax, frame)
    loads = _draw_loads(ax, frame, name)
    _equalize(ax, frame)
    _legend(ax, frame, supports, loads, hinges)
    shown_peak = peak * units.force_scale
    _title(ax, _t(f"轴力图   工况 {name}   最大 |N| = {shown_peak:.2f} {units.force_unit}",
                  f"Axial force - case {name} - "
                  f"max |N| = {shown_peak:.2f} {units.force_unit}"))
    from matplotlib.colors import BoundaryNorm as _BoundaryNorm
    shown_bounds = np.linspace(-shown_peak, shown_peak, levels + 1)
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=_BoundaryNorm(shown_bounds, cmap.N))
    bar = fig.colorbar(sm, cax=fig.add_axes([0.855, 0.24, 0.016, 0.46]),
                       ticks=shown_bounds[::2])
    bar.set_label(_t(f"轴力 N ({units.force_unit})   正为受拉",
                     f"Axial N ({units.force_unit}), + = tension"),
                  color=T.INK_SECONDARY, fontsize=9)
    bar.ax.tick_params(colors=T.INK_MUTED, labelsize=8)
    bar.outline.set_edgecolor(T.GRID)
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)

    worst = max(forces, key=lambda k: abs(forces[k]))
    physical_member = (mapping.physical_member_id(worst)
                       if mapping is not None else worst)
    return {"path": str(path), "case": name,
            "max_abs_axial_kN": round(shown_peak, 6),
            "at_member": physical_member,
            "font": _FONT or "英文回退"}


def plot_diagram(frame, solution, component: str = "Mz", case: str | None = None,
                 path: str | Path = "diagram.png", scale: float | None = None,
                 dpi: int = 150, mapping=None) -> dict[str, Any]:
    """整体内力图的静态版本，供报告使用。与交互版同一套数学与配色。

    弯矩按工程习惯画在受拉侧。
    """
    from frame3d import local_axes
    from internal_forces import all_diagrams, physical_member_diagram

    name = case or solution.primary
    units = unit_system(frame)
    if mapping is None:
        diagrams = all_diagrams(frame, solution, name, stations=41)
    else:
        diagrams = {
            physical_id: physical_member_diagram(
                frame, solution, mapping, physical_id, name, stations=41)
            for physical_id in sorted(mapping.physical_to_elements)
        }
    peak = max((abs(d.extreme(component)[1]) for d in diagrams.values()), default=0.0)
    coords = np.array([n.xyz for n in frame.nodes.values()])
    size = float(np.max(coords.max(axis=0) - coords.min(axis=0))) or 1.0
    factor = scale if scale is not None else T.safe_scale(size, peak, 0.09)
    axis_index = 2 if component in {"Vz", "My"} else 1
    draw_sign = -1.0 if component in {"My", "Mz"} else 1.0

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig, ax = _figure()
    for kind, segs in _member_segments(frame).items():
        ax.add_collection3d(Line3DCollection(
            segs, colors=T.INK_MUTED, linewidths=T.MEMBER_WIDTH[kind]))

    worst = {"member": None, "value": 0.0, "x": 0.0, "point": None}
    outlines: list = []
    faces: dict[str, list] = {"正": [], "负": []}
    worst_point = None
    for mid, d in diagrams.items():
        element_id = (mapping.element_ids(mid)[0] if mapping is not None else mid)
        member = frame.members[element_id]
        pi = frame.nodes[member.i].xyz
        _, rot = local_axes(pi, frame.nodes[member.j].xyz, member.ref_vector)
        base = pi[None, :] + d.x[:, None] * rot[0][None, :]
        values = d.component(component)
        offset = base + draw_sign * factor * values[:, None] * rot[axis_index][None, :]
        outlines.append(list(offset))
        # 填充图比梯子线可读得多：一眼看出受拉侧和正负分区。
        # 按符号切段，同号的相邻站点连成一个多边形。
        start = 0
        for k in range(1, len(values) + 1):
            end = k == len(values)
            if not end and (values[k] >= 0) == (values[start] >= 0):
                continue
            stop = k if end else k + 1
            if stop - start >= 2:
                patch = list(base[start:stop]) + list(offset[start:stop])[::-1]
                faces["正" if values[start] >= 0 else "负"].append(patch)
            start = k
        pos, value = d.extreme(component)
        if abs(value) > abs(worst["value"]):
            index = int(np.argmax(np.abs(values)))
            worst = {"member": mid, "value": value, "x": pos}
            worst_point = offset[index]

    for sign, color in (("负", T.DIVERGING_LOW), ("正", T.DIVERGING_HIGH)):
        if faces[sign]:
            ax.add_collection3d(Poly3DCollection(
                faces[sign], facecolors=color, edgecolors="none", alpha=0.30))
    ax.add_collection3d(Line3DCollection(outlines, colors=T.ACCENT, linewidths=1.6))

    moment = component in {"T", "My", "Mz", "M"}
    value_scale = units.moment_scale if moment else units.force_scale
    unit = units.moment_unit if moment else units.force_unit
    shown = worst["value"] * value_scale
    if worst_point is not None:
        _callout(ax, frame, worst_point, f"{shown:.3f} {unit}")

    supports = _draw_supports(ax, frame)
    hinges = _draw_hinges(ax, frame)
    _equalize(ax, frame)
    from matplotlib.patches import Patch
    signs = [Patch(facecolor=c, alpha=0.30, edgecolor="none",
                   label=_t(f"{component} {s}", f"{component} {s}"))
             for s, c in (("正", T.DIVERGING_HIGH), ("负", T.DIVERGING_LOW))
             if faces[s]]
    _legend(ax, frame, supports, {}, hinges,
            extra=_structure_handles(frame) + signs)
    _title(ax, _t(f"{component} 内力图   工况 {name}   "
                  f"最大 {shown:.3f} {unit} @ 杆件 {worst['member']}",
                  f"{component} diagram - case {name} - "
                  f"max {shown:.3f} {unit} @ member {worst['member']}"))
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {"path": str(path), "component": component, "case": name,
            "peak": round(shown, 6), "unit": unit,
            "at_member": worst["member"],
            "at_x": round(worst["x"] * units.length_to_m, 4)}
