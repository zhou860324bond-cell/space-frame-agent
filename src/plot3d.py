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
    if peak <= 0 or size <= 0:
        return 1.0
    return 0.05 * size / peak


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


def _equalize(ax, frame) -> None:
    """按真实比例设视图。三个方向各自取数据范围，再用 box_aspect 保住比例，

    否则一个 18x6x7.2 的框架会被塞进立方体里，看着像另一个结构。
    """
    coords = np.array([n.xyz for n in frame.nodes.values()])
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    pad = 0.08 * float(span.max())
    lo, hi = lo - pad, hi + pad
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(tuple(hi - lo))
    length = unit_system(frame).length
    ax.set_xlabel(f"X ({length})")
    ax.set_ylabel(f"Y ({length})")
    ax.set_zlabel(f"Z ({length})")
    ax.view_init(elev=22, azim=-58)


def plot_deformed(frame, solution, case: str | None = None, path: str | Path = "deformed.png",
                  scale: float | None = None, dpi: int = 150,
                  mapping=None) -> dict[str, Any]:
    """画变形图：灰色为原始位置，彩色为放大后的变形。"""
    name = case or solution.primary
    res = solution[name]
    factor = scale if scale is not None else _autoscale(frame, solution, name)
    units = unit_system(frame)

    fig = plt.figure(figsize=(9, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(Line3DCollection(_segments(frame), colors=T.REFERENCE,
                                         linewidths=1.0, linestyles="--"))
    ax.add_collection3d(Line3DCollection(
                                         _deformed_segments(frame, solution, name, factor),
                                         colors=T.ACCENT, linewidths=2.0))

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
    ax.scatter(*p, color=T.HIGHLIGHT, s=45, zorder=5)
    ax.text(p[0], p[1], p[2],
            f"  {worst['value'] * units.disp_scale:.3f} {units.disp_unit}",
            color=T.HIGHLIGHT, fontsize=9)

    _equalize(ax, frame)
    ax.set_title(_t(f"变形图  工况 {name}  放大 {factor:.0f} 倍",
                    f"Deformed shape - case {name} - scale x{factor:.0f}"))
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
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

    fig = plt.figure(figsize=(9, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    # 发散色标：受压 ← 中性灰 → 受拉。不用彩虹，中点不上色相。
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "axial", [T.DIVERGING_LOW, T.DIVERGING_MID, T.DIVERGING_HIGH])
    segs, colors, widths = [], [], []
    for m in frame.members.values():
        segs.append([frame.nodes[m.i].xyz, frame.nodes[m.j].xyz])
        n = forces.get(m.id, 0.0)
        colors.append(cmap(0.5 + 0.5 * n / peak))
        widths.append(1.0 + 3.0 * abs(n) / peak)
    ax.add_collection3d(Line3DCollection(segs, colors=colors, linewidths=widths))

    _equalize(ax, frame)
    shown_peak = peak * units.force_scale
    ax.set_title(_t(f"轴力图  工况 {name}  最大 |N| = {shown_peak:.2f} {units.force_unit}",
                    f"Axial force - case {name} - max |N| = {shown_peak:.2f} {units.force_unit}"))
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=-shown_peak, vmax=shown_peak))
    fig.colorbar(sm, ax=ax, shrink=0.6,
                 label=_t(f"轴力 N ({units.force_unit})  正为拉",
                          f"Axial N ({units.force_unit}), + = tension"))
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
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
    factor = scale if scale is not None else (0.09 * size / peak if peak > 0 else 0.0)
    axis_index = 2 if component in {"Vz", "My"} else 1
    draw_sign = -1.0 if component in {"My", "Mz"} else 1.0

    fig = plt.figure(figsize=(9, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(Line3DCollection(_segments(frame), colors=T.INK_MUTED,
                                         linewidths=1.6))

    worst = {"member": None, "value": 0.0, "x": 0.0}
    outlines, ladders = [], []
    for mid, d in diagrams.items():
        element_id = (mapping.element_ids(mid)[0] if mapping is not None else mid)
        member = frame.members[element_id]
        pi = frame.nodes[member.i].xyz
        _, rot = local_axes(pi, frame.nodes[member.j].xyz, member.ref_vector)
        base = pi[None, :] + d.x[:, None] * rot[0][None, :]
        offset = base + draw_sign * factor * d.component(component)[:, None] * rot[axis_index][None, :]
        outlines.append(list(offset))
        ladders += [[base[k], offset[k]] for k in range(0, len(d.x), 2)]
        pos, value = d.extreme(component)
        if abs(value) > abs(worst["value"]):
            worst = {"member": mid, "value": value, "x": pos}

    ax.add_collection3d(Line3DCollection(ladders, colors=T.ACCENT,
                                         linewidths=0.6, alpha=0.45))
    ax.add_collection3d(Line3DCollection(outlines, colors=T.ACCENT, linewidths=1.8))

    _equalize(ax, frame)
    moment = component in {"T", "My", "Mz", "M"}
    value_scale = units.moment_scale if moment else units.force_scale
    unit = units.moment_unit if moment else units.force_unit
    shown = worst["value"] * value_scale
    label = _t(f"{component} 内力图  工况 {name}  最大 {shown:.3f} {unit}"
               f"  @ 杆件 {worst['member']}",
               f"{component} diagram - case {name} - "
               f"max {shown:.3f} {unit} @ member {worst['member']}")
    ax.set_title(label)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return {"path": str(path), "component": component, "case": name,
            "peak": round(shown, 6), "unit": unit,
            "at_member": worst["member"],
            "at_x": round(worst["x"] * units.length_to_m, 4)}
