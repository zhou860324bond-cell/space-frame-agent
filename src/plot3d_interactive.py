"""交互式三维视图（Plotly）。

与 `plot3d.py` 是同一套配色、同一份数据，区别只在载体：这里出的图能旋转、
缩放、悬停读数，用于界面；`plot3d.py` 出静态 PNG，用于报告。

悬停信息是这里的重点——**把节点编号、坐标、位移分量直接挂在图元上**，
用户不必再去翻表格核对哪个节点是哪个。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go

import viz_theme as T
from units import of as unit_system

# 三维视口用深色（见 viz_theme 里的说明）。二维内力图仍走浅底——
# 那更像一张图表，不是视口，深底反而怪。
_AXIS = dict(showbackground=False, gridcolor=T.VIEWPORT_GRID,
             zerolinecolor=T.VIEWPORT_AXIS, linecolor=T.VIEWPORT_AXIS,
             tickfont=dict(color=T.VIEWPORT_INK_MUTED, size=10))
_AXIS_TITLE = dict(font=dict(color=T.VIEWPORT_INK_MUTED, size=11))


def _layout(title: str, height: int, length_unit: str = "m") -> dict:
    return dict(
        title=dict(text=title, font=dict(color=T.VIEWPORT_INK, size=15),
                   x=0.015, y=0.97),
        paper_bgcolor=T.VIEWPORT_BG, plot_bgcolor=T.VIEWPORT_BG,
        margin=dict(l=0, r=0, t=40, b=0), height=height,
        scene=dict(xaxis=dict(title=dict(text=f"X ({length_unit})", **_AXIS_TITLE), **_AXIS),
                   yaxis=dict(title=dict(text=f"Y ({length_unit})", **_AXIS_TITLE), **_AXIS),
                   zaxis=dict(title=dict(text=f"Z ({length_unit})", **_AXIS_TITLE), **_AXIS),
                   aspectmode="data",          # 真实比例，不把结构塞进立方体
                   dragmode="orbit",
                   bgcolor=T.VIEWPORT_BG,
                   camera=dict(eye=dict(x=1.55, y=-1.65, z=0.85))),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=0.0, x=0.0,
                    font=dict(color=T.VIEWPORT_INK_MUTED, size=10),
                    bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#252c35", bordercolor=T.VIEWPORT_AXIS,
                        font=dict(color=T.VIEWPORT_INK, size=11)),
    )


def _member_segments(frame, U=None, scale=0.0):
    """把所有杆件串成一条带断点的折线，一次 trace 画完，浏览器里更流畅。"""
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for m in frame.members.values():
        for nid in (m.i, m.j):
            p = frame.nodes[nid].xyz
            if U is not None and scale:
                p = p + scale * U[frame.node_dofs(nid)[:3]]
            xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
        xs.append(None); ys.append(None); zs.append(None)
    return xs, ys, zs


def _deformed_member_segments(frame, solution, case: str, scale: float,
                              stations: int = 41):
    """全部杆件恢复后的中心线，杆与杆之间用 None 断开。"""
    from frame3d import local_axes, member_endpoints
    from internal_forces import member_displacement

    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for mid in sorted(frame.members):
        member = frame.members[mid]
        x, displacement = member_displacement(
            frame, solution, mid, case, stations=stations)
        pi, pj = member_endpoints(frame, member)
        _, rot = local_axes(pi, pj, member.ref_vector)
        points = pi[None, :] + x[:, None] * rot[0] + scale * displacement
        xs.extend(points[:, 0]); ys.extend(points[:, 1]); zs.extend(points[:, 2])
        xs.append(None); ys.append(None); zs.append(None)
    return xs, ys, zs


def _auto_scale(frame, solution, case: str) -> float:
    from internal_forces import max_centerline_displacement

    coords = np.array([n.xyz for n in frame.nodes.values()])
    size = float(np.max(coords.max(axis=0) - coords.min(axis=0))) or 1.0
    peak = float(max_centerline_displacement(
        frame, solution, case, stations=201)["value"])
    return T.safe_scale(size, peak, 0.05) or 1.0


def figure_deformed(frame, solution, case: str | None = None,
                    scale: float | None = None, height: int = 440) -> tuple[go.Figure, dict]:
    name = case or solution.primary
    res = solution[name]
    factor = scale if scale is not None else _auto_scale(frame, solution, name)
    units = unit_system(frame)

    fig = go.Figure()
    x0, y0, z0 = _member_segments(frame)
    fig.add_trace(go.Scatter3d(x=x0, y=y0, z=z0, mode="lines", name="原始位置",
                               line=dict(color=T.VIEWPORT_REFERENCE, width=2, dash="dot"),
                               hoverinfo="skip"))
    x1, y1, z1 = _deformed_member_segments(frame, solution, name, factor)
    fig.add_trace(go.Scatter3d(x=x1, y=y1, z=z1, mode="lines",
                               name=f"变形后（放大 {factor:.0f} 倍）",
                               line=dict(color=T.VIEWPORT_ACCENT, width=5), hoverinfo="skip"))

    nodes, xs, ys, zs, mags, text = [], [], [], [], [], []
    for nid in frame.order():
        d = frame.node_dofs(nid)
        u = res.U[d[:3]]
        p = frame.nodes[nid].xyz + factor * u
        nodes.append(nid); xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
        mag = float(np.linalg.norm(u))
        mags.append(mag * units.disp_scale)
        text.append(f"<b>节点 {nid}</b><br>"
                    f"坐标 ({frame.nodes[nid].x:g}, {frame.nodes[nid].y:g}, "
                    f"{frame.nodes[nid].z:g}) {units.length}<br>"
                    f"合位移 {mag * units.disp_scale:.4f} {units.disp_unit}<br>"
                    f"ux {u[0] * units.disp_scale:.4f}　"
                    f"uy {u[1] * units.disp_scale:.4f}　"
                    f"uz {u[2] * units.disp_scale:.4f} {units.disp_unit}")
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="markers", name="节点",
                               marker=dict(size=3.5, color=T.VIEWPORT_INK, opacity=0.9),
                               hovertemplate="%{text}<extra></extra>", text=text))

    peak = int(np.argmax(mags))
    fig.add_trace(go.Scatter3d(
        x=[xs[peak]], y=[ys[peak]], z=[zs[peak]], mode="markers+text",
        name="最大位移",
        marker=dict(size=9, color=T.VIEWPORT_HIGHLIGHT, line=dict(color=T.VIEWPORT_BG, width=2)),
        text=[f"  {mags[peak]:.4f} {units.disp_unit}"], textposition="middle right",
        textfont=dict(color=T.VIEWPORT_HIGHLIGHT, size=12), hoverinfo="skip"))

    # 节点最大值仍保留给结果表兼容；图形标出的真正最大值来自杆件中心线，
    # 单单元简支梁的跨中挠度不会再被漏掉。
    from frame3d import local_axes, member_endpoints
    from internal_forces import max_centerline_displacement
    worst = max_centerline_displacement(frame, solution, name, stations=201)
    member = frame.members[worst["member"]]
    pi, pj = member_endpoints(frame, member)
    _, rot = local_axes(pi, pj, member.ref_vector)
    point = pi + worst["x"] * rot[0] + factor * np.asarray(worst["vector"])
    fig.add_trace(go.Scatter3d(
        x=[point[0]], y=[point[1]], z=[point[2]], mode="markers+text",
        name="杆件中心线最大位移",
        marker=dict(size=8, color=T.VIEWPORT_HIGHLIGHT),
        text=[f"  杆件 {worst['member']}："
              f"{worst['value'] * units.disp_scale:.4f} {units.disp_unit}"],
        textposition="top right", hoverinfo="skip"))

    sx = [frame.nodes[n].x for n in frame.supports]
    sy = [frame.nodes[n].y for n in frame.supports]
    sz = [frame.nodes[n].z for n in frame.supports]
    fig.add_trace(go.Scatter3d(x=sx, y=sy, z=sz, mode="markers", name="支座",
                               marker=dict(size=5, color=T.VIEWPORT_INK_MUTED,
                                           symbol="diamond"),
                               hovertemplate="支座 节点 %{customdata}<extra></extra>",
                               customdata=list(frame.supports)))

    fig.update_layout(**_layout(f"变形图　工况 {name}", height, units.length))
    return fig, {"case": name, "scale": round(factor, 3),
                 "max_displacement_mm": round(mags[peak], 6), "at_node": nodes[peak],
                 "max_centerline_displacement_mm": round(
                     worst["value"] * units.disp_scale, 6),
                 "at_member": worst["member"],
                 "at_x_m": round(worst["x"] * units.length_to_m, 6)}


def figure_axial(frame, solution, case: str | None = None,
                 height: int = 560) -> tuple[go.Figure, dict]:
    """杆件按轴力着色。以受拉为正，取 j 端分量 f[6]。"""
    name = case or solution.primary
    res = solution[name]
    units = unit_system(frame)
    forces = {mid: float(f[6]) for mid, f in res.member_forces.items()}
    peak = max((abs(v) for v in forces.values()), default=1.0) or 1.0

    fig = go.Figure()
    for m in frame.members.values():
        n = forces.get(m.id, 0.0)
        a, b = frame.nodes[m.i].xyz, frame.nodes[m.j].xyz
        f = res.member_forces[m.id]
        label = (f"<b>杆件 {m.id}</b>　节点 {m.i} → {m.j}<br>"
                 f"轴力 N {n / 1e3:.3f} kN（{'受拉' if n > 0 else '受压'}）<br>"
                 f"Mz(i) {f[5] / 1e3:.3f}　Mz(j) {f[11] / 1e3:.3f} kN·m")
        fig.add_trace(go.Scatter3d(
            x=[a[0], b[0]], y=[a[1], b[1]], z=[a[2], b[2]], mode="lines",
            line=dict(color=[n / 1e3, n / 1e3], colorscale=T.VIEWPORT_DIVERGING,
                      cmin=-peak / 1e3, cmax=peak / 1e3,
                      width=2 + 6 * abs(n) / peak,
                      colorbar=dict(title=dict(text="N (kN)", side="top",
                                               font=dict(color=T.VIEWPORT_INK_MUTED, size=11)),
                                    tickfont=dict(color=T.VIEWPORT_INK_MUTED, size=10),
                                    thickness=12, len=0.62, x=0.99, y=0.5)),
            showlegend=False, hovertemplate=label + "<extra></extra>"))

    # 细线轮廓画在最上层。轴力接近零的杆件颜色落在中性灰上，若只靠颜色，
    # 梁会连同几何一起从画面里消失；这条压在上面的发丝线保证几何始终可读。
    # 轮廓担几何，颜色只担大小——两件事分开编码。
    x0, y0, z0 = _member_segments(frame)
    fig.add_trace(go.Scatter3d(x=x0, y=y0, z=z0, mode="lines", showlegend=False,
                               line=dict(color=T.VIEWPORT_REFERENCE, width=1.5),
                               hoverinfo="skip"))

    worst = max(forces, key=lambda k: abs(forces[k]))
    fig.update_layout(**_layout(
        f"轴力图　工况 {name}　最大 |N| = "
        f"{peak * units.force_scale:.2f} {units.force_unit}",
        height, units.length))
    fig.update_layout(showlegend=False)
    return fig, {"case": name, "max_abs_axial_kN": round(peak / 1e3, 6),
                 "at_member": worst}


# --------------------------------------------------------------- 内力图

_DIAGRAM_LABEL = {"N": "轴力 N (kN)", "Vy": "剪力 Vy (kN)", "Vz": "剪力 Vz (kN)",
                  "T": "扭矩 T (kN·m)", "My": "弯矩 My (kN·m)", "Mz": "弯矩 Mz (kN·m)"}
_IS_MOMENT = {"T", "My", "Mz"}


def figure_member_diagram(frame, solution, member_id: int, component: str = "Mz",
                          case: str | None = None, height: int = 300) -> tuple[go.Figure, dict]:
    """单根杆件的内力图。横轴是沿杆长的距离，纵轴是内力。"""
    from internal_forces import member_diagram

    d = member_diagram(frame, solution, member_id, case, stations=201)
    units = unit_system(frame)
    moment = component in _IS_MOMENT
    value_scale = units.moment_scale if moment else units.force_scale
    values = d.component(component) * value_scale
    shown_x = d.x * units.length_to_m
    x_at, peak = d.extreme(component)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=shown_x, y=values, mode="lines", name=component,
                             line=dict(color=T.ACCENT, width=2),
                             fill="tozeroy",
                             fillcolor="rgba(42,120,214,0.12)",
                             hovertemplate="x = %{x:.3f} m<br>"
                                           f"{_DIAGRAM_LABEL[component]} = "
                                           "%{y:.3f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=T.AXIS, width=1))
    fig.add_trace(go.Scatter(x=[x_at * units.length_to_m],
                             y=[peak * value_scale], mode="markers+text",
                             marker=dict(size=9, color=T.HIGHLIGHT,
                                         line=dict(color=T.SURFACE, width=2)),
                             text=[f"  {peak * value_scale:.3f}"], textposition="middle right",
                             textfont=dict(color=T.HIGHLIGHT, size=11),
                             showlegend=False, hoverinfo="skip"))

    fig.update_layout(
        title=dict(text=f"杆件 {member_id}　{_DIAGRAM_LABEL[component]}　"
                        f"工况 {d.case}",
                   font=dict(color=T.INK, size=14), x=0.02, y=0.95),
        paper_bgcolor=T.SURFACE, plot_bgcolor=T.SURFACE, height=height,
        margin=dict(l=60, r=20, t=44, b=44), showlegend=False,
        xaxis=dict(title=dict(text="沿杆长 x (m)",
                              font=dict(color=T.INK_SECONDARY, size=11)),
                   gridcolor=T.GRID, zerolinecolor=T.AXIS, linecolor=T.AXIS,
                   tickfont=dict(color=T.INK_MUTED, size=10)),
        yaxis=dict(title=dict(text=_DIAGRAM_LABEL[component],
                              font=dict(color=T.INK_SECONDARY, size=11)),
                   gridcolor=T.GRID, zerolinecolor=T.AXIS, linecolor=T.AXIS,
                   tickfont=dict(color=T.INK_MUTED, size=10)),
    )
    return fig, {"member": member_id, "case": d.case, "component": component,
                 "peak": round(peak * value_scale, 6),
                 "at_x": round(x_at * units.length_to_m, 4),
                 "length": round(d.length * units.length_to_m, 4)}


def figure_diagram_3d(frame, solution, component: str = "Mz",
                      case: str | None = None, scale: float | None = None,
                      height: int = 480) -> tuple[go.Figure, dict]:
    """整体内力图：沿每根杆件把内力画成垂直于杆轴的轮廓。

    弯矩 Mz 绕局部 z，弯曲发生在局部 x-y 平面内，所以画在局部 y 方向；
    Vz 与 My 对应局部 z 方向。这样图形方向和实际弯曲平面一致，
    不是随便找个方向摊开。
    """
    from frame3d import local_axes
    from internal_forces import all_diagrams, exceeds

    name = case or solution.primary
    units = unit_system(frame)
    diagrams = all_diagrams(frame, solution, name, stations=41)
    peak = max((abs(d.extreme(component)[1]) for d in diagrams.values()), default=0.0)

    coords = np.array([n.xyz for n in frame.nodes.values()])
    size = float(np.max(coords.max(axis=0) - coords.min(axis=0))) or 1.0
    factor = scale if scale is not None else T.safe_scale(size, peak, 0.09)

    # 局部 y 承载 Vy / Mz，局部 z 承载 Vz / My；N 与 T 沿局部 y 摊开即可
    axis_index = 2 if component in {"Vz", "My"} else 1
    # 弯矩按国内习惯画在**受拉侧**：重力荷载下梁跨中正弯矩，受拉在下，图形向下鼓。
    # 不翻的话图会画在受压侧，土木背景的人一眼就看出别扭。
    draw_sign = -1.0 if component in {"My", "Mz"} else 1.0

    fig = go.Figure()
    xs, ys, zs = _member_segments(frame)
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name="结构",
                               line=dict(color=T.INK_MUTED, width=3),
                               hoverinfo="skip"))

    worst = {"member": None, "value": 0.0, "x": 0.0}
    for mid, d in diagrams.items():
        member = frame.members[mid]
        pi, pj = frame.nodes[member.i].xyz, frame.nodes[member.j].xyz
        _, rot = local_axes(pi, pj, member.ref_vector)
        axis_dir, along = rot[axis_index], rot[0]
        values = d.component(component)
        base = pi[None, :] + d.x[:, None] * along[None, :]
        offset = base + draw_sign * factor * values[:, None] * axis_dir[None, :]

        px, py, pz, text = [], [], [], []
        for k in range(len(d.x)):        # 闭合成填充轮廓：基线 -> 偏移 -> 基线
            px += [base[k, 0], offset[k, 0], None]
            py += [base[k, 1], offset[k, 1], None]
            pz += [base[k, 2], offset[k, 2], None]
        fig.add_trace(go.Scatter3d(x=px, y=py, z=pz, mode="lines", showlegend=False,
                                   line=dict(color="rgba(42,120,214,0.35)", width=1),
                                   hoverinfo="skip"))
        for k in range(len(d.x)):
            value_scale = (units.moment_scale if component in _IS_MOMENT
                           else units.force_scale)
            text.append(f"<b>杆件 {mid}</b><br>"
                        f"x = {d.x[k] * units.length_to_m:.3f} m<br>"
                        f"{_DIAGRAM_LABEL[component]} = {values[k] * value_scale:.3f}")
        fig.add_trace(go.Scatter3d(
            x=offset[:, 0], y=offset[:, 1], z=offset[:, 2], mode="lines",
            showlegend=False, line=dict(color=T.ACCENT, width=3),
            hovertemplate="%{text}<extra></extra>", text=text))

        pos, value = d.extreme(component)
        if exceeds(value, worst["value"]):
            worst = {"member": mid, "value": value, "x": pos}

    if worst["member"] is not None:
        d = diagrams[worst["member"]]
        member = frame.members[worst["member"]]
        pi = frame.nodes[member.i].xyz
        _, rot = local_axes(pi, frame.nodes[member.j].xyz, member.ref_vector)
        point = (pi + worst["x"] * rot[0]
                 + draw_sign * factor * worst["value"] * rot[axis_index])
        fig.add_trace(go.Scatter3d(
            x=[point[0]], y=[point[1]], z=[point[2]], mode="markers+text",
            name="最大值",
            marker=dict(size=8, color=T.HIGHLIGHT, line=dict(color=T.SURFACE, width=2)),
            text=[f"  {worst['value'] * value_scale:.3f}"], textposition="middle right",
            textfont=dict(color=T.HIGHLIGHT, size=11), hoverinfo="skip"))

    value_scale = (units.moment_scale if component in _IS_MOMENT
                   else units.force_scale)
    unit = units.moment_unit if component in _IS_MOMENT else units.force_unit
    note = "　弯矩画在受拉侧" if component in {"My", "Mz"} else ""
    fig.update_layout(**_layout(
        f"{_DIAGRAM_LABEL[component]}　工况 {name}　"
        f"最大 {worst['value'] * value_scale:.3f} {unit} @ 杆件 {worst['member']}{note}",
        height, units.length))
    return fig, {"component": component, "case": name,
                 "peak": round(worst["value"] * value_scale, 6),
                 "at_member": worst["member"],
                 "at_x": round(worst["x"] * units.length_to_m, 4),
                 "scale": round(factor, 6)}


def figure_member_envelope(frame, solution, member_id: int, component: str = "Mz",
                           cases=None, height: int = 320) -> tuple[go.Figure, dict]:
    """单根杆件的内力包络：上下包线之间填色，并标出控制组合。

    包络必须逐点画。"先各工况取极值再横着拉一条线"画出来的是个矩形框，
    看着更保守，其实位置全错——设计时按它选截面会选错地方。
    """
    from envelope import member_envelope

    env = member_envelope(frame, solution, member_id, cases, stations=201)
    hi = env.upper[component] / 1e3
    lo = env.lower[component] / 1e3
    worst = env.extreme(component)

    fig = go.Figure()
    # 先画下包线再画上包线并 fill='tonexty'，两线之间才会填上
    fig.add_trace(go.Scatter(x=env.x, y=lo, mode="lines", name="下包线",
                             line=dict(color=T.DIVERGING_LOW, width=1.6),
                             hovertemplate="x = %{x:.3f} m<br>下 = %{y:.3f}"
                                           "<extra></extra>"))
    fig.add_trace(go.Scatter(x=env.x, y=hi, mode="lines", name="上包线",
                             line=dict(color=T.DIVERGING_HIGH, width=1.6),
                             fill="tonexty", fillcolor="rgba(42,120,214,0.10)",
                             hovertemplate="x = %{x:.3f} m<br>上 = %{y:.3f}"
                                           "<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=T.AXIS, width=1))
    fig.add_trace(go.Scatter(
        x=[worst["x"]], y=[worst["value"] / 1e3], mode="markers+text",
        marker=dict(size=9, color=T.HIGHLIGHT,
                    line=dict(color=T.SURFACE, width=2)),
        text=[f"  {worst['value'] / 1e3:.3f}　{worst['case']}"],
        textposition="middle right", textfont=dict(color=T.HIGHLIGHT, size=11),
        showlegend=False, hoverinfo="skip"))

    fig.update_layout(
        title=dict(text=f"杆件 {member_id}　{_DIAGRAM_LABEL[component]} 包络　"
                        f"{len(env.cases)} 个组合",
                   font=dict(color=T.INK, size=14), x=0.02, y=0.95),
        paper_bgcolor=T.SURFACE, plot_bgcolor=T.SURFACE, height=height,
        margin=dict(l=60, r=20, t=44, b=44),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right",
                    font=dict(color=T.INK_SECONDARY, size=10)),
        xaxis=dict(title=dict(text="沿杆长 x (m)",
                              font=dict(color=T.INK_SECONDARY, size=11)),
                   gridcolor=T.GRID, zerolinecolor=T.AXIS, linecolor=T.AXIS,
                   tickfont=dict(color=T.INK_MUTED, size=10)),
        yaxis=dict(title=dict(text=_DIAGRAM_LABEL[component],
                              font=dict(color=T.INK_SECONDARY, size=11)),
                   gridcolor=T.GRID, zerolinecolor=T.AXIS, linecolor=T.AXIS,
                   tickfont=dict(color=T.INK_MUTED, size=10)),
    )
    return fig, {"member": member_id, "component": component,
                 "cases": list(env.cases),
                 "peak": round(worst["value"] / 1e3, 6),
                 "at_x": round(worst["x"], 4),
                 "governing_case": worst["case"],
                 "governing_list": env.governing(component)}


def figure_model(frame, case: str | None = None, height: int = 440,
                 supports: bool = True, loads: bool = True
                 ) -> tuple[go.Figure, dict]:
    """建模阶段的三维视图：几何 + 约束符号 + 荷载符号，**不需要求解结果**。

    这是"先看一眼再算"的那一步。荷载方向加反是最常见也最难从数字上察觉的
    错误——2400 kN 的反力合计看着完全正常，直到你发现荷载是朝上的。
    画出来一眼就能看见。
    """
    from viz_symbols import load_traces, model_size, support_traces

    fig = go.Figure()
    xs, ys, zs = _member_segments(frame)
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name="杆件",
                               line=dict(color=T.ACCENT, width=4),
                               hoverinfo="skip"))

    px, py, pz, text = [], [], [], []
    for nid in frame.order():
        n = frame.nodes[nid]
        px.append(n.x); py.append(n.y); pz.append(n.z)
        mask = frame.supports.get(nid)
        note = ""
        if mask:
            names = [d for d, f in zip(("ux", "uy", "uz", "rx", "ry", "rz"), mask, strict=True) if f]
            note = f"<br>约束 {'、'.join(names)}"
        text.append(f"<b>节点 {nid}</b><br>({n.x:g}, {n.y:g}, {n.z:g}) m{note}")
    fig.add_trace(go.Scatter3d(x=px, y=py, z=pz, mode="markers", name="节点",
                               marker=dict(size=3.5, color=T.INK_SECONDARY,
                                           opacity=0.8),
                               hovertemplate="%{text}<extra></extra>", text=text))

    size = model_size(frame)
    info: dict[str, Any] = {"nodes": len(frame.nodes), "members": len(frame.members)}
    if supports:
        traces, meta = support_traces(frame, size)
        for t in traces:
            fig.add_trace(t)
        info["supports"] = meta
    if loads and case:
        traces, meta = load_traces(frame, case, size)
        for t in traces:
            fig.add_trace(t)
        info["loads"] = meta

    title = "模型视图" + (f"　工况 {case}" if case else "")
    fig.update_layout(**_layout(title, height))
    return fig, info
