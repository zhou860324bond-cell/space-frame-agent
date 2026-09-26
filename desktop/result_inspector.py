"""结果探针、极值定位与截面正应力图所需的纯计算。"""

from __future__ import annotations

import numpy as np


FORCES = {"N", "Vy", "Vz", "V"}
MOMENTS = {"T", "My", "Mz", "M"}


def _display_system(frame):
    from units import of
    return of(frame)


def point_on_member(frame, member_id: int, x: float) -> np.ndarray:
    from frame3d import member_endpoints

    member = frame.members[member_id]
    pi, pj = member_endpoints(frame, member)
    length = float(np.linalg.norm(pj - pi))
    ratio = 0.0 if length == 0.0 else float(np.clip(x / length, 0.0, 1.0))
    return pi + ratio * (pj - pi)


def projected_station(frame, member_id: int, point) -> float:
    """将拾取点投影到杆件中心线，返回距 i 端位置。"""
    from frame3d import member_endpoints

    member = frame.members[member_id]
    pi, pj = member_endpoints(frame, member)
    axis = pj - pi
    length = float(np.linalg.norm(axis))
    if length == 0.0:
        return 0.0
    return float(np.clip(np.dot(np.asarray(point) - pi, axis) / length,
                         0.0, length))


def section_stress_grid(section, N: float, My: float, Mz: float,
                        samples: int = 41) -> dict:
    """截面包络内的线性正应力场；受拉为正。"""
    from stress import StressUnavailable

    if section.cy is None or section.cz is None:
        raise StressUnavailable(
            f"截面 {section.name!r} 缺少 cy/cz，无法绘制截面正应力分布。")
    y = np.linspace(-section.cy, section.cy, samples)
    z = np.linspace(-section.cz, section.cz, samples)
    yy, zz = np.meshgrid(y, z, indexing="ij")
    sigma = N / section.A - Mz * yy / section.Iz + My * zz / section.Iy
    mask = ((yy / section.cy) ** 2 + (zz / section.cz) ** 2 <= 1.0
            if section.circular else np.ones_like(sigma, dtype=bool))
    return {"y": y, "z": z, "sigma": np.where(mask, sigma, np.nan),
            "circular": bool(section.circular)}


def probe_member(frame, solution, member_id: int, point,
                 case: str | None = None) -> dict:
    """读取任意杆件截面的六内力、全局/局部位移与极端纤维正应力。"""
    from frame3d import local_axes, member_endpoints
    from internal_forces import member_diagram, member_displacement
    from stress import StressUnavailable, extreme_normal_stress

    name = case or solution.primary
    member = frame.members[member_id]
    pi, pj = member_endpoints(frame, member)
    length, rotation = local_axes(pi, pj, member.ref_vector)
    x = projected_station(frame, member_id, point)
    diagram = member_diagram(frame, solution, member_id, name,
                             at_x=np.asarray([x]))
    raw = {component: float(diagram.component(component)[0])
           for component in ("N", "Vy", "Vz", "T", "My", "Mz")}
    stations, displacement = member_displacement(
        frame, solution, member_id, name, stations=401)
    global_u = np.array([np.interp(x, stations, displacement[:, axis])
                         for axis in range(3)])
    local_u = rotation @ global_u
    system = _display_system(frame)
    values = {
        component: raw[component] * (system.moment_scale
                                     if component in MOMENTS
                                     else system.force_scale)
        for component in raw
    }
    stress = None
    stress_error = None
    section = frame.sections[member.section]
    try:
        low, high = extreme_normal_stress(
            section, raw["N"], raw["My"], raw["Mz"])
        stress_scale = 1e-6 if system.name == "N-m-Pa" else 1.0
        stress = {"min": float(low) * stress_scale,
                  "max": float(high) * stress_scale, "unit": "MPa",
                  "grid": section_stress_grid(
                      section, raw["N"], raw["My"], raw["Mz"])}
        stress["grid"]["sigma"] *= stress_scale
    except StressUnavailable as exc:
        stress_error = str(exc)
    return {
        "member": member_id, "case": name, "section": section.name,
        "i": member.i, "j": member.j, "x": x,
        "length": length, "point": point_on_member(frame, member_id, x),
        "forces": values, "force_unit": system.force_unit,
        "moment_unit": system.moment_unit,
        "global_displacement": global_u * system.disp_scale,
        "local_displacement": local_u * system.disp_scale,
        "displacement_unit": system.disp_unit,
        "stress": stress, "stress_error": stress_error,
    }


def global_extreme(frame, solution, component: str,
                   case: str | None = None, stations: int = 101) -> dict:
    """全结构指定分量的绝对极值及其中心线位置。"""
    from internal_forces import exceeds, member_diagram, peak_index

    from .scene import STRESS, member_scalar

    name = case or solution.primary
    best = None
    for member_id in sorted(frame.members):
        diagram = member_diagram(frame, solution, member_id, name,
                                 stations=stations)
        if component == STRESS:
            # σ 不是内力分量，diagram.extreme 认不出它；按同一条标量口径
            # （scene.member_scalar）自己取极值，免得云图和标注各算各的。
            values = member_scalar(frame, frame.members[member_id],
                                   diagram, STRESS)
            k = peak_index(values)
            x, value = float(diagram.x[k]), float(values[k])
        else:
            x, value = diagram.extreme(component)
        if best is None or exceeds(value, best["raw_value"]):
            best = {"member": member_id, "x": x, "raw_value": value}
    if best is None:
        return {"member": None, "x": 0.0, "value": 0.0, "unit": ""}
    system = _display_system(frame)
    if component == STRESS:
        scale, unit = system.stress_scale, system.stress_unit
    elif component in MOMENTS:
        scale, unit = system.moment_scale, system.moment_unit
    else:
        scale, unit = system.force_scale, system.force_unit
    best.update({"component": component, "case": name,
                 "value": best.pop("raw_value") * scale, "unit": unit,
                 "point": point_on_member(frame, best["member"], best["x"])})
    return best


def show_stress_dialog(parent, probe: dict) -> None:
    """显示探针截面的线性正应力分布；剪应力不在当前截面契约内。"""
    from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    stress = probe.get("stress")
    if stress is None:
        raise ValueError(probe.get("stress_error") or "该截面无法计算正应力。")
    dialog = QDialog(parent)
    dialog.setWindowTitle(
        f"截面正应力 — 杆件 {probe['member']}，x={probe['x']:.3g}")
    dialog.resize(560, 480)
    box = QVBoxLayout(dialog)
    note = QLabel(
        f"截面 {probe['section']}　σmin={stress['min']:+.3f} MPa　"
        f"σmax={stress['max']:+.3f} MPa\n"
        "轴力 + 双向弯曲线性正应力，受拉为正；不含剪力和扭转剪应力。")
    note.setWordWrap(True)
    box.addWidget(note)
    figure = Figure(figsize=(5.2, 3.8))
    canvas = FigureCanvasQTAgg(figure)
    box.addWidget(canvas, 1)
    axis = figure.add_subplot(111)
    grid = stress["grid"]
    image = axis.pcolormesh(grid["z"], grid["y"], grid["sigma"],
                            shading="auto", cmap="coolwarm")
    axis.set_aspect("equal")
    axis.set_xlabel("local z")
    axis.set_ylabel("local y")
    axis.set_title("Normal stress sigma (MPa)")
    figure.colorbar(image, ax=axis, label="MPa")
    figure.tight_layout()
    dialog.exec()
