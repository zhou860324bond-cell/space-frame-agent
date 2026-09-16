"""Gmsh 网格 + 自研 C3D10 内核的节点局部实体分析。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import solid3d
from solid_joint import (JointSpec, SolidJointError, diagnose_peak_convergence,
                         hot_spot_stress_linear, mesh_reference_length,
                         prepare_joint_spec)


def _gmsh_module():
    try:
        import gmsh
    except (ImportError, OSError) as exc:
        raise SolidJointError(
            "native-solid 需要 Gmsh Python SDK 生成实体网格；请执行 pip install gmsh") from exc
    return gmsh


def _canonical_tet10(raw: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """按几何边中点识别六个二次节点，并统一为本项目/VTK 顺序。"""
    corners = [int(v) for v in raw[:4]]
    mids = [int(v) for v in raw[4:]]

    def ordered(cs: list[int]) -> list[int]:
        remaining = set(mids)
        result = list(cs)
        for i, j in solid3d.EDGE_PAIRS:
            target = 0.5 * (xyz[cs[i]] + xyz[cs[j]])
            found = min(remaining, key=lambda n: float(np.linalg.norm(xyz[n] - target)))
            result.append(found)
            remaining.remove(found)
        return result

    conn = ordered(corners)
    try:
        solid3d._b_matrix(xyz[conn], np.full(4, 0.25))
    except solid3d.Solid3DError:
        corners[1], corners[2] = corners[2], corners[1]
        conn = ordered(corners)
        solid3d._b_matrix(xyz[conn], np.full(4, 0.25))
    return np.asarray(conn, dtype=np.int64)


def generate_joint_mesh(spec: JointSpec, mesh_size_mm: float,
                        hotspot_size_mm: float | None = None
                        ) -> tuple[solid3d.SolidMesh, dict[int, np.ndarray]]:
    """用 Gmsh OpenCASCADE 建圆钢/圆管布尔并集并输出二次四面体。"""
    gmsh = _gmsh_module()
    size = float(mesh_size_mm)
    if not math.isfinite(size) or size <= 0.0:
        raise SolidJointError("实体网格尺寸必须为正数")
    local_size = size if hotspot_size_mm is None else float(hotspot_size_mm)
    if not math.isfinite(local_size) or local_size <= 0.0 or local_size > size:
        raise SolidJointError("热点局部网格尺寸必须为正数且不大于全局尺寸")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(f"native_joint_{spec.node_id}")
        occ = gmsh.model.occ
        volumes: list[tuple[int, int]] = []
        for arm in spec.arms:
            d = arm.direction
            outer = occ.addCylinder(0.0, 0.0, 0.0,
                                    d[0] * arm.length_mm,
                                    d[1] * arm.length_mm,
                                    d[2] * arm.length_mm,
                                    0.5 * arm.outer_diameter_mm)
            current = [(3, outer)]
            if arm.inner_diameter_mm > 1e-9:
                # 内孔向节点内外各伸一点，避免布尔运算在共面端部留下薄片。
                eps = max(1e-5 * arm.length_mm, 1e-5)
                inner = occ.addCylinder(-d[0] * eps, -d[1] * eps, -d[2] * eps,
                                        d[0] * (arm.length_mm + 2.0 * eps),
                                        d[1] * (arm.length_mm + 2.0 * eps),
                                        d[2] * (arm.length_mm + 2.0 * eps),
                                        0.5 * arm.inner_diameter_mm)
                current, _ = occ.cut(current, [(3, inner)],
                                     removeObject=True, removeTool=True)
            volumes.extend(current)
        if len(volumes) > 1:
            volumes, _ = occ.fuse([volumes[0]], volumes[1:],
                                  removeObject=True, removeTool=True)
        occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMin", local_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        if local_size < size:
            # 布尔合并后，节点附近的几何边就是管子相贯线；切割面和
            # 沿杆长的纵边伸到短臂末端，用包围盒可以将它们排除。
            radius = 0.75 * min(arm.length_mm for arm in spec.arms)
            curves = []
            for _dim, tag in gmsh.model.getEntities(1):
                box = gmsh.model.getBoundingBox(1, tag)
                if max(abs(float(v)) for v in box) < radius:
                    curves.append(int(tag))
            if curves:
                distance = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(distance, "CurvesList", curves)
                gmsh.model.mesh.field.setNumber(distance, "Sampling", 120)
                threshold = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
                gmsh.model.mesh.field.setNumber(threshold, "SizeMin", local_size)
                gmsh.model.mesh.field.setNumber(threshold, "SizeMax", size)
                gmsh.model.mesh.field.setNumber(threshold, "DistMin", local_size)
                gmsh.model.mesh.field.setNumber(threshold, "DistMax", 3.0 * local_size)
                gmsh.model.mesh.field.setAsBackgroundMesh(threshold)
        gmsh.option.setNumber("Mesh.ElementOrder", 2)
        gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(2)

        node_tags, flat_xyz, _ = gmsh.model.mesh.getNodes()
        xyz = np.asarray(flat_xyz, dtype=float).reshape((-1, 3))
        tag_to_index = {int(tag): i for i, tag in enumerate(node_tags)}
        types, _element_tags, element_nodes = gmsh.model.mesh.getElements(3)
        rows = []
        # getElements 的三个返回值按单元类型一一对应，是 Gmsh 的 API 约定
        for kind, flat in zip(types, element_nodes, strict=True):
            if int(kind) != 11:                 # Gmsh type 11 = tetrahedron10
                continue
            tags = np.asarray(flat, dtype=np.int64).reshape((-1, 10))
            for tags_row in tags:
                raw = np.asarray([tag_to_index[int(tag)] for tag in tags_row])
                rows.append(_canonical_tet10(raw, xyz))
        if not rows:
            raise SolidJointError("Gmsh 没有生成 10 节点四面体单元")
        mesh = solid3d.SolidMesh(xyz, np.asarray(rows, dtype=np.int64))

        cut_faces: dict[int, np.ndarray] = {}
        tolerance = max(size * 0.08, 1e-5)
        for arm in spec.arms:
            direction = np.asarray(arm.direction)
            axial = xyz @ direction
            indices = np.flatnonzero(np.abs(axial - arm.length_mm) <= tolerance)
            if len(indices) < 3:
                raise SolidJointError(
                    f"Gmsh 未识别出杆件 {arm.member_id} 的切割面节点")
            cut_faces[arm.member_id] = indices
        return mesh, cut_faces
    finally:
        gmsh.finalize()


def _write_vtu_and_png(mesh: solid3d.SolidMesh, result: solid3d.SolidResult,
                       stem: Path) -> tuple[str | None, str | None]:
    """结果始终尝试写 VTU；无 OpenGL 时保留数值结果而不让分析失败。"""
    try:
        import pyvista as pv
        grid = pv.UnstructuredGrid(
            {pv.CellType.QUADRATIC_TETRA: mesh.elements}, mesh.nodes)
        grid.cell_data["Mises_MPa"] = result.element_mises
        grid.cell_data["AbsPrincipal_MPa"] = result.element_abs_principal
        grid.point_data["Displacement_mm"] = result.displacement
        vtu = str(stem.with_suffix(".vtu"))
        grid.save(vtu)
        png = str(stem.with_suffix(".png"))
        plotter = pv.Plotter(off_screen=True, window_size=(1400, 900))
        upper = float(np.percentile(result.element_mises, 99.0))
        plotter.add_mesh(grid, scalars="Mises_MPa", clim=(0.0, upper),
                         cmap="turbo", show_edges=False,
                         scalar_bar_args={"title": "Mises stress (MPa) - P99 clipped"})
        plotter.add_axes()
        plotter.view_isometric()
        plotter.show(screenshot=png, auto_close=True)
        return vtu, png
    except Exception:  # noqa: BLE001  图形环境不是数值求解的前置条件；数值结果已算完
        return None, None


def surface_samples(spec: JointSpec, mesh: solid3d.SolidMesh,
                    result: solid3d.SolidResult,
                    mesh_size_mm: float) -> dict[str, list[list[float]]]:
    """按 Abaqus 后端同一几何规则提取各管臂焊趾外表面应力路径。"""
    stress = solid3d.nodal_abs_principal_envelope(mesh, result.gauss_stress)
    boundary = solid3d.boundary_nodes(mesh)
    xyz = mesh.nodes
    tolerance = 0.35 * float(mesh_size_mm)
    arms = list(spec.arms)
    output: dict[str, list[list[float]]] = {}
    for arm in arms:
        thickness = arm.wall_thickness_mm
        if not thickness:
            continue
        direction = np.asarray(arm.direction)
        radius = 0.5 * arm.outer_diameter_mm
        candidates: list[tuple[float, np.ndarray, float]] = []
        for node in boundary:
            value = stress[node]
            if not math.isfinite(value):
                continue
            point = xyz[node]
            axial = float(point @ direction)
            if axial <= 0.0 or axial > arm.length_mm + tolerance:
                continue
            radial = point - axial * direction
            if abs(float(np.linalg.norm(radial)) - radius) > tolerance:
                continue
            candidates.append((axial, point, float(value)))
        if not candidates:
            continue

        toe = 0.0
        for axial, point, _value in candidates:
            for other in arms:
                if other.member_id == arm.member_id:
                    continue
                other_d = np.asarray(other.direction)
                other_axial = float(point @ other_d)
                if other_axial < -tolerance or other_axial > other.length_mm + tolerance:
                    continue
                other_radial = point - other_axial * other_d
                if (np.linalg.norm(other_radial)
                        <= 0.5 * other.outer_diameter_mm + tolerance):
                    toe = max(toe, axial)

        band = 0.2 * thickness
        bins: dict[int, tuple[float, float]] = {}
        for axial, _point, value in candidates:
            offset = axial - toe
            if offset < 0.0 or offset > 1.6 * thickness:
                continue
            index = int(offset / band)
            if index not in bins or value > bins[index][1]:
                bins[index] = (offset, value)
        if len(bins) >= 2:
            output[str(arm.member_id)] = [
                [float(bins[k][0]), float(bins[k][1])] for k in sorted(bins)]
    return output


def diagnose_hotspot_convergence(values: list[tuple[float, float]],
                                 tolerance: float = 0.05) -> dict[str, Any]:
    """热点应力只有在局部网格加密后稳定，才能用来计算 Kt。"""
    clean = [(float(h), float(s)) for h, s in values
             if math.isfinite(float(h)) and h > 0.0 and math.isfinite(float(s))]
    if len(clean) < 3:
        return {"verdict": "inconclusive", "levels": len(clean),
                "relative_changes": [],
                "reason": "至少需要三档局部网格才能判断热点应力收敛"}
    changes = [abs(b[1] - a[1]) / max(abs(b[1]), abs(a[1]), 1e-12)
               for a, b in zip(clean, clean[1:], strict=False)]
    stable = changes[-1] <= tolerance
    improving = len(changes) < 2 or changes[-1] <= changes[-2] + 1e-12
    if stable and improving:
        verdict = "converging"
        reason = f"最后两档热点应力变化 {changes[-1]:.1%}，已进入稳定区"
    else:
        verdict = "inconclusive"
        reason = ("热点应力逐档变化为 "
                  + " → ".join(f"{v:.1%}" for v in changes)
                  + f"，未满足 {tolerance:.0%} 稳定性门禁")
    return {"verdict": verdict, "levels": len(clean),
            "relative_changes": changes, "tolerance": tolerance, "reason": reason}


def run_native_joint_analysis(session, node_id: int, case: str | None = None,
                              anchor_member: int | None = None,
                              mesh_sizes_mm: list[float] | None = None,
                              output_dir: Path | str = "results/native_solid_joint",
                              *, max_elements: int = 80000) -> dict[str, Any]:
    """使用自研 C3D10 求解器运行一档或多档节点实体网格。"""
    spec = prepare_joint_spec(session, node_id, case, anchor_member)
    reference = mesh_reference_length(spec)
    # native-v1 默认先给一档可交互的工程网格。Python 稀疏直接解比 Abaqus
    # 慢，默认就上 h=t 会让桌面一次作业跨进几十万自由度；用户需要收敛判断时
    # 再显式给三档。Gmsh仍会受薄壁几何约束，在厚度方向放入二次节点。
    sizes = ([2.5 * reference] if mesh_sizes_mm is None
             else sorted({float(v) for v in mesh_sizes_mm}, reverse=True))
    if not sizes or any(not math.isfinite(v) or v <= 0.0 for v in sizes):
        raise SolidJointError("native-solid 网格尺寸必须为正数")
    target = Path(output_dir).resolve() / f"node_{spec.node_id}_{spec.case}"
    target.mkdir(parents=True, exist_ok=True)
    levels: list[dict[str, Any]] = []
    for index, size in enumerate(sizes):
        # 三档远场网格 40/30/20 mm 对应焊趾 16/12/8 mm；C3D10 的
        # 边中点使最细档表面取样间距约为 0.5t，同时守住直接解规模。
        hotspot_size = max(reference, 0.4 * size)
        try:
            mesh, cut_faces = generate_joint_mesh(spec, size, hotspot_size)
        except solid3d.Solid3DError as exc:
            raise SolidJointError(f"自研实体网格检查失败：{exc}") from exc
        if len(mesh.elements) > int(max_elements):
            raise SolidJointError(
                f"自研求解器安全上限为 {max_elements} 个 C3D10；当前网格有 "
                f"{len(mesh.elements)} 个。请先增大网格尺寸或提高 max_elements。")
        if 3 * len(mesh.nodes) > 180000:
            raise SolidJointError(
                f"native-v1 稀疏直接解安全上限为 180000 自由度；当前有 "
                f"{3 * len(mesh.nodes)}。请增大网格尺寸，或使用 Abaqus 后端跑细网格。")
        loads = np.zeros_like(mesh.nodes)
        for arm in spec.arms:
            if arm.member_id == spec.anchor_member:
                continue
            ids = cut_faces[arm.member_id]
            loads[ids] += solid3d.equivalent_nodal_wrench(
                mesh.nodes[ids], arm.force_n, arm.moment_nmm, (0.0, 0.0, 0.0))
        try:
            result = solid3d.solve(
                mesh, spec.elastic_modulus_mpa, spec.poisson,
                cut_faces[spec.anchor_member], loads)
        except solid3d.Solid3DError as exc:
            raise SolidJointError(f"自研 C3D10 求解失败：{exc}") from exc
        stem = target / f"native_joint_{index}"
        vtu, png = _write_vtu_and_png(mesh, result, stem)
        equilibrium = result.reaction + loads
        row = {
            # 峰值和热点由相贯线局部尺寸控制；全局尺寸另行保留，
            # 避免收敛诊断误把 20 mm 的远场网格当成焊趾网格。
            "mesh_size_mm": hotspot_size,
            "global_mesh_size_mm": size,
            "hotspot_mesh_size_mm": hotspot_size,
            "nodes": len(mesh.nodes),
            "elements": len(mesh.elements),
            "max_mises_mpa": float(result.element_mises.max()),
            "p99_mises_mpa": float(np.percentile(result.element_mises, 99.0)),
            "max_abs_principal_mpa": float(result.element_abs_principal.max()),
            "p99_abs_principal_mpa": float(
                np.percentile(result.element_abs_principal, 99.0)),
            "max_displacement_mm": float(
                np.linalg.norm(result.displacement, axis=1).max()),
            "free_residual_norm_n": result.residual_norm,
            "force_balance_n": [float(v) for v in equilibrium.sum(axis=0)],
            "moment_balance_nmm": [float(v) for v in
                                     np.cross(mesh.nodes, equilibrium).sum(axis=0)],
            "vtu": vtu,
            "contour_png": png,
        }
        samples = surface_samples(spec, mesh, result, hotspot_size)
        row["surface_samples"] = samples
        row_hotspots: dict[str, Any] = {}
        for arm in spec.arms:
            if arm.member_id == spec.anchor_member:
                continue
            arm_samples = samples.get(str(arm.member_id))
            if not arm_samples:
                row_hotspots[str(arm.member_id)] = {
                    "error": "该臂没有取到表面应力样本"}
                continue
            try:
                row_hotspots[str(arm.member_id)] = hot_spot_stress_linear(
                    [(sample[0], sample[1]) for sample in arm_samples],
                    arm.wall_thickness_mm)
            except SolidJointError as exc:
                row_hotspots[str(arm.member_id)] = {"error": str(exc)}
        row["hot_spot_extrapolation"] = row_hotspots
        np.savez_compressed(
            stem.with_suffix(".npz"), nodes=mesh.nodes, elements=mesh.elements,
            displacement=result.displacement, reaction=result.reaction,
            element_mises=result.element_mises,
            element_abs_principal=result.element_abs_principal,
            gauss_stress=result.gauss_stress)
        levels.append(row)

    hot_spots = levels[-1]["hot_spot_extrapolation"]
    hotspot_convergence: dict[str, Any] = {}
    for arm in spec.arms:
        if arm.member_id == spec.anchor_member:
            continue
        mid = str(arm.member_id)
        values = []
        for level in levels:
            item = level["hot_spot_extrapolation"].get(mid, {})
            if "hot_spot_mpa" in item:
                values.append((level["hotspot_mesh_size_mm"], item["hot_spot_mpa"]))
        hotspot_convergence[mid] = diagnose_hotspot_convergence(values)
        hotspot_convergence[mid]["values"] = [
            {"hotspot_mesh_size_mm": h, "hot_spot_mpa": value}
            for h, value in values]
    usable = {
        mid: item["hot_spot_mpa"] for mid, item in hot_spots.items()
        if "hot_spot_mpa" in item
        and hotspot_convergence.get(mid, {}).get("verdict") == "converging"
    }
    if usable and spec.nominal_normal_mpa > 1e-12:
        governing = max(usable, key=lambda mid: abs(usable[mid]))
        kt = abs(usable[governing]) / spec.nominal_normal_mpa
        kt_basis = f"native C3D10 0.4t/1.0t 热点外推（控制臂 {governing}）/ 名义正应力"
        kt_refused = None
    else:
        governing, kt, kt_basis = None, None, None
        kt_refused = (
            "未给出正式应力集中系数：0.4t/1.0t 热点路径未覆盖，"
            "或三档局部网格下的热点应力尚未收敛。保留试算值供诊断，"
            "但不使用相贯线奇异峰值或未稳定外推值替代正式 Kt。")
    summary = {
        "schema": "solid-joint-analysis/native-v1",
        "backend": "native-c3d10+gmsh",
        "node_id": spec.node_id,
        "case": spec.case,
        "anchor_member": spec.anchor_member,
        "nominal_normal_mpa": spec.nominal_normal_mpa,
        "peak_convergence": diagnose_peak_convergence(levels),
        "hot_spot_extrapolation": hot_spots,
        "hot_spot_convergence": hotspot_convergence,
        "governing_arm": governing,
        "stress_concentration_factor": kt,
        "stress_concentration_basis": kt_basis,
        "stress_concentration_refused": kt_refused,
        "meshes": levels,
        "warnings": list(spec.warnings),
        "files": {
            "directory": str(target),
            "contour_png": levels[-1]["contour_png"],
            "finest_vtu": levels[-1]["vtu"],
        },
    }
    (target / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
