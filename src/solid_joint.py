"""梁模型节点的 Abaqus 局部实体子模型。

整体结构继续使用快速、可验证的空间梁内核；用户选中一个节点后，本模块把与它
相连的圆形杆件截取为短臂，在 Abaqus/Standard 中做 C3D10 实体分析。切割面通过
参考点传入整体梁模型的六分量截面力，ODB 中只统计节点附近 ``HOTSPOT`` 单元集，
避免把刚性加载端的局部峰值误报成节点应力集中。

第一版有意只支持实心圆和圆管，而且要求同一种各向同性材料。I/H 型节点需要板件、
焊缝、加劲肋和倒圆参数，若缺这些几何就自动拼实体会得到“精确但不是用户结构”的
答案；因此明确拒绝，不做猜测。
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from frame3d import local_axes


class SolidJointError(RuntimeError):
    """局部实体分析不能安全建立或执行。"""


@dataclass(frozen=True)
class JointArm:
    member_id: int
    direction: tuple[float, float, float]
    outer_diameter_mm: float
    inner_diameter_mm: float
    length_mm: float
    force_n: tuple[float, float, float]
    moment_nmm: tuple[float, float, float]
    nominal_normal_mpa: float
    # 壁厚。热点应力外推的两个参考点按 0.4t / 1.0t 定义，没有壁厚就没有 t。
    # 实心圆因此拿不到 IIW 意义上的热点应力——这里如实写 None，而不是拿
    # 半径之类的量凑一个"看起来像 t"的数。
    wall_thickness_mm: float | None = None


@dataclass(frozen=True)
class JointSpec:
    node_id: int
    case: str
    material: str
    elastic_modulus_mpa: float
    poisson: float
    anchor_member: int
    hotspot_radius_mm: float
    nominal_normal_mpa: float
    arms: tuple[JointArm, ...]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def infer_circular_dimensions(section: dict[str, Any]) -> tuple[float, float]:
    """由 A 与 I 唯一反演圆/圆环的外径和内径，保持模型当前长度单位。"""
    area = float(section["A"])
    iy = float(section["Iy"])
    iz = float(section["Iz"])
    torsion = float(section["J"])
    inertia = 0.5 * (iy + iz)
    circular = bool(section.get("circular", False))
    matches_circle = (
        abs(iy - iz) <= 1e-6 * max(iy, iz)
        and abs(torsion - 2.0 * inertia) <= 2e-3 * max(torsion, 2.0 * inertia)
    )
    if not circular and not matches_circle:
        raise SolidJointError(
            f"截面 {section.get('name')!r} 不是可识别的圆钢/圆管；"
            "局部实体第一版只支持圆形截面。")
    sum_sq = 16.0 * inertia / area
    diff_sq = 4.0 * area / math.pi
    outer_sq = 0.5 * (sum_sq + diff_sq)
    inner_sq = 0.5 * (sum_sq - diff_sq)
    if outer_sq <= 0.0 or inner_sq < -1e-8 * outer_sq:
        raise SolidJointError(f"截面 {section.get('name')!r} 的 A/I 与圆形几何不一致")
    outer = math.sqrt(max(outer_sq, 0.0))
    inner = math.sqrt(max(inner_sq, 0.0))
    if inner / max(outer, 1e-30) < 1e-4:
        inner = 0.0
    return outer, inner


def hot_spot_stress_linear(samples: list[tuple[float, float]],
                           thickness_mm: float) -> dict[str, Any]:
    """IIW/DNV 线性热点应力外推。

    为什么需要它：两根管子直接布尔合并、不带焊缝倒圆，相贯线在线弹性里是
    **真正的应力奇异点**——网格越细峰值越高，不收敛到任何有限值。所以
    "最大主应力"这个量本身没有意义，它只是网格密度的函数。

    IIW 的解法是不去取奇异点上的值，而是在离焊趾 0.4t 和 1.0t 两处取表面
    应力，**线性外推回焊趾**。这两点都在奇异影响区之外，外推值因此是网格
    不敏感的——这正是它被写进规范的原因。

        sigma_hs = sigma(0.4t) + (sigma(0.4t) - sigma(1.0t)) * 0.4t / 0.6t
                 = 5/3 * sigma(0.4t) - 2/3 * sigma(1.0t)

    ``samples`` 是沿路径的 (距焊趾距离 mm, 应力 MPa)，不要求恰好落在两个
    参考距离上——中间线性插值。数据必须真的跨过 [0.4t, 1.0t]；只覆盖一半
    就外推等于拿噪声乘以放大系数，这里直接拒绝而不是返回一个数。
    """
    if not thickness_mm or not math.isfinite(float(thickness_mm)) or thickness_mm <= 0.0:
        raise SolidJointError("热点应力外推需要正的壁厚 t（实心截面没有 t）")
    thickness = float(thickness_mm)
    points = sorted((float(d), float(s)) for d, s in samples
                    if math.isfinite(float(d)) and math.isfinite(float(s)))
    if len(points) < 2:
        raise SolidJointError("热点应力外推至少需要两个表面取样点")
    near, far = 0.4 * thickness, 1.0 * thickness
    lo, hi = points[0][0], points[-1][0]
    if lo > near + 1e-9 or hi < far - 1e-9:
        raise SolidJointError(
            f"表面取样范围 [{lo:.3f}, {hi:.3f}] mm 没有覆盖外推区 "
            f"[{near:.3f}, {far:.3f}] mm（t={thickness:.3f}）；"
            "不能只用半边数据外推")

    def at(distance: float) -> float:
        for (d0, s0), (d1, s1) in zip(points, points[1:]):
            if d0 <= distance <= d1:
                if d1 - d0 < 1e-12:
                    return s0
                ratio = (distance - d0) / (d1 - d0)
                return s0 + ratio * (s1 - s0)
        return points[-1][1]

    s_near, s_far = at(near), at(far)
    hot_spot = (5.0 / 3.0) * s_near - (2.0 / 3.0) * s_far
    return {
        "hot_spot_mpa": float(hot_spot),
        "thickness_mm": thickness,
        "sigma_at_0_4t_mpa": float(s_near),
        "sigma_at_1_0t_mpa": float(s_far),
        "method": "IIW linear extrapolation from 0.4t and 1.0t",
        "sample_count": len(points),
    }


def diagnose_peak_convergence(meshes: list[dict[str, Any]],
                              key: str = "max_abs_principal_mpa",
                              tolerance: float = 0.05) -> dict[str, Any]:
    """判断峰值应力到底在收敛，还是在跟着网格发散。

    这个判据存在的理由：原来的 ``mesh_converged_5pct`` 只比较最后两档网格，
    两档之间变化小于 5% 就报"已收敛"。但在应力奇异点上，峰值按
    ``sigma ~ h^(-lambda)`` 增长，**任意相邻两档的变化都可以很小**，只要
    网格没细到位——于是奇异问题会被判成收敛，然后拿一个纯属虚构的 Kt 去写报告。

    三档以上才能分辨这件事：真收敛时逐次变化会**逐档变小**（大致等比缩小），
    发散时逐次变化不缩小甚至变大，且峰值单调上升。同时用
    ``log(sigma) ~ -lambda * log(h)`` 拟合出指数：lambda 明显为正就是奇异特征。

    返回 verdict 之一：``converging`` / ``diverging`` / ``inconclusive``。
    """
    rows = [row for row in meshes
            if row.get("mesh_size_mm") and row.get(key) is not None]
    rows = sorted(rows, key=lambda row: -float(row["mesh_size_mm"]))  # 粗 -> 细
    if len(rows) < 3:
        return {"verdict": "inconclusive", "reason": "少于三档网格，无法分辨收敛与发散",
                "levels": len(rows)}
    sizes = [float(row["mesh_size_mm"]) for row in rows]
    values = [float(row[key]) for row in rows]
    if any(abs(v) <= 1e-12 for v in values):
        return {"verdict": "inconclusive", "reason": "存在零应力档，无法比较",
                "levels": len(rows)}

    changes = [(values[i] - values[i - 1]) / abs(values[i])
               for i in range(1, len(values))]
    monotone_up = all(change > 0.0 for change in changes)
    last, prev = abs(changes[-1]), abs(changes[-2])
    shrinking = last < 0.6 * prev

    # log-log 斜率：sigma ~ h^slope，奇异时 slope < 0（网格越细应力越大）
    mean_x = sum(math.log(s) for s in sizes) / len(sizes)
    mean_y = sum(math.log(abs(v)) for v in values) / len(values)
    num = sum((math.log(s) - mean_x) * (math.log(abs(v)) - mean_y)
              for s, v in zip(sizes, values))
    den = sum((math.log(s) - mean_x) ** 2 for s in sizes)
    slope = num / den if den > 1e-15 else 0.0

    if monotone_up and not shrinking:
        verdict, reason = "diverging", (
            f"峰值随网格加密单调上升且逐次变化不缩小"
            f"（{prev:.1%} → {last:.1%}），log-log 斜率 {slope:.3f}；"
            "这是应力奇异点的特征，峰值不存在有限极限")
    elif last <= tolerance and shrinking:
        verdict, reason = "converging", (
            f"逐次变化在缩小（{prev:.1%} → {last:.1%}）且最后一档 "
            f"≤ {tolerance:.0%}")
    else:
        verdict, reason = "inconclusive", (
            f"逐次变化 {prev:.1%} → {last:.1%}，既不满足收敛判据也不构成"
            "明确的发散特征；需要更多网格档位")
    return {"verdict": verdict, "reason": reason, "levels": len(rows),
            "relative_changes": [float(c) for c in changes],
            "log_log_slope": float(slope), "tolerance": float(tolerance)}


def _physical_end_force(session, member: dict[str, Any], node_id: int,
                        case: str) -> tuple[np.ndarray, np.ndarray]:
    element_ids = session.compilation.mapping.element_ids(int(member["id"]))
    element_id = element_ids[0] if int(member["i"]) == node_id else element_ids[-1]
    element = session.frame.members[element_id]
    force = session.solution[case].member_forces[element_id]
    if element.i == node_id:
        local = np.asarray(force[:6], dtype=float)
    elif element.j == node_id:
        local = np.asarray(force[6:], dtype=float)
    else:
        raise SolidJointError(
            f"物理杆件 {member['id']} 的分析端没有落在节点 {node_id}；"
            "当前不支持带端部刚域偏移的局部实体映射。")
    pi = session.frame.nodes[element.i].xyz
    pj = session.frame.nodes[element.j].xyz
    _, rotation = local_axes(pi, pj, element.ref_vector)
    # member_forces 是“单元对节点”的局部杆端力；切割面上外部结构对局部子模型
    # 的作用取其反号，再转换到全局坐标。
    return -(rotation.T @ local[:3]), -(rotation.T @ local[3:])


def prepare_joint_spec(session, node_id: int, case: str | None = None,
                       anchor_member: int | None = None,
                       arm_length_factor: float = 3.0) -> JointSpec:
    if session.solution is None or session.frame is None or session.compilation is None:
        raise SolidJointError("还没有整体梁模型结果，请先调用 solve_model")
    node_id = int(node_id)
    nodes = {int(item["id"]): item for item in session.model.get("nodes") or []}
    if node_id not in nodes:
        raise SolidJointError(f"模型中没有节点 {node_id}")
    incident = [item for item in session.model.get("members") or []
                if node_id in (int(item["i"]), int(item["j"]))]
    if len(incident) < 2:
        raise SolidJointError("局部实体节点至少需要两根相连杆件")
    if any(any(abs(float(v)) > 1e-12 for v in item.get(key, ()))
           for item in incident for key in ("offset_i", "offset_j")):
        raise SolidJointError("带刚域偏移的杆端暂不能映射到实体子模型")

    result_names = session.solution.all_results()
    chosen = case or session._controlling_case()
    if chosen not in result_names:
        raise SolidJointError(f"没有工况/组合 {chosen!r}，可选 {sorted(result_names)}")
    sections = {item["name"]: item for item in session.model["sections"]}
    materials = {item["name"]: item for item in session.model["materials"]}
    material_names = {item["material"] for item in incident}
    if len(material_names) != 1:
        raise SolidJointError("布尔合并的局部实体节点当前要求所有相连杆件使用同一种材料")
    material_name = next(iter(material_names))
    material = materials[material_name]

    units = session.units
    length_to_mm = units.length_to_m * 1000.0
    modulus_to_mpa = 1e-6 if units.name == "N-m-Pa" else 1.0
    moment_to_nmm = 1000.0 if units.name == "N-m-Pa" else 1.0
    origin = np.array([nodes[node_id][key] for key in "xyz"], dtype=float)
    arms: list[JointArm] = []
    warnings: list[str] = []
    for item in incident:
        other_id = int(item["j"] if int(item["i"]) == node_id else item["i"])
        other = np.array([nodes[other_id][key] for key in "xyz"], dtype=float)
        vector = other - origin
        member_length = float(np.linalg.norm(vector))
        if member_length <= 0.0:
            raise SolidJointError(f"杆件 {item['id']} 长度为零")
        direction = vector / member_length
        section = sections[item["section"]]
        outer, inner = infer_circular_dimensions(section)
        arm_length = min(arm_length_factor * outer, 0.35 * member_length)
        if arm_length < 1.5 * outer:
            raise SolidJointError(
                f"杆件 {item['id']} 太短，无法在节点外留出不受切割边界干扰的实体短臂")
        force, moment = _physical_end_force(session, item, node_id, chosen)
        # 圆形截面双向弯曲极值按平方和组合。这里保留正应力定义，Kt 也使用
        # ODB 的最大绝对主应力，而不是把 Mises 与正应力混为一谈。
        c = 0.5 * outer
        normal = (abs(float(force @ direction)) / float(section["A"])
                  + math.hypot(float(moment @ np.asarray(local_axes(origin, other)[1][1]))
                               * c / float(section["Iy"]),
                               float(moment @ np.asarray(local_axes(origin, other)[1][2]))
                               * c / float(section["Iz"])))
        normal_mpa = normal * modulus_to_mpa
        arms.append(JointArm(
            member_id=int(item["id"]), direction=tuple(float(v) for v in direction),
            outer_diameter_mm=outer * length_to_mm,
            inner_diameter_mm=inner * length_to_mm,
            length_mm=arm_length * length_to_mm,
            force_n=tuple(float(v) for v in force),
            moment_nmm=tuple(float(v) * moment_to_nmm for v in moment),
            nominal_normal_mpa=float(normal_mpa),
            wall_thickness_mm=(0.5 * (outer - inner) * length_to_mm
                               if inner > 1e-9 * outer else None),
        ))

    if anchor_member is None:
        # 优先把通向支座的杆件作为锚固臂；没有时取编号最小者，保证结果可复现。
        supported = {int(item["node"]) for item in session.model.get("supports") or []}
        candidates = [item for item in incident
                      if int(item["i"]) in supported or int(item["j"]) in supported]
        anchor_member = int(min(candidates or incident, key=lambda item: int(item["id"]))["id"])
    if anchor_member not in {arm.member_id for arm in arms}:
        raise SolidJointError(f"锚固杆件 {anchor_member} 不与节点 {node_id} 相连")

    # 若节点上直接有集中力，整体杆端力仍包含它的传递效应，但实体模型不知道力在
    # 节点域中如何扩散；明确告警，避免把连接板/加载头几何假装成已知。
    source_case = next((c for c in session.model.get("load_cases") or []
                        if c.get("name") == chosen), session.model)
    if any(int(load.get("node", -1)) == node_id
           for load in source_case.get("nodal_loads") or []):
        warnings.append("该节点存在直接节点荷载；局部模型按杆端截面力传递，未建加载头几何。")
    nominal = max((arm.nominal_normal_mpa for arm in arms
                   if arm.member_id != anchor_member), default=0.0)
    if nominal <= 1e-12:
        warnings.append("非锚固臂名义正应力接近零，不能给出有意义的应力集中系数。")
    max_diameter = max(arm.outer_diameter_mm for arm in arms)
    return JointSpec(
        node_id=node_id, case=chosen, material=material_name,
        elastic_modulus_mpa=float(material["E"]) * modulus_to_mpa,
        poisson=float(material["nu"]), anchor_member=int(anchor_member),
        hotspot_radius_mm=1.35 * max_diameter,
        nominal_normal_mpa=float(nominal), arms=tuple(arms), warnings=tuple(warnings))


def _script_text(spec: JointSpec, mesh_sizes_mm: list[float], output_json: str) -> str:
    """生成 Abaqus 6.14/Python 2.7 兼容脚本。"""
    data = spec.to_dict()
    return r'''# -*- coding: ascii -*-
from abaqus import mdb, session
from abaqusConstants import *
from odbAccess import openOdb
import json, math, os
# Abaqus registers most Model/Part members only when the matching module is
# imported -- `mdb.Model(...)` alone has no `fieldOutputRequests`, no Coupling,
# no EncastreBC and no Job. Importing only `mesh` and `regionToolset` is what
# made this script die at the first output request. Import the full standard
# set once, up front, instead of chasing one AttributeError per run.
import part, material, section, assembly, step, interaction
import load, mesh, job, sketch, visualization, connectorBehavior
import regionToolset

SPEC = json.loads(%s)
MESH_SIZES = %s
OUTPUT_JSON = %r

def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

def norm(a):
    total = 0.0
    for value in a:
        total += value*value
    return math.sqrt(total)

def radial(direction):
    ref = (0.0, 0.0, 1.0)
    if abs(direction[2]) > 0.9:
        ref = (1.0, 0.0, 0.0)
    v = cross(direction, ref)
    n = norm(v)
    return tuple(x/n for x in v)

def orient(instance, direction, assembly):
    z = (0.0, 0.0, 1.0)
    dot = max(-1.0, min(1.0, direction[2]))
    if dot > 1.0-1.0e-12:
        return
    if dot < -1.0+1.0e-12:
        assembly.rotate(instanceList=(instance.name,), axisPoint=(0.0,0.0,0.0),
                        axisDirection=(1.0,0.0,0.0), angle=180.0)
        return
    axis = cross(z, direction)
    assembly.rotate(instanceList=(instance.name,), axisPoint=(0.0,0.0,0.0),
                    axisDirection=axis, angle=math.degrees(math.acos(dot)))

def surface_samples(odb, frame, size):
    """Sample outer-surface stress versus distance from the weld toe, per arm.

    The toe is found numerically, not from geometry: a surface node of arm k
    that also lies inside another arm's cylinder is still within the merged
    intersection, so the largest axial station among those nodes is the toe.
    Values are the envelope over the circumference, which is what hot-spot
    extrapolation needs -- the critical meridian is wherever the peak is.
    """
    instance = odb.rootAssembly.instances['JOINT-1']
    coords = {}
    for node in instance.nodes:
        coords[node.label] = node.coordinates
    field = frame.fieldOutputs['S'].getSubset(position=ELEMENT_NODAL)
    stress = {}
    for value in field.values:
        label = value.nodeLabel
        if label is None:
            continue
        magnitude = max(abs(float(value.maxPrincipal)), abs(float(value.minPrincipal)))
        if magnitude > stress.get(label, -1.0):
            stress[label] = magnitude

    arms = SPEC['arms']
    tol = 0.35*size
    out = {}
    for arm in arms:
        thickness = arm['wall_thickness_mm']
        if not thickness:
            continue                      # solid circle has no t; IIW needs one
        d = tuple(arm['direction'])
        radius = 0.5*arm['outer_diameter_mm']
        on_surface = []
        for label in stress:
            p = coords[label]
            s = p[0]*d[0]+p[1]*d[1]+p[2]*d[2]
            if s <= 0.0 or s > arm['length_mm']:
                continue
            rad = (p[0]-s*d[0], p[1]-s*d[1], p[2]-s*d[2])
            if abs(norm(rad)-radius) > tol:
                continue
            on_surface.append((s, p, stress[label]))
        if not on_surface:
            continue
        toe = 0.0
        for s, p, _value in on_surface:
            for other in arms:
                if other['member_id'] == arm['member_id']:
                    continue
                od = tuple(other['direction'])
                t_axis = p[0]*od[0]+p[1]*od[1]+p[2]*od[2]
                if t_axis < 0.0 or t_axis > other['length_mm']:
                    continue
                rad = (p[0]-t_axis*od[0], p[1]-t_axis*od[1], p[2]-t_axis*od[2])
                if norm(rad) <= 0.5*other['outer_diameter_mm']+tol and s > toe:
                    toe = s
        band = 0.25*thickness
        bins = {}
        for s, _p, value in on_surface:
            offset = s-toe
            if offset < 0.0 or offset > 1.6*thickness:
                continue
            index = int(offset/band)
            if value > bins.get(index, -1.0):
                bins[index] = value
        if len(bins) < 2:
            continue
        out[str(arm['member_id'])] = [[(k+0.5)*band, bins[k]] for k in sorted(bins)]
    return out

def build(index, size):
    model_name = 'LocalJoint_%%d' %% index
    job_name = 'solid_joint_%%d' %% index
    if model_name in mdb.models:
        del mdb.models[model_name]
    model = mdb.Model(name=model_name)
    assembly = model.rootAssembly
    instances = []
    for arm in SPEC['arms']:
        pname = 'Arm_%%d' %% arm['member_id']
        part = model.Part(name=pname, dimensionality=THREE_D, type=DEFORMABLE_BODY)
        sketch = model.ConstrainedSketch(name='__profile__',
                                         sheetSize=4.0*arm['outer_diameter_mm'])
        ro = 0.5*arm['outer_diameter_mm']
        sketch.CircleByCenterPerimeter(center=(0.0,0.0), point1=(ro,0.0))
        ri = 0.5*arm['inner_diameter_mm']
        if ri > 1.0e-6*ro:
            sketch.CircleByCenterPerimeter(center=(0.0,0.0), point1=(ri,0.0))
        part.BaseSolidExtrude(sketch=sketch, depth=arm['length_mm'])
        del model.sketches['__profile__']
        inst = assembly.Instance(name=pname+'-1', part=part, dependent=ON)
        orient(inst, tuple(arm['direction']), assembly)
        instances.append(inst)
    merged = assembly.InstanceFromBooleanMerge(
        name='Joint', instances=tuple(instances), keepIntersections=ON,
        originalInstances=DELETE, domain=GEOMETRY)
    part = model.parts['Joint']
    material = model.Material(name='JointMaterial')
    material.Elastic(table=((SPEC['elastic_modulus_mpa'], SPEC['poisson']),))
    model.HomogeneousSolidSection(name='JointSection', material='JointMaterial')
    part.SectionAssignment(region=regionToolset.Region(cells=part.cells),
                           sectionName='JointSection')
    elem = mesh.ElemType(elemCode=C3D10, elemLibrary=STANDARD)
    part.setMeshControls(regions=part.cells, elemShape=TET, technique=FREE)
    part.setElementType(regions=(part.cells,), elemTypes=(elem,))
    part.seedPart(size=size, deviationFactor=0.1, minSizeFactor=0.1)
    part.generateMesh()
    assembly.regenerate()
    instance = assembly.instances['Joint-1']

    # Measure only the joint hot zone; exclude rigid cut-face boundary peaks.
    hot_labels = []
    for element in part.elements:
        # MeshElement.connectivity on a native part contains zero-based node indices,
        # not the labels written to the input deck.
        coords = [part.nodes[node_index].coordinates
                  for node_index in element.connectivity]
        center = [0.0, 0.0, 0.0]
        for point in coords:
            for k in range(3):
                center[k] += point[k]
        center = tuple(value/float(len(coords)) for value in center)
        if norm(center) <= SPEC['hotspot_radius_mm']:
            hot_labels.append(element.label)
    if not hot_labels:
        raise RuntimeError('HOTSPOT element set is empty')
    part.Set(name='HOTSPOT', elements=part.elements.sequenceFromLabels(labels=tuple(hot_labels)))
    assembly.regenerate()

    model.StaticStep(name='Static', previous='Initial', nlgeom=OFF,
                     initialInc=1.0, maxInc=1.0, maxNumInc=1)
    # Create the request instead of editing 'F-Output-1'. A model made with
    # mdb.Model() has no default output request at all, so the old setValues
    # was reaching for something that need not exist.
    model.FieldOutputRequest(name='F-Joint', createStepName='Static',
                             variables=('S','U','RF'))
    for arm in SPEC['arms']:
        d = tuple(arm['direction'])
        end = tuple(d[k]*arm['length_mm'] for k in range(3))
        rv = radial(d)
        radius = 0.25*(arm['outer_diameter_mm']+arm['inner_diameter_mm'])
        if radius <= 1.0e-8:
            radius = 0.25*arm['outer_diameter_mm']
        probe = tuple(end[k]+radius*rv[k] for k in range(3))
        # findAt with a tuple *of points* returns a sequence of faces, which is
        # what Surface wants. Wrapping it again in (face,) hands Surface a
        # sequence-of-sequences and it refuses.
        faces = instance.faces.findAt((probe,))
        surface = assembly.Surface(name='CUT_%%d' %% arm['member_id'], side1Faces=faces)
        rp_feature = assembly.ReferencePoint(point=end)
        rp = assembly.referencePoints[rp_feature.id]
        rp_set = assembly.Set(name='RP_%%d' %% arm['member_id'], referencePoints=(rp,))
        model.Coupling(name='COUPLE_%%d' %% arm['member_id'], controlPoint=rp_set,
                       surface=surface, influenceRadius=WHOLE_SURFACE,
                       couplingType=KINEMATIC, localCsys=None,
                       u1=ON,u2=ON,u3=ON,ur1=ON,ur2=ON,ur3=ON)
        if arm['member_id'] == SPEC['anchor_member']:
            model.EncastreBC(name='ANCHOR', createStepName='Initial', region=rp_set)
        else:
            f = arm['force_n']
            m = arm['moment_nmm']
            if max([abs(x) for x in f]) > 0.0:
                model.ConcentratedForce(name='F_%%d' %% arm['member_id'],
                    createStepName='Static', region=rp_set, cf1=f[0], cf2=f[1], cf3=f[2])
            if max([abs(x) for x in m]) > 0.0:
                model.Moment(name='M_%%d' %% arm['member_id'],
                    createStepName='Static', region=rp_set, cm1=m[0], cm2=m[1], cm3=m[2])
    job = mdb.Job(name=job_name, model=model_name, numCpus=1, numDomains=1)
    job.writeInput(consistencyChecking=OFF)
    job.submit(consistencyChecking=OFF)
    job.waitForCompletion()
    odb = openOdb(path=job_name+'.odb', readOnly=True)
    frame = odb.steps['Static'].frames[-1]
    region = odb.rootAssembly.instances['JOINT-1'].elementSets['HOTSPOT']
    values = frame.fieldOutputs['S'].getSubset(region=region).values
    if not values:
        odb.close()
        raise RuntimeError('ODB contains no HOTSPOT stress output')
    mises = sorted(float(v.mises) for v in values)
    principal = sorted([max(abs(float(v.maxPrincipal)), abs(float(v.minPrincipal)))
                        for v in values])
    p99 = principal[min(len(principal)-1, int(0.99*(len(principal)-1)))]
    peak = principal[-1]
    peak_value = values[0]
    for value in values[1:]:
        current = max(abs(float(value.maxPrincipal)), abs(float(value.minPrincipal)))
        old = max(abs(float(peak_value.maxPrincipal)), abs(float(peak_value.minPrincipal)))
        if current > old:
            peak_value = value
    out = {'mesh_size_mm':size, 'nodes':len(part.nodes), 'elements':len(part.elements),
           'hotspot_values':len(values), 'max_mises_mpa':mises[-1],
           'max_abs_principal_mpa':peak, 'p99_abs_principal_mpa':p99,
           'peak_element':int(peak_value.elementLabel),
           'peak_integration_point':int(peak_value.integrationPoint)}
    # Sampling is geometric bookkeeping on top of a solved job. If it fails the
    # solve is still valid, so record why and keep the run rather than losing it.
    try:
        out['surface_samples'] = surface_samples(odb, frame, size)
        out['surface_samples_error'] = None
    except Exception as exc:
        # `as` (not the py2-only comma form) keeps this script parseable by
        # python3 too, which is what lets the offline test syntax-check it.
        out['surface_samples'] = None
        out['surface_samples_error'] = str(exc)
    if index == len(MESH_SIZES)-1:
        viewport = session.Viewport(name='SolidJointViewport')
        viewport.setValues(displayedObject=odb)
        viewport.odbDisplay.setPrimaryVariable(variableLabel='S',
            outputPosition=INTEGRATION_POINT, refinement=(INVARIANT, 'Mises'))
        viewport.view.fitView()
        session.printToFile(fileName='solid_joint_mises', format=PNG,
                            canvasObjects=(viewport,))
    odb.close()
    return out

results = []
for i, size in enumerate(MESH_SIZES):
    results.append(build(i, size))
with open(OUTPUT_JSON, 'w') as handle:
    json.dump({'meshes':results}, handle, indent=2, sort_keys=True)
''' % (repr(json.dumps(data, ensure_ascii=True)),
       repr([float(v) for v in mesh_sizes_mm]), output_json)


def run_joint_analysis(session, node_id: int, case: str | None = None,
                       anchor_member: int | None = None,
                       mesh_sizes_mm: list[float] | None = None,
                       output_dir: Path | str = "results/solid_joint",
                       timeout: float = 1800.0) -> dict[str, Any]:
    """建立、运行并验证两档网格的局部实体节点模型。"""
    from abaqus_backend import find_abaqus, scan_log

    spec = prepare_joint_spec(session, node_id, case, anchor_member)
    smallest_d = min(arm.outer_diameter_mm for arm in spec.arms)
    # 三档，不是两档。两档只能算出"最后一次变化有多大"，而奇异点上相邻两档
    # 的变化本来就可以很小——三档才能看出这个变化是在缩小还是不缩小。
    sizes = ([smallest_d / 4.0, smallest_d / 6.0, smallest_d / 9.0]
             if mesh_sizes_mm is None else [float(value) for value in mesh_sizes_mm])
    if len(sizes) < 3 or any(not math.isfinite(value) or value <= 0.0 for value in sizes):
        raise SolidJointError("mesh_sizes_mm 至少需要三个正数：两档分辨不了收敛与发散")
    sizes = sorted(set(sizes), reverse=True)
    if len(sizes) < 3:
        raise SolidJointError("三档网格尺寸不能有重复")
    executable = find_abaqus()
    if executable is None:
        raise SolidJointError("找不到 Abaqus 6.14 命令，无法运行局部实体分析")

    target = Path(output_dir).resolve() / f"node_{spec.node_id}_{spec.case}"
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="solid_joint_") as temp:
        run_dir = Path(temp)
        script = run_dir / "build_joint.py"
        script.write_text(_script_text(spec, sizes, "solid_joint_results.json"),
                          encoding="ascii")
        try:
            clean_env = os.environ.copy()
            clean_env.pop("PYTHONPATH", None)
            clean_env.pop("PYTHONHOME", None)
            proc = subprocess.run(
                [executable, "cae", f"noGUI={script.name}"], cwd=str(run_dir),
                capture_output=True, text=True, timeout=float(timeout), env=clean_env)
        except subprocess.TimeoutExpired as exc:
            raise SolidJointError(f"局部实体分析超过 {timeout:.0f} 秒") from exc
        logs = (proc.stdout or "") + "\n" + (proc.stderr or "")
        for log_path in run_dir.glob("solid_joint_*.*"):
            if log_path.suffix.lower() in {".msg", ".dat", ".sta", ".log"}:
                logs += "\n" + log_path.read_text(encoding="utf-8", errors="replace")
        scan = scan_log(logs)
        result_path = run_dir / "solid_joint_results.json"
        if proc.returncode != 0 or not scan["ok"] or not result_path.is_file():
            detail = "；".join(scan["fatal"][:3]) or logs[-1000:]
            for path in run_dir.iterdir():
                if path.is_file():
                    shutil.copy2(path, target / path.name)
            raise SolidJointError("Abaqus 局部实体作业失败：" + detail)
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        for path in run_dir.iterdir():
            if path.is_file() and path.suffix.lower() in {
                    ".inp", ".odb", ".sta", ".msg", ".dat", ".log", ".png", ".json", ".py"}:
                shutil.copy2(path, target / path.name)

    meshes = payload["meshes"]
    fine = meshes[-1]
    nominal = spec.nominal_normal_mpa
    diagnosis = diagnose_peak_convergence(meshes)

    # 热点应力外推。取样在 Abaqus 侧完成，可能失败（几何退化、表面点太少），
    # 失败时脚本会写 surface_samples=null——那不该让整个作业作废，这里逐臂
    # 兜住，把原因记下来继续。
    hot_spots: dict[str, Any] = {}
    for arm in spec.arms:
        if arm.member_id == spec.anchor_member:
            continue
        samples = (fine.get("surface_samples") or {}).get(str(arm.member_id))
        if not samples:
            hot_spots[str(arm.member_id)] = {"error": "该臂没有取到表面应力样本"}
            continue
        try:
            hot_spots[str(arm.member_id)] = hot_spot_stress_linear(
                [(row[0], row[1]) for row in samples], arm.wall_thickness_mm)
        except SolidJointError as exc:
            hot_spots[str(arm.member_id)] = {"error": str(exc)}

    # Kt 只在**站得住**的时候给。几何里没有焊缝倒圆，相贯线是应力奇异点；
    # 拿峰值或 P99 去除名义应力得到的"应力集中系数"是网格的函数，不是结构的
    # 性质。所以：有热点外推值就用它算 Kt；没有就明确拒绝并说明理由，
    # 绝不退回去用峰值凑一个数。
    usable = {mid: item.get("hot_spot_mpa") for mid, item in hot_spots.items()
              if item.get("hot_spot_mpa") is not None}
    if usable and nominal > 1e-12:
        governing = max(usable, key=lambda mid: abs(usable[mid]))
        kt: float | None = abs(usable[governing]) / nominal
        kt_basis = (f"IIW 0.4t/1.0t 线性外推热点应力（控制臂 {governing}）/ 名义正应力")
        kt_refused = None
    else:
        governing, kt, kt_basis = None, None, None
        kt_refused = (
            "未给出应力集中系数：" + (
                "名义正应力接近零" if nominal <= 1e-12 else
                "没有拿到可用的热点外推值") +
            "。几何未建焊缝倒圆，相贯线在线弹性下是应力奇异点，"
            "峰值/P99 随网格加密无上界，不能作为 Kt 的分子。")

    summary = {
        "schema": "solid-joint-analysis/v2",
        "backend": "abaqus-6.14",
        "node_id": spec.node_id,
        "case": spec.case,
        "anchor_member": spec.anchor_member,
        "nominal_normal_mpa": nominal,
        "peak_convergence": diagnosis,
        "hot_spot_extrapolation": hot_spots,
        "governing_arm": governing,
        "stress_concentration_factor": kt,
        "stress_concentration_basis": kt_basis,
        "stress_concentration_refused": kt_refused,
        "finest": fine,
        "meshes": meshes,
        "warnings": list(spec.warnings),
        "scope": ("圆钢/圆管节点的线弹性 C3D10 局部实体子模型；切割面采用整体梁模型"
                  "六分量截面力。几何**不含焊缝与倒圆**，相贯线是应力奇异点，"
                  "因此峰值与 P99 仅供观察应力分布，Kt 一律以 IIW 0.4t/1.0t "
                  "线性外推的热点应力为分子。"),
        "files": {
            "directory": str(target),
            "contour_png": str(target / "solid_joint_mises.png"),
            "finest_odb": str(target / f"solid_joint_{len(sizes)-1}.odb"),
        },
    }
    (target / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
