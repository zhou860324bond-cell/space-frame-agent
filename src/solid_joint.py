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
import json, math, mesh, os, regionToolset

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
    model.fieldOutputRequests['F-Output-1'].setValues(variables=('S','U','RF'))
    for arm in SPEC['arms']:
        d = tuple(arm['direction'])
        end = tuple(d[k]*arm['length_mm'] for k in range(3))
        rv = radial(d)
        radius = 0.25*(arm['outer_diameter_mm']+arm['inner_diameter_mm'])
        if radius <= 1.0e-8:
            radius = 0.25*arm['outer_diameter_mm']
        probe = tuple(end[k]+radius*rv[k] for k in range(3))
        face = instance.faces.findAt((probe,))
        surface = assembly.Surface(name='CUT_%%d' %% arm['member_id'], side1Faces=(face,))
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
    sizes = ([smallest_d / 4.0, smallest_d / 6.0]
             if mesh_sizes_mm is None else [float(value) for value in mesh_sizes_mm])
    if len(sizes) < 2 or any(not math.isfinite(value) or value <= 0.0 for value in sizes):
        raise SolidJointError("mesh_sizes_mm 至少需要两个正数，用于网格收敛检查")
    sizes = sorted(set(sizes), reverse=True)
    if len(sizes) < 2:
        raise SolidJointError("两档网格尺寸不能相同")
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
    coarse, fine = meshes[-2], meshes[-1]
    reference = max(abs(float(fine["p99_abs_principal_mpa"])), 1e-12)
    convergence = abs(float(fine["p99_abs_principal_mpa"])
                      - float(coarse["p99_abs_principal_mpa"])) / reference
    nominal = spec.nominal_normal_mpa
    fine["stress_concentration_factor"] = (
        float(fine["p99_abs_principal_mpa"]) / nominal if nominal > 1e-12 else None)
    summary = {
        "schema": "solid-joint-analysis/v1",
        "backend": "abaqus-6.14",
        "node": spec.node_id,
        "case": spec.case,
        "anchor_member": spec.anchor_member,
        "nominal_normal_mpa": nominal,
        "mesh_convergence_relative": convergence,
        "mesh_converged_5pct": convergence <= 0.05,
        "finest": fine,
        "meshes": meshes,
        "warnings": list(spec.warnings),
        "scope": ("圆钢/圆管节点的线弹性 C3D10 局部实体子模型；切割面采用整体梁模型"
                  "六分量截面力，Kt 使用节点邻域 P99 最大绝对主应力/名义正应力。"),
        "files": {
            "directory": str(target),
            "contour_png": str(target / "solid_joint_mises.png"),
            "finest_odb": str(target / f"solid_joint_{len(sizes)-1}.odb"),
        },
    }
    (target / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
