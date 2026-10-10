"""按参考外形生成拱形空间网格屋盖，通过真实工具求解并核对外力平衡。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np  # noqa: E402
import capsule  # noqa: E402
from agent import Session  # noqa: E402
from sections import circular_tube  # noqa: E402


def generate(output: Path, strengthened: bool = False) -> dict:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    capsule.DEFAULT_CAPSULE_DIR = output / "capsules"
    session = Session()
    calls = []

    def call(name, **arguments):
        started = time.perf_counter()
        result = session.dispatch(name, arguments)
        calls.append({"tool": name, "arguments": arguments, "ok": result.ok,
                      "payload": result.payload, "elapsed_s": round(time.perf_counter() - started, 3)})
        if not result.ok:
            raise RuntimeError(f"{name}: {result.payload}")
        return result.payload

    span, length, rise, eave = 24.0, 36.0, 5.0, 6.0
    nx, ny = 12, 16
    radius = ((span / 2) ** 2 + rise ** 2) / (2 * rise)
    angle = math.asin(span / (2 * radius))
    nodes, members, supports = [], [], []
    roof = {}
    arch_section = "ARCH-P273x10" if strengthened else "ARCH-P219x8"
    arch_diameter, arch_thickness = (.273, .010) if strengthened else (.219, .008)
    sections = [circular_tube(arch_section, arch_diameter, arch_thickness),
                circular_tube("LONG-P168x6", .168, .006),
                circular_tube("DIAG-P159x6", .159, .006),
                circular_tube("COL-P406x16", .406, .016)]
    section_lookup = {s["name"]: s for s in sections}
    material = {"name": "Q355", "E": 206e9, "nu": .3, "density": 7850,
                "yield_stress": 355e6, "allow_tension": 305e6, "allow_compression": 305e6}

    def add_node(x, y, z):
        nid = len(nodes) + 1
        nodes.append({"id": nid, "x": round(x, 9), "y": round(y, 9), "z": round(z, 9)})
        return nid

    def add_member(i, j, section):
        members.append({"id": len(members) + 1, "i": i, "j": j,
                        "section": section, "material": "Q355"})

    for j in range(ny + 1):
        for i in range(nx + 1):
            theta = -angle + 2 * angle * i / nx
            roof[i, j] = add_node(span / 2 + radius * math.sin(theta), length * j / ny,
                                 eave + radius * (math.cos(theta) - math.cos(angle)))
    for j in range(ny + 1):
        for i in range(nx):
            add_member(roof[i, j], roof[i + 1, j], arch_section)
    for j in range(ny):
        for i in range(nx + 1):
            add_member(roof[i, j], roof[i, j + 1], "LONG-P168x6")
        for i in range(nx):
            ends = (roof[i, j], roof[i + 1, j + 1]) if (i + j) % 2 == 0 else (
                roof[i + 1, j], roof[i, j + 1])
            add_member(*ends, "DIAG-P159x6")
    for j in range(0, ny + 1, 2):
        for i in (0, nx):
            base = add_node(span * i / nx, length * j / ny, 0)
            add_member(base, roof[i, j], "COL-P406x16")
            supports.append({"name": f"BASE-{base}", "node": base, "fix": [1] * 6})

    coordinates = {n["id"]: np.array([n["x"], n["y"], n["z"]]) for n in nodes}
    weights = {nid: 0.0 for nid in coordinates}
    half_weights = weights.copy()
    for j in range(ny):
        for i in range(nx):
            corners = [roof[i, j], roof[i + 1, j], roof[i, j + 1], roof[i + 1, j + 1]]
            area = (coordinates[corners[1]][0] - coordinates[corners[0]][0]) * length / ny
            for nid in corners:
                weights[nid] += area / 4
                if i < nx // 2:
                    half_weights[nid] += area / 4
    assert math.isclose(sum(weights.values()), span * length, rel_tol=1e-10)
    assert math.isclose(sum(half_weights.values()), span * length / 2, rel_tol=1e-10)

    self_weight = {nid: 0.0 for nid in coordinates}
    for member in members:
        member_length = np.linalg.norm(coordinates[member["j"]] - coordinates[member["i"]])
        force = section_lookup[member["section"]]["A"] * member_length * material["density"] * 9.81
        self_weight[member["i"]] += force / 2
        self_weight[member["j"]] += force / 2
    vectors = {name: {nid: np.zeros(6) for nid in coordinates} for name in ("DL", "LL", "WX", "WU", "ASYM")}
    for nid in coordinates:
        vectors["DL"][nid][2] = -600 * weights[nid] - self_weight[nid]
        vectors["LL"][nid][2] = -400 * weights[nid]
        vectors["WX"][nid][0] = 250 * weights[nid]
        vectors["WU"][nid][2] = 500 * weights[nid]
        vectors["ASYM"][nid][2] = -400 * half_weights[nid]
    cases = [{"name": name, "nodal_loads": [
        {"name": f"{name}-N{nid}", "node": nid, "load": load.tolist()}
        for nid, load in loads.items() if np.any(load)]} for name, loads in vectors.items()]
    combos = [{"name": "ULS-G", "factors": {"DL": 1.3, "LL": 1.5}},
              {"name": "ULS-GW", "factors": {"DL": 1.3, "LL": 1.5, "WX": 1.5}},
              {"name": "ULS-UP", "factors": {"DL": .9, "WU": 1.5, "WX": 1.5}},
              {"name": "ULS-ASYM", "factors": {"DL": 1.3, "ASYM": 1.5, "WX": 1.5}},
              {"name": "SLS", "factors": {"DL": 1, "LL": 1}}]
    model = {"units": "N-m-Pa", "nodes": nodes, "members": members,
             "materials": [material], "sections": sections, "supports": supports,
             "load_cases": cases, "combos": combos}
    call("set_model", model=model)
    validation = call("validate_model")
    call("solve_model")
    # 由题设面积与质量独立构造外力和力矩，不读取求解器的平衡摘要。
    applied = {name: loads.copy() for name, loads in vectors.items()}
    for combo in combos:
        applied[combo["name"]] = {
            nid: sum((factor * vectors[name][nid] for name, factor in combo["factors"].items()), np.zeros(6))
            for nid in coordinates}
    balance, displacement = {}, {}
    for name, loads in applied.items():
        external_force = sum((v[:3] for v in loads.values()), np.zeros(3))
        external_moment = sum((np.cross(coordinates[nid], v[:3]) + v[3:] for nid, v in loads.items()), np.zeros(3))
        result = session.solution[name]
        reaction_force, reaction_moment = np.zeros(3), np.zeros(3)
        for nid in session.frame.supports:
            r = result.R[session.frame.node_dofs(nid)]
            reaction_force += r[:3]
            reaction_moment += np.cross(coordinates[nid], r[:3]) + r[3:]
        assert np.allclose(reaction_force, -external_force, atol=.01, rtol=1e-8), name
        assert np.allclose(reaction_moment, -external_moment, atol=.1, rtol=1e-8), name
        balance[name] = {"external_kN": (external_force / 1000).tolist(),
                         "support_kN": (reaction_force / 1000).tolist(),
                         "force_residual_N": (reaction_force + external_force).tolist(),
                         "moment_residual_Nm": (reaction_moment + external_moment).tolist()}
        displacement[name] = call("query_results", what="max_displacement", case=name)
    strength = call("check_strength")
    envelope = call("query_envelope", component="N")
    model_path = output / "拱形网格屋盖.json"
    model_path.write_text(json.dumps(session.model, ensure_ascii=False, indent=2), encoding="utf-8")
    record = {"description": "参考用户图片外形的教学模型；图片未提供尺寸，参数与荷载均为演示假设。",
              "arch_section": arch_section,
              "geometry": {"span_m": span, "length_m": length, "rise_m": rise, "eave_m": eave,
                           "nodes": len(nodes), "members": len(members), "supports": len(supports),
                           "roof_nodes": len(roof), "arch_divisions": nx, "length_divisions": ny},
              "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
              "steel_self_weight_kN": sum(self_weight.values()) / 1000,
              "validation": validation, "balance": balance,
              "displacement": displacement, "strength": strength, "envelope": envelope, "calls": calls}
    (output / "拱形网格屋盖工具结果.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    summary = {"geometry": record["geometry"], "steel_self_weight_kN": record["steel_self_weight_kN"],
               "displacement": displacement, "strength_ok": strength["ok"],
               "failed_members": strength["failed_members"],
               "max_stress_ratio": max(m["stress_ratio"] for m in strength["members"])}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/defense_vault")
    parser.add_argument("--strengthened", action="store_true", help="将拱向杆组调整为 P273×10，并重新计算自重")
    args = parser.parse_args()
    generate(args.output, args.strengthened)
