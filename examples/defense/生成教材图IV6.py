"""按教材图 IV-6 建模，并核对集中力拆分前后的求解与整体平衡。"""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np  # noqa: E402

import capsule  # noqa: E402
from agent import Session  # noqa: E402
from sections import solid_circle  # noqa: E402


def generate() -> None:
    output = ROOT / "examples" / "defense" / "教材图IV-6"
    output.mkdir(parents=True, exist_ok=True)
    capsule.DEFAULT_CAPSULE_DIR = ROOT / "results" / "教材图IV6" / "capsules"
    # 节点 2 是原点，三根柱高 2 m，两个水平跨度分别沿 +X 与 +Y。
    coordinates = {1: (2, 0, 0), 2: (0, 0, 0), 3: (0, 2, 0),
                   4: (2, 0, 2), 5: (0, 0, 2), 6: (0, 2, 2)}
    common = {"section": "实心圆D100", "material": "题设材料E210GPa"}
    model = {
        "units": "N-m-Pa",
        "nodes": [dict(id=n, x=x, y=y, z=z) for n, (x, y, z) in coordinates.items()],
        "members": [dict(id=n, i=i, j=j, **common)
                    for n, i, j in ((1, 1, 4), (2, 4, 5), (3, 2, 5),
                                    (4, 5, 6), (5, 3, 6))],
        "materials": [{"name": common["material"], "E": 210e9, "nu": 0.3}],
        "sections": [solid_circle(common["section"], 0.1)],
        "supports": [dict(name=f"节点{n}固定支座", node=n, fix=[1] * 6)
                     for n in (1, 2, 3)],
        "load_cases": [{
            "name": "图IV6",
            "nodal_loads": [
                dict(name="节点4绕正X力矩5kNm", node=4, load=[0, 0, 0, 5000, 0, 0]),
                dict(name="节点6绕正Y力矩5kNm", node=6, load=[0, 0, 0, 0, 5000, 0])],
            "member_loads": [dict(name="杆件3沿负Y均布4kN每米", member=3,
                                  w=[0, -4000, 0])],
            "member_spans": [
                dict(name="杆件2中点沿正Y集中力10kN", member=2,
                     kind="point", w1=[0, 10000, 0], a=1),
                dict(name="杆件4中点沿正X集中力10kN", member=4,
                     kind="point", w1=[10000, 0, 0], a=1)]}]
    }
    table_model = deepcopy(model)
    table_model["nodes"] += [dict(id=7, x=1, y=0, z=2), dict(id=8, x=0, y=1, z=2)]
    table_model["members"][1]["j"] = 7
    table_model["members"][3]["j"] = 8
    table_model["members"] += [dict(id=6, i=7, j=5, **common),
                               dict(id=7, i=8, j=6, **common)]
    case = table_model["load_cases"][0]
    case.pop("member_spans")
    case["nodal_loads"] += [
        dict(name="节点7沿正Y集中力10kN", node=7, load=[0, 10000, 0, 0, 0, 0]),
        dict(name="节点8沿正X集中力10kN", node=8, load=[10000, 0, 0, 0, 0, 0])]

    # 独立外力与关于原点的力矩；均布合力位于柱高中点，不读取求解器等效荷载。
    force = np.array([10000., 10000. - 4000. * 2, 0.])
    moment = (np.cross([1, 0, 2], [0, 10000, 0])
              + np.cross([0, 1, 2], [10000, 0, 0])
              + np.cross([0, 0, 1], [0, -8000, 0]) + [5000, 5000, 0])
    records, solutions = {}, {}
    for label, current in (("原题五杆", model), ("表格七杆", table_model)):
        session = Session()
        calls = []
        for tool, args in (("set_model", {"model": current}), ("validate_model", {}),
                           ("solve_model", {})):
            result = session.dispatch(tool, args)
            calls.append(dict(tool=tool, args=args, ok=result.ok, payload=result.payload))
            assert result.ok, result.payload
        frame, result = session.frame, session.solution["图IV6"]
        u = {n: result.U[frame.node_dofs(n)].tolist() for n in frame.nodes}
        reactions = {n: result.R[frame.node_dofs(n)] for n in (1, 2, 3)}
        residual_force = force + sum((r[:3] for r in reactions.values()), np.zeros(3))
        residual_moment = moment + sum((np.cross(coordinates[n], r[:3]) + r[3:]
                                       for n, r in reactions.items()), np.zeros(3))
        np.testing.assert_allclose(residual_force, 0, atol=1e-7)
        np.testing.assert_allclose(residual_moment, 0, atol=1e-7)
        if label == "原题五杆":
            for member in range(1, 6):
                for component in ("N", "Vy", "Vz", "T", "My", "Mz"):
                    args = dict(member=member, component=component, case="图IV6", stations=3)
                    queried = session.dispatch("query_diagram", args)
                    assert queried.ok, queried.payload
                    calls.append(dict(tool="query_diagram", args=args, ok=True,
                                      payload=queried.payload))
        records[label] = dict(node_displacements_m_rad=u,
                             reactions_N_Nm={n: r.tolist() for n, r in reactions.items()},
                             force_residual_N=residual_force.tolist(),
                             moment_residual_Nm=residual_moment.tolist(), calls=calls)
        solutions[label] = session.model
    for n in range(1, 7):
        np.testing.assert_allclose(records["原题五杆"]["node_displacements_m_rad"][n],
                                   records["表格七杆"]["node_displacements_m_rad"][n],
                                   atol=1e-11, rtol=1e-8)
    for n in (1, 2, 3):
        np.testing.assert_allclose(records["原题五杆"]["reactions_N_Nm"][n],
                                   records["表格七杆"]["reactions_N_Nm"][n], atol=1e-7)
    for label, filename in (("原题五杆", "图IV-6_原题五杆版.json"),
                             ("表格七杆", "图IV-6_表格建模版.json")):
        (output / filename).write_text(json.dumps(solutions[label], ensure_ascii=False,
                                                  indent=2) + "\n", encoding="utf-8")
    (output / "实际校验与求解结果.json").write_text(json.dumps(dict(
        applied_force_N=force.tolist(), applied_moment_Nm=moment.tolist(), results=records,
        interpretation="③号杆单一全长 -Y 均布 4kN/m；两个横杆中点集中力及节点4/6力矩；未加自重。",
        verification="两模型真实求解、原六节点位移和支座反力等价、独立整体外力与力矩平衡通过。",
        limitation="尚未取得教材标准答案；求解通过不表示荷载释图已获得教材答案确认。"),
        ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(dict(output=str(output), verification="校验、求解、等价和平衡检查通过"),
                     ensure_ascii=False))


if __name__ == "__main__":
    generate()
