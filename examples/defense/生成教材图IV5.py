"""复原教材图 IV-5，并核对原题与可用表格离散模型的等价性。"""
from __future__ import annotations

import csv
import io
import json
import sys
import zipfile
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np  # noqa: E402

import capsule  # noqa: E402
from agent import Session  # noqa: E402
from model_io import validate_payload  # noqa: E402
from model_tables import COLUMNS, LABELS, from_tables, to_tables  # noqa: E402
from sections import solid_circle  # noqa: E402


def generate() -> None:
    output = ROOT / "examples" / "defense" / "教材图IV-5"
    output.mkdir(parents=True, exist_ok=True)
    capsule.DEFAULT_CAPSULE_DIR = ROOT / "results" / "教材图IV5" / "capsules"
    # 原图右手坐标：X 向左下、Y 向右、Z 向上。节点 2 偏离 Y 轴 +X 3 m。
    nodes = [dict(id=1, x=0., y=0., z=0.), dict(id=2, x=3., y=9., z=0.),
             dict(id=3, x=0., y=0., z=3.), dict(id=4, x=0., y=6., z=3.)]
    common = {"section": "实心圆D100", "material": "题设材料E210GPa"}
    members = [{"id": 1, "i": 3, "j": 4, **common},
               {"id": 2, "i": 1, "j": 3, **common},
               {"id": 3, "i": 4, "j": 2, **common}]
    # 所有数值按原题，不叠加自重，不臆设屈服强度或允许应力。
    nodal = [{"name": "节点3沿Y集中力2kN", "node": 3, "load": [0, 2000, 0, 0, 0, 0]},
             {"name": "节点4竖向集中力1kN", "node": 4, "load": [0, 0, -1000, 0, 0, 0]},
             {"name": "节点4绕负X力矩3kNm", "node": 4, "load": [0, 0, 0, -3000, 0, 0]}]
    original = {"units": "N-m-Pa", "nodes": nodes, "members": members,
                "materials": [{"name": common["material"], "E": 210e9, "nu": .3}],
                "sections": [solid_circle(common["section"], .1)],
                "supports": [{"name": "节点1固定支座", "node": 1, "fix": [1]*6},
                             {"name": "节点2固定支座", "node": 2, "fix": [1]*6}],
                "load_cases": [{"name": "图IV5", "nodal_loads": nodal,
                                "member_spans": [
                                    {"name": "横杆前3m均布荷载", "member": 1,
                                     "kind": "partial", "w1": [0, 0, -800], "a": 0, "b": 3},
                                    {"name": "横杆中点沿X集中力4kN", "member": 1,
                                     "kind": "point", "w1": [4000, 0, 0], "a": 3}]}]}
    table_model = deepcopy(original)
    table_model["nodes"].append(dict(id=5, x=0., y=3., z=3.))
    table_model["members"][0]["j"] = 5
    table_model["members"].append({"id": 4, "i": 5, "j": 4, **common})
    case = table_model["load_cases"][0]
    case.pop("member_spans")
    case["member_loads"] = [{"name": "横杆前3m均布荷载", "member": 1, "w": [0, 0, -800]}]
    case["nodal_loads"].append({"name": "节点5沿X集中力4kN", "node": 5,
                                "load": [4000, 0, 0, 0, 0, 0]})

    # 模拟桌面 CSV 导入：所有字段均为字符串，而非 Python 原生数值。
    tables = to_tables(table_model)
    imported = {}
    for key, columns in COLUMNS.items():
        buf = io.StringIO(newline="")
        writer = csv.DictWriter(buf, fieldnames=columns, lineterminator="\r\n")
        writer.writeheader()
        for row in tables[key]:
            if key == "supports":
                row = {k: ("1" if v else "") if k in columns[2:] else v for k, v in row.items()}
            writer.writerow(row)
        content = buf.getvalue()
        (output / f"{LABELS[key]}.csv").write_bytes(content.encode("utf-8-sig"))
        imported[key] = list(csv.DictReader(io.StringIO(content)))
    restored = from_tables(table_model, imported)
    sessions, calls = {}, {}
    for label, model in (("原题三杆", original), ("表格四杆", table_model), ("CSV写回", restored)):
        assert not validate_payload(model), validate_payload(model)
        session = Session()
        records = []
        for tool, args in (("set_model", {"model": model}), ("validate_model", {}), ("solve_model", {})):
            result = session.dispatch(tool, args)
            records.append({"tool": tool, "ok": result.ok, "payload": result.payload})
            assert result.ok, result.payload
        sessions[label], calls[label] = session, records
    # 独立计算外力合力与关于原点力矩，均布合力取其作用区间形心。
    force = np.array([4000., 2000., -3400.])
    moment = (np.cross([0, 0, 3], [0, 2000, 0]) +
              np.cross([0, 6, 3], [0, 0, -1000]) +
              np.cross([0, 3, 3], [4000, 0, 0]) +
              np.cross([0, 1.5, 3], [0, 0, -2400]) + [-3000, 0, 0])
    results = {}
    for label, session in sessions.items():
        solution = session.solution["图IV5"]
        frame = session.frame
        reactions = {nid: solution.R[frame.node_dofs(nid)] for nid in (1, 2)}
        rforce = sum((v[:3] for v in reactions.values()), np.zeros(3))
        rmoment = sum((np.cross([n["x"], n["y"], n["z"]], reactions[n["id"]][:3]) +
                      reactions[n["id"]][3:] for n in nodes if n["id"] in reactions), np.zeros(3))
        np.testing.assert_allclose(rforce + force, 0, atol=1e-7)
        np.testing.assert_allclose(rmoment + moment, 0, atol=1e-7)
        results[label] = {"node_displacements_m_rad": {
            n["id"]: solution.U[frame.node_dofs(n["id"])].tolist() for n in session.model["nodes"]},
            "reactions_N_Nm": {k: v.tolist() for k, v in reactions.items()},
            "force_residual_N": (rforce + force).tolist(),
            "moment_residual_Nm": (rmoment + moment).tolist()}
    for nid in (1, 2, 3, 4):
        u0 = results["原题三杆"]["node_displacements_m_rad"][nid]
        for label in ("表格四杆", "CSV写回"):
            np.testing.assert_allclose(results[label]["node_displacements_m_rad"][nid],
                                       u0, atol=1e-11, rtol=1e-8)
    for label in ("表格四杆", "CSV写回"):
        for nid in (1, 2):
            np.testing.assert_allclose(results[label]["reactions_N_Nm"][nid],
                                       results["原题三杆"]["reactions_N_Nm"][nid], atol=1e-7, rtol=1e-8)
    for filename, label in (("图IV-5_表格建模版.json", "表格四杆"),
                            ("图IV-5_原题三杆版.json", "原题三杆")):
        (output / filename).write_text(json.dumps(sessions[label].model, ensure_ascii=False, indent=2)+"\n",
                                       encoding="utf-8")
    (output / "实际校验与求解结果.json").write_text(json.dumps({
        "interpretation": "节点2=(3,9,0)，两端固定；4kN 沿 +X，3kNm 沿 -X；未添加自重。",
        "applied_force_N": force.tolist(), "applied_moment_Nm": moment.tolist(),
        "verification": "三模型校验求解、独立外力平衡、原节点位移及支座反力等价检查通过。",
        "results": results, "calls": calls}, ensure_ascii=False, indent=2, default=str)+"\n", encoding="utf-8")
    with zipfile.ZipFile(output / "图IV-5_建模文件.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            if path.suffix in (".csv", ".json", ".md"):
                archive.write(path, arcname=path.name)
    print(json.dumps({"output": str(output), "nodes": 5, "members": 4,
                      "applied_force_N": force.tolist(), "applied_moment_Nm": moment.tolist(),
                      "verification": "原题、表格模型与 CSV 写回求解和平衡检查全部通过"}, ensure_ascii=False))


if __name__ == "__main__":
    generate()
