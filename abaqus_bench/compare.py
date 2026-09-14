r"""把本程序的结果与 Abaqus 结果逐节点对比，产出误差表。

误差用全场归一化相对误差，与结构计算文献里常用的定义一致：

    e_X = sqrt( Σ (X_本程序,i − X_abaqus,i)² ) / sqrt( Σ X_abaqus,i² )

逐分量算（U1/U2/U3/UR1/UR2/UR3 与反力），再给位移模长的整体误差。

用法（Python 3，本项目环境）：

    set PYTHONPATH=src;abaqus_bench
    python abaqus_bench\compare.py
    python abaqus_bench\compare.py --element B31
    python abaqus_bench\compare.py --selftest      不需要 Abaqus，自检对比逻辑

预期量级：**B33 与本程序同为 Euler-Bernoulli 格式，误差应落在 1e-8 ~ 1e-12；
B31 是含剪切变形的 Timoshenko 梁，会留下系统性偏差**，细长构件下约 1e-3、
粗短构件下可达 1e-2。两者一起报，差异来源才说得清。
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from frame3d import solve
from model_io import from_dict
from models import all_models


from console import use_utf8   # 见 src/console.py：别让一个字符打死一次成功的运行

use_utf8()
HERE = Path(__file__).resolve().parent
COMPONENTS = ["u1", "u2", "u3", "ur1", "ur2", "ur3"]
REACTIONS = ["rf1", "rf2", "rf3"]


def read_abaqus_results(inp_dir: Path, name: str, element: str
                        ) -> tuple[dict[int, dict[str, float]], str] | None:
    """优先读 .dat（不需要 Abaqus），没有就退回 ODB 导出的 CSV。"""
    stem = f"{name}_{element}"
    dat = inp_dir / f"{stem}.dat"
    if dat.is_file():
        from read_dat import job_status, parse_dat
        status = job_status(dat)
        if not status["completed"]:
            return None
        return parse_dat(dat), ".dat"
    csv_path = inp_dir / f"{stem}_abaqus.csv"
    if csv_path.is_file():
        return read_abaqus_csv(csv_path), "ODB CSV"
    return None


def read_abaqus_csv(path: Path) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            node = int(float(row["node"]))
            out[node] = {k: float(v) for k, v in row.items() if k != "node"}
    return out


def our_results(model: dict) -> dict[int, dict[str, float]]:
    payload = {k: v for k, v in model.items() if k not in ("name", "note")}
    frame = from_dict(payload)
    sol = solve(frame)
    res = sol[sol.primary]
    out: dict[int, dict[str, float]] = {}
    for nid in frame.order():
        d = frame.node_dofs(nid)
        row = {COMPONENTS[k]: float(res.U[d[k]]) for k in range(6)}
        for k in range(3):
            row[REACTIONS[k]] = float(res.R[d[k]])
        out[nid] = row
    return out


# 归一化误差只有一份实现，放在 abaqus_backend 里，界面里的并排对比也用它
from abaqus_backend import normalized_error  # noqa: E402,F401


def compare_one(model: dict, abaqus: dict[int, dict[str, float]]) -> dict:
    ours = our_results(model)
    shared = sorted(set(ours) & set(abaqus))
    report: dict = {"name": model["name"], "nodes_compared": len(shared),
                    "nodes_ours": len(ours), "nodes_abaqus": len(abaqus),
                    "errors": {}, "peaks": {}}
    if not shared:
        report["error"] = "两侧没有共同的节点编号，检查 .inp 是否由同一份模型生成"
        return report

    for key in COMPONENTS + REACTIONS:
        e = normalized_error([ours[n][key] for n in shared],
                             [abaqus[n].get(key, 0.0) for n in shared])
        report["errors"][key] = e

    mag_ours = [math.sqrt(sum(ours[n][k] ** 2 for k in COMPONENTS[:3])) for n in shared]
    mag_theirs = [math.sqrt(sum(abaqus[n].get(k, 0.0) ** 2 for k in COMPONENTS[:3]))
                  for n in shared]
    report["errors"]["|U|"] = normalized_error(mag_ours, mag_theirs)
    peak = int(np.argmax(mag_theirs))
    report["peaks"] = {"node": shared[peak],
                       "ours_mm": round(mag_ours[peak] * 1000, 6),
                       "abaqus_mm": round(mag_theirs[peak] * 1000, 6)}
    return report


def render(reports: list[dict], element: str) -> str:
    lines = [f"# 与 Abaqus 对标结果（{element}）", ""]
    if element == "B33":
        lines += ["B33 是 Abaqus 的三次梁单元，不计横向剪切变形，与本程序的 "
                  "Euler-Bernoulli 格式属于同一套理论，误差应落在数值精度量级。", ""]
    else:
        lines += ["B31 是两节点一次 Timoshenko 梁。粗网格结果同时包含低阶离散误差"
                  "和剪切柔度，不能把两者统称为梁理论差异；网格细化证据见 "
                  "`benchmark_B31_refinement.md`。", ""]

    lines += ["| 算例 | 节点数 | 最大节点位移 本程序 / Abaqus (mm) | e(\\|U\\|) | e(U3) | e(RF3) |",
              "|---|---|---|---|---|---|"]
    for r in reports:
        if r.get("error"):
            lines.append(f"| {r['name']} | — | — | — | — | {r['error']} |")
            continue

        def fmt(key):
            v = r["errors"].get(key)
            return "—" if v is None else f"{v:.2e}"

        p = r["peaks"]
        lines.append(f"| {r['name']} | {r['nodes_compared']} | "
                     f"{p['ours_mm']:.4f} / {p['abaqus_mm']:.4f} | "
                     f"{fmt('|U|')} | {fmt('u3')} | {fmt('rf3')} |")

    lines += ["", "## 逐分量误差", ""]
    for r in reports:
        if r.get("error"):
            continue
        lines.append(f"### {r['name']}")
        cells = []
        for key in COMPONENTS + REACTIONS + ["|U|"]:
            v = r["errors"].get(key)
            cells.append(f"{key} = {'—' if v is None else format(v, '.2e')}")
        lines.append("- " + "，".join(cells))
        lines.append("")
    lines.append("注：参考解整体为零的分量记作 —，归一化误差在那里没有意义。")
    return "\n".join(lines)


def selftest() -> int:
    """不需要 Abaqus：拿本程序自己的结果冒充参考解，验证对比逻辑本身。"""
    print("自检：用本程序结果冒充 Abaqus 结果，误差应当恒为 0\n")
    ok = True
    for model in all_models():
        ours = our_results(model)
        r = compare_one(model, ours)
        worst = max((v for v in r["errors"].values() if v is not None), default=0.0)
        status = "OK" if worst < 1e-12 else "FAIL"
        ok &= worst < 1e-12
        print(f"  {model['name']:<24} 最大误差 {worst:.2e}  {status}")

    print("\n自检：把 U3 整体放大 1%，e(U3) 应当约等于 1e-2")
    model = all_models()[0]
    ours = our_results(model)
    perturbed = {n: dict(row) for n, row in ours.items()}
    for row in perturbed.values():
        row["u3"] *= 1.01
    e = compare_one(model, perturbed)["errors"]["u3"]
    print(f"  实得 e(U3) = {e:.6e}，期望 ≈ 9.90e-03")
    ok &= abs(e - 0.01 / 1.01) < 1e-9
    print("\n" + ("自检通过" if ok else "自检未通过"))
    return 0 if ok else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--element", default="B33", choices=["B33", "B31"])
    parser.add_argument("--inp-dir", default=str(HERE / "inp"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        raise SystemExit(selftest())

    inp_dir = Path(args.inp_dir)
    reports = []
    missing = []
    sources: set[str] = set()
    for model in all_models():
        found = read_abaqus_results(inp_dir, model["name"], args.element)
        if found is None:
            missing.append(f"{model['name']}_{args.element}")
            continue
        results, source = found
        sources.add(source)
        reports.append(compare_one(model, results))

    if missing:
        print("以下算例还没有可用结果（.dat 或 ODB 导出的 CSV），"
              "先在装了 Abaqus 的机器上跑 run_abaqus.bat：")
        for name in missing:
            print("   ", name)
        print()
    if sources:
        print(f"结果来源：{'、'.join(sorted(sources))}\n")
    if not reports:
        raise SystemExit(1)

    text = render(reports, args.element)
    out = Path(args.out or HERE / f"benchmark_{args.element}.md")
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n报告已写入 {out}")


if __name__ == "__main__":
    main()
