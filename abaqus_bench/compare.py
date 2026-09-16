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

    # 误差之外还要记两侧各自的范数。理由见 render() 里 _cell 的注释：
    # 只看归一化误差分不出"参考解为零"和"两边都为零"，而这两种情况
    # 一个是分歧、一个是无事发生，混在一起会把真问题盖掉。
    report["ref_norm"] = {}
    report["our_norm"] = {}
    for key in COMPONENTS + REACTIONS:
        ours_col = [ours[n][key] for n in shared]
        theirs_col = [abaqus[n].get(key, 0.0) for n in shared]
        report["errors"][key] = normalized_error(ours_col, theirs_col)
        report["ref_norm"][key] = float(np.linalg.norm(theirs_col))
        report["our_norm"][key] = float(np.linalg.norm(ours_col))

    mag_ours = [math.sqrt(sum(ours[n][k] ** 2 for k in COMPONENTS[:3])) for n in shared]
    mag_theirs = [math.sqrt(sum(abaqus[n].get(k, 0.0) ** 2 for k in COMPONENTS[:3]))
                  for n in shared]
    report["errors"]["|U|"] = normalized_error(mag_ours, mag_theirs)
    peak = int(np.argmax(mag_theirs))
    report["peaks"] = {"node": shared[peak],
                       "ours_mm": round(mag_ours[peak] * 1000, 6),
                       "abaqus_mm": round(mag_theirs[peak] * 1000, 6)}
    return report


# 同族分量（位移 / 转角 / 反力）单位一致，可以互相比大小。
_FAMILIES = (("u1", "u2", "u3"), ("ur1", "ur2", "ur3"), ("rf1", "rf2", "rf3"))
# 低于本族最大参考值这个倍数，就当作"参考解在这个分量上是零"。
# 实测数据分得很开：真正为零的分量比值在 1e-16 ~ 1e-19，而最小的真实分量
# 也有 1e-3 量级，1e-9 落在中间很宽的空当里。
_ZERO_REL = 1e-9


def _family_of(key: str):
    for fam in _FAMILIES:
        if key in fam:
            return fam
    return None


def _cell(report: dict, key: str) -> str:
    """一个分量该怎么印。

    **为什么不能只印归一化误差**：e = |a-b| / |b| 在 |b| 趋于 0 时会爆成
    1e+13 这种数字，它只反映分母多小，没有物理含义；而如果因此一律记作 —，
    又会把"参考解说这里是零、本程序说不是"这种**真分歧**藏起来——那恰恰是
    最该看见的一种。B31 的 space_frame 就是：u2 / ur1 / rf2 三个分量
    Abaqus 给的是机器零，本程序给的是 1.4e-05 / 3.8e-03 / 4.8e+03，
    而同一份输入换成 B33 时两边吻合到五位有效数字。所以分三种情况印。
    """
    v = report["errors"].get(key)
    fam = _family_of(key)
    if fam is None:                       # |U| 没有同族可比，按原样印
        return "—" if v is None else format(v, ".2e")

    ref = report.get("ref_norm", {})
    our = report.get("our_norm", {})
    scale = max((ref.get(k, 0.0) for k in fam), default=0.0)
    if scale <= 0.0:
        return "—"
    ref_zero = ref.get(key, 0.0) <= scale * _ZERO_REL
    our_zero = our.get(key, 0.0) <= scale * _ZERO_REL
    if ref_zero and our_zero:
        return "—"
    if ref_zero:
        return f"**参考0/本程序 {our.get(key, 0.0):.2e}**"
    return "—" if v is None else format(v, ".2e")


def render(reports: list[dict], element: str) -> str:
    lines = [f"# 与 Abaqus 对标结果（{element}）", ""]
    if element == "B33":
        lines += ["B33 是 Abaqus 的三次梁单元，不计横向剪切变形，与本程序的 "
                  "Euler-Bernoulli 格式属于同一套理论，误差应落在数值精度量级。", ""]
    else:
        lines += [
            "B31 是一点缩减积分的 Timoshenko 梁，含剪切变形与细长度补偿，"
            "与本程序的 Euler-Bernoulli 格式存在系统性偏差。**量级上的差**"
            "（悬臂梁 6.7e-03、门式刚架 1.5e-01）用格式差异解释得通。",
            "",
            "**但 space_frame 的 u2 / ur1 / rf2 三项解释不通，列为待查项。**"
            "这三项 Abaqus 的 B31 给的是机器零，本程序给的是 "
            "1.40e-05 / 3.79e-03 / 4.80e+03；而**同一份输入文件**只把 "
            "`TYPE=B33` 换成 `TYPE=B31`（两份 .inp 的 diff 除此之外没有一行"
            "不同），B33 那次两边吻合到五位有效数字。剪切变形只会让量级差"
            "几个百分点，不会让一整个方向的响应塌成零——所以这不是格式差异，"
            "机理尚未查清。",
            ""]

    lines += ["| 算例 | 节点数 | 最大位移 本程序 / Abaqus (mm) | e(\\|U\\|) | e(U3) | e(RF3) |",
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
        cells = [f"{key} = {_cell(r, key)}"
                 for key in COMPONENTS + REACTIONS + ["|U|"]]
        lines.append("- " + "，".join(cells))
        lines.append("")
    lines += ["注：参考解整体为零、本程序也为零的分量记作 —，归一化误差在那里",
              "没有意义。参考解为零而**本程序不为零**的分量记作 `参考0/本程序 X`：",
              "那不是误差大小的问题，是两边根本不一致，除出来的比值"
              "（动辄 1e+13）只反映分母多小，没有任何物理含义。"]
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
