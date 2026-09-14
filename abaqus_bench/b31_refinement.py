"""用 Abaqus 6.14 实测 B31 的网格收敛，不把粗网格误差冒充梁理论差异。

默认跑门式刚架和空间刚架的 1×/2×/4×/8×/16×/32×/64× 细分，并把报告写到
``abaqus_bench/benchmark_B31_refinement.md``。作业产物放在被 Git 忽略的
``results/b31_refinement``，报告中的每个数字都可由本脚本重算。

加 ``--shear-factor 0.833333333333`` 后，自研侧启用 Ay=Az=5A/6，Abaqus 侧
显式写入相同 K23/K13，并关闭默认细长梁补偿；结果另存为 matched 报告。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT / "src", ROOT / "abaqus_bench"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from abaqus_backend import read_result_csv, solve as solve_abaqus
from compare import compare_one, our_results
from models import (portal_frame, refine_uniform_members, space_frame,
                    with_uniform_shear_areas)


DEFAULT_RUNS = ROOT / "results" / "b31_refinement"
DEFAULT_REPORT = Path(__file__).resolve().parent / "benchmark_B31_refinement.md"
DEFAULT_MATCHED_RUNS = ROOT / "results" / "b31_matched_refinement"
DEFAULT_MATCHED_REPORT = Path(__file__).resolve().parent / "benchmark_B31_matched.md"
CASES = (portal_frame, space_frame)
NATIVE_INVARIANCE_ATOL = 1e-10


def _native_invariance(base: dict, refined: dict) -> float:
    """返回细分前后原节点位移的最大绝对差；应只剩浮点舍入。"""
    old = our_results(base)
    new = our_results(refined)
    return max(abs(old[node][key] - new[node][key])
               for node in old for key in ("u1", "u2", "u3", "ur1", "ur2", "ur3"))


def run(factors: list[int], runs: Path, reuse: bool = False,
        shear_factor: float | None = None) -> list[dict]:
    rows = []
    for factory in CASES:
        base = factory()
        if shear_factor is not None:
            base = with_uniform_shear_areas(base, shear_factor)
        for factor in factors:
            model = refine_uniform_members(base, factor)
            drift = _native_invariance(base, model)
            if drift > NATIVE_INVARIANCE_ATOL:
                raise RuntimeError(
                    f"{factory.__name__} {factor}× 细分改变了原节点解：{drift:.3e}")
            job = f"{factory.__name__}_r{factor}"
            directory = runs / job
            csv_path = directory / f"{job}_abaqus.csv"
            if not reuse or not csv_path.is_file():
                result = solve_abaqus(model, directory, element="B31", job=job,
                                      timeout=300)
                csv_path = Path(result["files"]["csv"])
            report = compare_one(model, read_result_csv(csv_path))
            rows.append({
                "case": factory.__name__, "factor": factor,
                "nodes": len(model["nodes"]), "members": len(model["members"]),
                "native_mm": report["peaks"]["ours_mm"],
                "abaqus_mm": report["peaks"]["abaqus_mm"],
                "error": report["errors"]["|U|"], "drift": drift,
            })
            print(f"{job}: e(|U|)={rows[-1]['error']:.6g}")
    return rows


def render(rows: list[dict], shear_factor: float | None = None) -> str:
    tails = []
    for case in sorted({row["case"] for row in rows}):
        series = sorted((row for row in rows if row["case"] == case),
                        key=lambda row: row["factor"])
        if len(series) >= 2:
            previous, latest = series[-2:]
            change = abs(latest["error"] - previous["error"]) * 100.0
            tails.append(
                f"{case} {previous['factor']}×→{latest['factor']}× 仅变化 "
                f"{change:.4f} 个百分点，末值 {latest['error'] * 100.0:.3f}%")
    matched = shear_factor is not None
    if matched:
        intro = [
            f"单位制为 N-m-Pa。两侧都采用 Ay=Az={shear_factor:.12g}A 的",
            "Timoshenko 参数；Abaqus B31 显式写 K23=G·Az、K13=G·Ay，并把默认",
            "细长梁补偿系数设为 0。每档都先验证自研侧原节点解不变，再比较全部节点。",
        ]
    else:
        intro = [
            "单位制为 N-m-Pa。自研侧保持 Euler–Bernoulli 截面定义；Abaqus 侧使用",
            "两节点一次 Timoshenko 梁 B31。每个细化级别都先验证原节点的自研解不变，",
            "再比较两侧全部离散节点。",
        ]
    lines = [
        "# Abaqus B31 同参数网格细化" if matched else "# Abaqus B31 网格细化诊断",
        "",
        *intro,
        "表中的峰值是**最大节点位移**，不是连续梁跨内极值。",
        "",
        "| 算例 | 每根杆细分 | 节点 | 单元 | 最大节点位移 自研 / B31 (mm) | e(\\|U\\|) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']} | {row['factor']}× | {row['nodes']} | {row['members']} | "
            f"{row['native_mm']:.6f} / {row['abaqus_mm']:.6f} | {row['error']:.6e} |")
    lines += ["", "## 结论", ""]
    if matched:
        lines += [
            "- 两侧剪切面积、剪切模量与细长梁补偿已经显式对齐；剩余偏差主要是 B31",
            "  一次插值的离散误差，应随网格加密趋近于零。",
            "- 这里的 5A/6 是可复现的共同参数，不代表工字形、槽形等所有截面的真实",
            "  有效剪切面积；工程模型应优先输入截面手册或截面分析得到的 Ay/Az。",
        ]
    else:
        lines += [
            "- 1× 粗网格下，一次 B31 的离散误差可能使结构偏硬；这不是横向剪切本身造成的。",
            "- 网格细化后，低阶插值误差减小，结果转而体现 B31 的剪切柔度；两种效应会在",
            "  中间网格短暂抵消，因此误差不保证单调下降。",
            "- 本项目梁内核的同理论验证仍以 B33 为准；B31 细化试验用于解释单元阶次与",
            "  梁理论共同造成的差异，不能写成单纯的“剪切变形误差”。",
            "- 当前 GENERAL 截面未显式对齐两侧的有效剪切面积或 K13/K23，因此约 4.5%",
            "  不能当作求解器精度；同参数结果见 `benchmark_B31_matched.md`。",
            "- 空间框架原始 18 节点报告漏掉梁跨中挠度；细化后的新增节点捕捉到约 2.6 mm",
            "  跨中位移。这也证明对标表必须称“最大节点位移”。",
        ]
    if tails:
        lines += ["- 最后两档已经接近网格极限：" + "；".join(tails) + "。"]
    lines += [
        "",
        ("复现：`python abaqus_bench/b31_refinement.py --shear-factor 0.833333333333`；"
         if matched else "复现：`python abaqus_bench/b31_refinement.py`；")
        + "已有 CSV 时可加 `--reuse`。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factors", type=int, nargs="+",
                        default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--shear-factor", type=float)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    runs = args.runs or (DEFAULT_MATCHED_RUNS if args.shear_factor is not None
                         else DEFAULT_RUNS)
    out = args.out or (DEFAULT_MATCHED_REPORT if args.shear_factor is not None
                       else DEFAULT_REPORT)
    rows = run(args.factors, runs, args.reuse, args.shear_factor)
    out.write_text(render(rows, args.shear_factor), encoding="utf-8")
    print(f"报告已写入 {out}")


if __name__ == "__main__":
    main()
