"""规模—耗时—存储对比：一维变带宽 LDLᵀ vs SciPy 稀疏 LU。

讲义讲的是带状/变带宽存储，本项目主链路用稀疏 LU。这个脚本用来回答
「那为什么现代求解器换了方案」——不是靠说的，是靠一张有数的表。

跑法::

    python tools/skyline_bench.py            # 打印 markdown 表
    python tools/skyline_bench.py --out docs/带宽存储对比.md

**每一档都会先对表**：两条路径的位移必须一致，不一致就直接报错退出，
免得拿着一张"跑得快但算错了"的表去写报告。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frame3d import assemble, constrained_dofs, solve            # noqa: E402
from generator import generate_frame                             # noqa: E402
from model_compiler import compile_model                         # noqa: E402
from numbering import matrix_bandwidth, permute, rcm_order       # noqa: E402
from skyline import Skyline, factory                             # noqa: E402

CASES = [
    ("2跨2层", [6.0, 6.0], [3.6] * 2, []),
    ("2跨4层", [6.0, 6.0], [3.6] * 4, []),
    ("3跨6层", [6.0] * 3, [3.6] * 6, []),
    ("3跨6层×2进深", [6.0] * 3, [3.6] * 6, [5.0]),
    ("4跨10层×2进深", [6.0] * 4, [3.6] * 10, [5.0]),
]


def build(spans, storeys, bays) -> dict:
    g = generate_frame(spans=spans, storeys=storeys, bays=bays,
                       column_section="C", beam_section="B", material="M")
    return {
        "units": "N-m-Pa",
        "materials": [{"name": "M", "E": 2.06e11, "nu": 0.3}],
        "sections": [{"name": "C", "A": 1.2e-2, "Iy": 1.6e-4, "Iz": 1.6e-4,
                      "J": 3.2e-4},
                     {"name": "B", "A": 8.0e-3, "Iy": 1.0e-4, "Iz": 6.0e-5,
                      "J": 1.2e-4}],
        "nodes": g["nodes"], "members": g["members"], "supports": g["supports"],
        "load_cases": [{"name": "D", "member_loads":
                        [{"member": m["id"], "w": [0, 0, -12e3]}
                         for m in g["members"]]}],
    }


def timed(fn, repeat: int = 3) -> tuple[float, object]:
    best, out = float("inf"), None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def row(label: str, payload: dict) -> dict:
    frame = compile_model(payload).analysis_model
    t_sparse, ref = timed(lambda: solve(frame))
    t_sky, got = timed(lambda: solve(frame, factorize=factory()))

    scale = max(np.max(np.abs(ref["D"].U)), 1e-30)
    err = float(np.max(np.abs(ref["D"].U - got["D"].U)) / scale)
    if err > 1e-8:
        raise SystemExit(f"{label}：两条路径不一致（相对差 {err:.2e}），先查这个")

    K, _, _ = assemble(frame)
    fixed = constrained_dofs(frame)
    free = np.setdiff1d(np.arange(frame.num_dofs), fixed)
    Kff = K[free][:, free]
    order = rcm_order(Kff)
    before = Skyline.from_matrix(Kff).storage()
    after = Skyline.from_matrix(permute(Kff, order)).storage()
    return {"label": label, "n": before["n"],
            "bw_before": matrix_bandwidth(Kff),
            "bw_after": matrix_bandwidth(permute(Kff, order)),
            "full": before["full"], "banded": after["banded"],
            "sky_before": before["skyline"], "sky_after": after["skyline"],
            "sparse_nnz": int(Kff.nnz),
            "t_sparse": t_sparse, "t_sky": t_sky, "err": err}


_PREAMBLE = """# 一维变带宽存储 vs 稀疏 LU —— 规模、耗时、存储

> **本文件由 `tools/skyline_bench.py` 生成，不要手改。**
> 重跑：`python tools/skyline_bench.py --out docs/带宽存储对比.md`
> 每一档都先对表：两条路径的位移不一致就直接报错退出，表里不会出现算错的数。

判据说明：

* **半带宽**：矩阵实际的 `max(j−i)+1`，不是按节点号估的上界。
* **等带宽 n·b**：讲义 §3-10 的方案，每列都按最大列高铺满。
* **变带宽**：讲义 §4-6 的方案，按各列自己的列高存（含列内的零）。
* **稀疏非零**：SciPy 只存非零元，是三者里最省的；但它需要额外的行列索引，
  且分解会产生填充，所以这一列不是「稀疏 LU 的真实内存」，只是同一件事的下界。
* **（打乱编号）**：把节点号随机对调一遍，结构与答案完全不变，只有编号变了。
  这一档专门用来看讲义 §3-9 八说的编号问题。

"""


def render(rows: list[dict]) -> str:
    head = ("| 算例 | 自由度 n | 半带宽(原编号→RCM) | 满阵 n² | 等带宽 n·b | "
            "变带宽(原→RCM) | 稀疏非零 | 稀疏LU 耗时 | 变带宽 耗时 | 位移相对差 |\n"
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    body = "".join(
        f"| {r['label']} | {r['n']} | {r['bw_before']}→{r['bw_after']} | "
        f"{r['full']:,} | {r['banded']:,} | "
        f"{r['sky_before']:,}→{r['sky_after']:,} | {r['sparse_nnz']:,} | "
        f"{r['t_sparse']*1e3:.1f} ms | {r['t_sky']*1e3:.1f} ms | "
        f"{r['err']:.1e} |\n"
        for r in rows)
    return _PREAMBLE + head + body


def shuffled(payload: dict, seed: int = 0) -> dict:
    """把节点**号**随机对调一遍，结构本身一点不变。

    自由度是按节点号排序后的位次编排的，所以换一套节点号就是换一套自由度编号——
    结构、荷载、答案全都一样，只有带宽和存储量变。讲义 §3-9 八说「编号要让一根杆
    两端的号差尽量小」，这一档就是反过来看：编号没排好到底亏多少，RCM 又能救回
    多少。
    """
    ids = [int(n["id"]) for n in payload["nodes"]]
    shuffled_ids = list(ids)
    np.random.default_rng(seed).shuffle(shuffled_ids)
    remap = dict(zip(ids, shuffled_ids, strict=True))

    out = dict(payload)
    out["nodes"] = [{**n, "id": remap[int(n["id"])]} for n in payload["nodes"]]
    out["members"] = [{**m, "i": remap[int(m["i"])], "j": remap[int(m["j"])]}
                      for m in payload["members"]]
    out["supports"] = [{**s, "node": remap[int(s["node"])]}
                       for s in payload["supports"]]
    out["load_cases"] = [
        {**c, **({"nodal_loads": [{**e, "node": remap[int(e["node"])]}
                                  for e in c["nodal_loads"]]}
                 if c.get("nodal_loads") else {})}
        for c in payload.get("load_cases", [])]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    rows = [row(label, build(s, st, b)) for label, s, st, b in CASES]
    # 再跑两档打乱编号的，用来看 RCM 到底能救回多少
    for label, s, st, b in CASES[-2:]:
        rows.append(row(label + "（打乱编号）", shuffled(build(s, st, b))))
    table = render(rows)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table, encoding="utf-8")
        print(f"写入 {args.out}")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
