r"""跑评测集，产出一份可直接放进报告的成绩单。

    cd /d C:\Users\你的用户名\Desktop\agent开发
    .venv\Scripts\activate.bat
    set PYTHONPATH=src;evalset
    python evalset\run_eval.py                      # 全部 14 题
    python evalset\run_eval.py --id R01 A01         # 只跑指定题
    python evalset\run_eval.py --category 解析解
    python evalset\run_eval.py --model deepseek-v4-pro

密钥同 agent_live.py：环境变量 DEEPSEEK_API_KEY，或仓库根目录的 deepseek.key。
报告写到 evalset/report.md，同时在终端打印。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from agent import DeepSeekProvider, Session, run_turn
from cases import CASES
from score import Verdict, aggregate, score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from console import use_utf8            # 见 src/console.py：别让一个字符打死一次成功的运行
use_utf8()
from agent_live import load_api_key  # noqa: E402  复用同一套密钥读取，避免两处实现


def run_one(case: dict, api_key: str, model: str, max_rounds: int) -> Verdict:
    """跑一题。任何异常都收成这一题的失败，不许打断整批。"""
    started = time.time()
    provider = None
    try:
        provider = DeepSeekProvider(api_key=api_key, model=model)
        out = run_turn(case["prompt"], provider, session=Session(), max_rounds=max_rounds)
    except Exception as exc:  # noqa: BLE001  评测要跑满所有题：一题崩了记进 Verdict.error 继续下一题
        v = Verdict(case_id=case["id"], category=case["category"])
        v.error = f"{type(exc).__name__}: {exc}"
        v.seconds = time.time() - started
        if provider is not None:
            v.usage = provider.usage
        return v
    out.usage = provider.usage
    v = score(case, out)
    v.seconds = time.time() - started
    v.usage = provider.usage
    if out.stopped_by_limit:
        v.detail["轮数"] = f"达到上限 {max_rounds} 轮仍未收敛"
        v.checks["未超轮数"] = False
    return v


def render(verdicts: list[Verdict], model: str, elapsed: float) -> str:
    agg = aggregate(verdicts)
    lines = ["# 空间刚架 Agent 评测结果", "",
             f"- 模型：`{model}`",
             f"- 题数：{agg['题数']}，通过 {agg['通过']}，**通过率 {agg['通过率']:.0%}**",
             f"- 平均轮数：{agg['平均轮数']}，总耗时 {elapsed:.0f} 秒",
             f"- token：输入 {agg['输入tokens']}（缓存命中 {agg['缓存命中']}），"
             f"输出 {agg['输出tokens']}", "",
             "## 分类成绩", "", "| 类别 | 通过 / 题数 |", "|---|---|"]
    for cat, s in agg["分类"].items():
        lines.append(f"| {cat} | {s['通过']} / {s['题数']} |")

    lines += ["", "## 逐题明细", "", "| 题号 | 类别 | 结果 | 轮数 | 未通过的检查项 |", "|---|---|---|---|---|"]
    for v in verdicts:
        failed = [k for k, ok in v.checks.items() if not ok]
        mark = "通过" if v.passed else ("调用失败" if v.error else "未通过")
        lines.append(f"| {v.case_id} | {v.category} | {mark} | {v.rounds} | "
                     f"{'、'.join(failed) or '—'} |")

    lines += ["", "## 未通过题目的诊断", ""]
    bad = [v for v in verdicts if not v.passed]
    if not bad:
        lines.append("全部通过。")
    for v in bad:
        lines.append(f"### {v.case_id}（{v.category}）")
        if v.error:
            lines.append(f"- 调用失败：{v.error}")
        for k, ok in v.checks.items():
            lines.append(f"- {'✓' if ok else '✗'} {k}")
        for k, d in v.detail.items():
            lines.append(f"  - {k}：{d}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--id", nargs="*", help="只跑这些题号")
    parser.add_argument("--category", help="只跑这一类")
    parser.add_argument("--max-rounds", type=int, default=12)
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "report.md"))
    args = parser.parse_args()

    api_key = load_api_key()
    if not api_key:
        print("找不到 API 密钥。在仓库根目录建 deepseek.key，里面只写一行密钥。")
        sys.exit(1)

    selected = CASES
    if args.id:
        selected = [c for c in selected if c["id"] in set(args.id)]
    if args.category:
        selected = [c for c in selected if c["category"] == args.category]
    if not selected:
        print("没有匹配的题目。")
        sys.exit(1)

    print(f"模型 {args.model}，共 {len(selected)} 题\n")
    started = time.time()
    verdicts = []
    for k, case in enumerate(selected, 1):
        print(f"[{k}/{len(selected)}] {case['id']} {case['category']} … ", end="", flush=True)
        v = run_one(case, api_key, args.model, args.max_rounds)
        verdicts.append(v)
        failed = [name for name, ok in v.checks.items() if not ok]
        print(("通过" if v.passed else f"未通过（{'、'.join(failed) or v.error}）")
              + f"  {v.rounds} 轮 {v.seconds:.0f}s")

    report = render(verdicts, args.model, time.time() - started)
    Path(args.out).write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\n报告已写入 {args.out}")


if __name__ == "__main__":
    main()
