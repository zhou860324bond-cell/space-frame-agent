r"""把评测集连跑多轮，汇总成「N 次中通过 M 次」的稳定性数据。

一次 100% 不等于稳定 100%。同一版代码、同一套判据连跑几轮，才知道哪些题是
真稳、哪些题是运气。报告里该贴的是这张表，不是某一轮的成绩单。

同时记录**代码指纹**：工具描述一改就等于提示词变了，`temperature=0` 也保证不了
跨版本复现。指纹把结果和代码绑死，别人拿着它才能复现。

    set PYTHONPATH=src;evalset
    python evalset\run_repeat.py --times 3
    python evalset\run_repeat.py --times 5 --id C02      只反复跑某一题
"""

from __future__ import annotations

# 直接 `python evalset\run_repeat.py` 也要能跑，不能要求调用方先设好 PYTHONPATH。
#
# 这几个脚本的用法是写在文档里、让人照着敲的（见 README.md）。原先照着敲会
# 立刻 ModuleNotFoundError——只有走 .bat（里面设了 PYTHONPATH）才活。
# 文档教的命令必须名副其实，所以脚本自己把路径铺好。
# pytest 那边由 pytest.ini 的 `pythonpath` 负责，同一个道理。
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT / "src", Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


import argparse
import hashlib
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from cases import CASES
from run_eval import MAX_ROUNDS, run_one
from score import Verdict


from console import use_utf8   # 见 src/console.py：别让一个字符打死一次成功的运行

use_utf8()
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# 按模块名找，而不是按目录写死——布局变了指纹也还算得出来
#
# **agent_tools.py 必须在列。** 指纹把 agent.py 列进来的本意是盯住工具描述
# （"工具描述一改就等于提示词变了"），但那 1180 行 TOOLS 后来搬去了
# agent_tools.py。只留 agent.py 的话，改一个工具描述指纹察觉不到——
# 指纹还在，保护没了。这是搬文件时漏掉的一处，补回来。
#
# session_*.py 同理：Session 的方法体从 agent.py 搬了出去，而工具的实际
# 行为就在那些方法里。指纹要绑的是"这一版代码"，不是"这一个文件"。
FINGERPRINT_MODULES = ("frame3d.py", "model_io.py", "generator.py",
                       "agent.py", "agent_tools.py",
                       "session_base.py", "session_modeling.py",
                       "session_loads.py", "session_solving.py",
                       "session_query.py", "session_checks.py",
                       "plot3d.py", "cases.py", "score.py")
_SKIP_DIRS = {".venv", "__pycache__", ".pytest_cache", ".git"}


def code_fingerprint() -> dict[str, str]:
    """git 提交号（若有）+ 关键源文件的内容摘要。结果与代码版本绑死。"""
    info: dict[str, str] = {}
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            info["git"] = out.stdout.strip()
            dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                   capture_output=True, text=True, timeout=10)
            if dirty.returncode == 0 and dirty.stdout.strip():
                info["git"] += "（有未提交改动）"
    except (OSError, subprocess.SubprocessError):
        pass

    # 用 os.walk 并就地剪掉 .venv 等目录。rglob 会先把 .venv 里上万个文件全走一遍
    # 再过滤，挂载目录下慢到能把测试拖超时。
    found: dict[str, Path] = {}
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        for name in sorted(filenames):
            if name in FINGERPRINT_MODULES and name not in found:
                found[name] = Path(dirpath) / name

    digest = hashlib.sha256()
    for name in FINGERPRINT_MODULES:
        path = found.get(name)
        if path is None:
            continue
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    info["源码摘要"] = digest.hexdigest()[:12]
    info["计入文件数"] = "%d/%d" % (len(found), len(FINGERPRINT_MODULES))
    missing = [n for n in FINGERPRINT_MODULES if n not in found]
    if missing:
        info["未找到"] = "、".join(missing)
    return info


def repeat(cases: list[dict], runner: Callable[[dict], Verdict],
           times: int, progress: bool = True) -> dict[str, list[Verdict]]:
    """跑 times 轮，返回每题的多轮结果。runner 可注入，便于离线测试汇总逻辑。"""
    results: dict[str, list[Verdict]] = {c["id"]: [] for c in cases}
    for run_index in range(1, times + 1):
        if progress:
            print(f"\n—— 第 {run_index}/{times} 轮 ——")
        for case in cases:
            v = runner(case)
            results[case["id"]].append(v)
            if progress:
                print(f"  {case['id']:<5} {'通过' if v.passed else '未通过':<6} "
                      f"{v.rounds} 轮 {v.seconds:.0f}s")
    return results


def _fail_reasons(runs: list[Verdict]) -> list[str]:
    reasons: list[str] = []
    for v in runs:
        if v.passed:
            continue
        if v.error:
            reasons.append(v.error.split(":")[0])
        reasons += [k for k, ok in v.checks.items() if not ok]
    return sorted(set(reasons))


def render(cases: list[dict], results: dict[str, list[Verdict]],
           times: int, elapsed: float, fingerprint: dict[str, str]) -> str:
    per_run_pass = [sum(1 for c in cases if results[c["id"]][k].passed)
                    for k in range(times)]
    total = len(cases)

    lines = ["# 评测集稳定性（多轮）", "",
             f"- 轮次：{times}，每轮 {total} 题，总耗时 {elapsed:.0f} 秒",
             f"- 每轮通过数：{per_run_pass}",
             f"- 通过率：{statistics.mean(per_run_pass) / total:.1%}"
             + (f"（范围 {min(per_run_pass)}/{total} ~ {max(per_run_pass)}/{total}）"
                if min(per_run_pass) != max(per_run_pass) else "（每轮一致）"), ""]

    lines.append("代码指纹：" + "，".join(f"{k} `{v}`" for k, v in fingerprint.items()))
    lines += ["", "> 工具描述一改就等于提示词变了，`temperature=0` 也保证不了跨版本复现。",
              "> 引用本表数据时请连同指纹一起注明。", "",
              "## 逐题稳定性", "",
              "| 题号 | 类别 | 通过 | 轮数 最小/均值/最大 | 未通过时的检查项 |",
              "|---|---|---|---|---|"]

    unstable = []
    for case in cases:
        runs = results[case["id"]]
        passed = sum(1 for v in runs if v.passed)
        rounds = [v.rounds for v in runs]
        mark = f"{passed}/{times}"
        if passed not in (0, times):
            mark = f"**{mark}**"
            unstable.append(case["id"])
        lines.append(f"| {case['id']} | {case['category']} | {mark} | "
                     f"{min(rounds)} / {statistics.mean(rounds):.1f} / {max(rounds)} | "
                     f"{'、'.join(_fail_reasons(runs)) or '—'} |")

    lines += ["", "## 用量", ""]
    fields = [("prompt_tokens", "输入"), ("cache_hit_tokens", "缓存命中"),
              ("completion_tokens", "输出")]
    for key, label in fields:
        total_tokens = sum(v.usage.get(key, 0)
                           for runs in results.values() for v in runs)
        lines.append(f"- {label}：{total_tokens}")

    lines += ["", "## 结论", ""]
    if not unstable:
        lines.append(f"{times} 轮结果完全一致，未见波动。")
    else:
        lines.append(f"以下题目在 {times} 轮中结果不一致，**不能只贴某一轮的成绩单**："
                     + "、".join(unstable))
        lines.append("")
        lines.append("波动可能来自模型采样、也可能来自题目本身留了多条合理路径。"
                     "把这几题单独多跑几次看分布，再决定是改题还是改工具。")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--times", type=int, default=3)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--id", nargs="*")
    parser.add_argument("--category")
    # 跟随产品自己的默认值，不要在这里另写一个数。
    # 原先硬编码成 12，而 agent.MAX_ROUNDS 是 24——**评测比产品还严**，
    # 于是 C02、M01 在三轮稳定性里因为"超轮数"而失败（C02 轮数 10/11.3/12，
    # 正好顶在天花板上）。那是评测卡出来的，不是模型的能力问题：
    # 真正出厂的产品会给它们 24 轮。评测该测的是出厂状态。
    parser.add_argument("--max-rounds", type=int, default=MAX_ROUNDS)
    parser.add_argument("--out", default=str(HERE / "stability.md"))
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "examples"))
    from agent_live import load_api_key
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

    fingerprint = code_fingerprint()
    print(f"模型 {args.model}，{len(selected)} 题 × {args.times} 轮")
    print("代码指纹：" + "，".join(f"{k}={v}" for k, v in fingerprint.items()))

    started = time.time()
    results = repeat(selected,
                     lambda case: run_one(case, api_key, args.model, args.max_rounds),
                     args.times)
    text = render(selected, results, args.times, time.time() - started, fingerprint)
    Path(args.out).write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"\n报告已写入 {args.out}")


if __name__ == "__main__":
    main()
