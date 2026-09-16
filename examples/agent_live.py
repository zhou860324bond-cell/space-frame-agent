r"""接真实大模型跑一批探针题，看它到底会怎么用工具。

这一步的目的不是"跑出结果"，而是**暴露系统提示的问题**。五道题各自盯一个行为，
输出里的「观察点」就是你要人工判断的东西。跑完把输出贴出来，据此改提示词。

密钥来源，按顺序尝试：

1. 环境变量 DEEPSEEK_API_KEY
2. 仓库根目录下的 deepseek.key 文件（只放密钥一行，已被 .gitignore 忽略）

**推荐第 2 种。** 密钥写在文件里，就不会出现在命令行、终端历史或你贴出来的日志里。
在命令行敲密钥，每贴一次日志就泄露一次。脚本只读取，不打印、不写回。

用法（cmd —— 注意 set 后面不要加引号，引号会变成值的一部分）：

    cd /d C:\Users\你的用户名\Desktop\agent开发
    .venv\Scripts\activate.bat
    pip install openai
    set PYTHONPATH=src
    python examples\agent_live.py                 # 跑全部五题
    python examples\agent_live.py --case 2        # 只跑第 2 题
    python examples\agent_live.py --model deepseek-v4-pro

PowerShell 里则是 .\.venv\Scripts\Activate.ps1 和 $env:PYTHONPATH = "src"。
两种 shell 语法不通用，混着用就会报「文件名、目录名或卷标语法不正确」。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from agent import DeepSeekProvider, Session, run_turn


from console import use_utf8   # 见 src/console.py：别让一个字符打死一次成功的运行

use_utf8()
KEY_FILE = "deepseek.key"


def load_api_key() -> str | None:
    """环境变量优先，其次读本地密钥文件。任何情况下都不打印它的值。"""
    from_env = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if from_env:
        return from_env
    path = Path(__file__).resolve().parent.parent / KEY_FILE
    if path.is_file():
        text = path.read_text(encoding="utf-8-sig").strip()
        if text:
            return text.splitlines()[0].strip()
    return None

# 每题：题干、这题在探什么、期望看到的行为
PROBES = [
    {
        "prompt": "三跨每跨 6 米、两层每层 3.6 米、开间 6 米的钢框架，柱底固接，"
                  "梁上均布荷载 20 kN/m。钢材 E=210 GPa、泊松比 0.3，"
                  "柱截面 A=0.012 Iy=8e-5 Iz=2.4e-4 J=1e-6，"
                  "梁截面 A=0.010 Iy=4e-5 Iz=3e-4 J=8e-7。"
                  "算一下最大位移和柱底反力。",
        "probes": "基本可用性 + 单位换算",
        "expect": [
            "调用 generate_frame，而不是自己逐个列节点坐标",
            "把 20 kN/m 换算成 20000 N/m 传进去，并在回答里写明换算",
            "最终回答里的数字与 query_results 返回的一致（竖向反力合计应为 2400 kN）",
        ],
    },
    {
        "prompt": "帮我算一个三跨两层的框架，梁上 20 kN/m。",
        "probes": "信息不全时追问，还是自己编默认值",
        "expect": [
            "**不应当**直接开始建模——跨度、层高、开间、截面、材料全都没给",
            "应当停下来追问缺失参数",
            "如果它自己填了默认值，那就是提示词第 3 条没生效，要加强",
        ],
    },
    {
        "prompt": "单跨 8 米的简支梁，跨中承受 50 kN 集中力，"
                  "截面 A=0.01 m²、Iy=4e-5、Iz=3e-4、J=8e-7，钢材 E=210 GPa。跨中挠度是多少？",
        "probes": "有解析解的题，看它会不会绕过工具直接心算",
        "expect": [
            "必须调工具求解，不能直接给公式答案",
            "PL³/48EI = 50000×512/(48×2.1e11×3e-4) ≈ 8.47 mm，用来核对它给的数",
            "注意：简支需要在支座上放开转动自由度，看它会不会用 set_model 处理",
        ],
    },
    {
        "prompt": "两跨每跨 7.5 米、单层 4 米的框架，梁两端铰接。"
                  "恒载 15 kN/m、活载 9 kN/m，组合 1.3 恒 + 1.5 活。"
                  "钢材 E=210 GPa，柱截面 A=0.012 Iy=8e-5 Iz=2.4e-4 J=1e-6，"
                  "梁截面 A=0.010 Iy=4e-5 Iz=3e-4 J=8e-7。柱底固接。",
        "probes": "会不会用 beam_release 和 load_case / combos",
        "expect": [
            "梁两端铰接应当传 beam_release=True，而不是把支座改成铰",
            "两个工况应当分别建，再用组合，而不是把荷载先加起来",
            "组合下柱轴力合计应为 (1.3×15+1.5×9)×15 = 495 kN，可手算复核",
        ],
    },
    {
        "prompt": "单跨 6 米、单层 3.6 米的平面框架，柱底只约束竖向位移，"
                  "梁上 20 kN/m。钢材 E=210 GPa、泊松比 0.3，"
                  "柱截面 A=0.012 Iy=8e-5 Iz=2.4e-4 J=1e-6，"
                  "梁截面 A=0.010 Iy=4e-5 Iz=3e-4 J=8e-7。"
                  "按我说的约束条件建模并求解，算最大位移。",
        "probes": "求解失败后会不会自诊断并自修复",
        "expect": [
            "solve_model 应当被主元检查拦下并报奇异",
            "接着应当调 diagnose_supports 而不是反复重试",
            "拿到诊断后应当改约束再算，或者向用户说明缺什么约束",
        ],
    },
]


def run_case(index: int, case: dict, model: str, max_rounds: int,
             api_key: str) -> None:
    print("=" * 78)
    print(f"探针 {index}：{case['probes']}")
    print("=" * 78)
    print(f"题干：{case['prompt']}\n")

    provider = DeepSeekProvider(api_key=api_key, model=model)
    started = time.time()
    try:
        out = run_turn(case["prompt"], provider, session=Session(), max_rounds=max_rounds)
    except Exception as exc:  # noqa: BLE001  探针脚本：网络/鉴权/模型名错误都要原样打给用户看
        print(f"调用失败：{type(exc).__name__}: {exc}")
        print("常见原因：密钥无效、模型名不对、本机无法访问 api.deepseek.com")
        return
    elapsed = time.time() - started

    print(f"工具调用序列（{len(out.tool_calls)} 次，{out.rounds} 轮，{elapsed:.1f} 秒）：")
    log = out.session.tool_log
    for k, (name, args) in enumerate(out.tool_calls, 1):
        shown = {a: v for a, v in args.items() if a not in {"model"}}
        text = str(shown)
        print(f"  {k}. {name}  {text[:110]}{'…' if len(text) > 110 else ''}")
        result = log[k - 1][2] if k <= len(log) and len(log[k - 1]) > 2 else None
        if result is not None:
            mark = "✓" if result.ok else "×"
            body = json.dumps(result.payload, ensure_ascii=False)
            print(f"      {mark} {body[:200]}{'…' if len(body) > 200 else ''}")
    if out.stopped_by_limit:
        print("  ⚠ 达到轮数上限就停了")

    print(f"\n最终回答：\n{out.reply}\n")

    u = provider.usage
    hit = u["cache_hit_tokens"]
    print(f"用量：{u['calls']} 次请求，输入 {u['prompt_tokens']}（缓存命中 {hit}），"
          f"输出 {u['completion_tokens']}")

    print("\n观察点：")
    for item in case["expect"]:
        print(f"  □ {item}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
                        help="模型名；调提示词阶段用便宜的那个就够")
    parser.add_argument("--case", type=int, help="只跑第几题（1-5）")
    parser.add_argument("--max-rounds", type=int, default=10)
    args = parser.parse_args()

    api_key = load_api_key()
    if not api_key:
        print("找不到 API 密钥。二选一：")
        print(f"  1) 在仓库根目录建 {KEY_FILE}，里面只写一行密钥"
              "（推荐：不会出现在终端或日志里）")
        print("  2) cmd:        set DEEPSEEK_API_KEY=你的密钥      注意不要加引号")
        print('     PowerShell: $env:DEEPSEEK_API_KEY = "你的密钥"')
        sys.exit(1)

    cases = [(args.case, PROBES[args.case - 1])] if args.case else list(enumerate(PROBES, 1))
    print(f"模型：{args.model}    题数：{len(cases)}\n")
    for index, case in cases:
        run_case(index, case, args.model, args.max_rounds, api_key)

    print("=" * 78)
    print("跑完了。把上面全部输出贴回对话里，我据此改系统提示。")
    print("重点看第 2 题有没有追问、第 3 题有没有绕过工具直接给数。")


if __name__ == "__main__":
    main()
