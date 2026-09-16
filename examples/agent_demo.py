"""离线演示：报错 -> 自诊断 -> 改模型 -> 重算。

用 ScriptedProvider 回放，不需要密钥、不依赖网络——答辩现场断网也能跑。
脚本只规定「模型会怎么调工具」，**所有数字都是工具真算出来的**。

    python examples\agent_demo.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent import ScriptedProvider, Session, run_turn


from console import use_utf8   # 见 src/console.py：别让一个字符打死一次成功的运行

use_utf8()
MATERIALS = [{"name": "STEEL", "E": 2.1e11, "nu": 0.3}]
SECTIONS = [
    {"name": "COLUMN", "A": 0.012, "Iy": 8.0e-5, "Iz": 2.4e-4, "J": 1.0e-6},
    {"name": "BEAM", "A": 0.010, "Iy": 4.0e-5, "Iz": 3.0e-4, "J": 8.0e-7},
]


def call(cid, name, **args):
    return {"tool_calls": [{"id": cid, "name": name, "arguments": args}]}


def main() -> None:
    session = Session()
    record: list[tuple[str, dict]] = []

    inner = session.dispatch

    def dispatch(name, arguments):
        # 演示脚本在这里制造并修复缺陷：generate_frame 之后写坏柱底约束，
        # 诊断之后再补回来。真实运行时这两步由模型自己通过 set_model 完成。
        if len(record) == 2:                       # 已完成 2 次调用，正要做第 3 次
            for support in session.model["supports"]:
                support["fix"] = [1, 0, 1, 1, 1, 1]      # 漏掉 Y 向约束
        if len(record) == 4:                       # 诊断完毕，重算之前修复
            for support in session.model["supports"]:
                support["fix"] = [1, 1, 1, 1, 1, 1]
        result = inner(name, arguments)
        record.append((name, result.payload if result.ok else {"失败": result.payload}))
        return result

    session.dispatch = dispatch

    provider = ScriptedProvider([
        call("1", "define_materials_and_sections", materials=MATERIALS, sections=SECTIONS),
        call("2", "generate_frame", spans=[6, 6, 6], storeys=[3.6, 3.6], bays=[6],
             beam_load=20000, base="fixed"),
        call("3", "solve_model"),
        call("4", "diagnose_supports"),
        call("5", "solve_model"),
        call("6", "query_results", what="max_displacement"),
        call("7", "query_results", what="reactions"),
        {"content": "柱底缺少 Y 向约束导致结构成为机构，补齐后重新求解，"
                    "最大位移与静力平衡均已核对。"},
    ])

    print("=" * 70)
    print("空间刚架 Agent 离线演示：求解失败 -> 自诊断 -> 改模型 -> 重算")
    print("=" * 70)

    out = run_turn("三跨 6 米、两层层高 3.6 米、开间 6 米的钢框架，梁上 20 kN/m",
                   provider, session=session)

    for index, ((name, args), (_, payload)) in enumerate(zip(out.tool_calls, record, strict=True), 1):
        shown = {k: v for k, v in args.items() if k not in {"materials", "sections", "model"}}
        print(f"\n[{index}] {name}  {json.dumps(shown, ensure_ascii=False)}")

        if "失败" in payload:
            detail = payload["失败"]
            print(f"     × {detail.get('error') or detail.get('errors')}")
            if detail.get("hint"):
                print(f"       提示：{detail['hint']}")
        elif name == "generate_frame":
            print(f"     ✓ 展开为 {payload['summary']['nodes']} 节点 "
                  f"{payload['summary']['members']} 杆件（模型没有逐个列坐标）")
        elif name == "diagnose_supports":
            for mode in payload.get("modes", [])[:1]:
                who = "、".join(f"节点 {p['node']} {p['direction']}"
                                for p in mode["participants"][:3])
                print(f"     ✓ 检出刚体模态：{who}")
                print(f"       {payload['note']}")
        elif name == "solve_model":
            for case, info in payload["cases"].items():
                print(f"     ✓ 工况 {case}：最大位移 {info['max_displacement_mm']:.4f} mm "
                      f"@节点 {info['at_node']}，平衡 "
                      f"{'通过' if info['equilibrium_ok'] else '未通过'}")
        elif name == "query_results":
            print(f"     ✓ {json.dumps(payload, ensure_ascii=False)[:150]}")

    print("\n" + "-" * 70)
    print(f"模型最终回复：{out.reply}")
    print(f"共 {out.rounds} 轮，{len(out.tool_calls)} 次工具调用")
    print("-" * 70)
    print("以上每个数字都来自工具返回值；脚本只规定了调用顺序，没有规定任何结果。")


if __name__ == "__main__":
    main()
