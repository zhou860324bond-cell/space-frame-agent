"""算例：24 m 单跨门式刚架，多轮走一遍全流程。

这个脚本用 `ScriptedProvider` 回放一串工具调用——**每个数字都是真算的**，
只有"该调哪个工具"这一步是写死的而不是大模型决定的。用途有二：

* 没有网络 / 不想花 token 时，照样能把整条链路演一遍；
* 作为多轮对话的**回归基线**：真模型跑出来的工具序列该长这样。

要接真模型，把 provider 换成 DeepSeekProvider 即可，其余不用动：

    from agent import DeepSeekProvider
    chat = Conversation(DeepSeekProvider(api_key=...), session=Session())
    chat.ask("……同样的中文描述……")

结构描述见 DESCRIPTION，那就是要喂给模型的原话。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from console import use_utf8            # 见 src/console.py：别让一个字符打死一次成功的运行
use_utf8()

from agent import ScriptedProvider, Session          # noqa: E402
from conversation import Conversation                # noqa: E402
from sections import i_section                       # noqa: E402

DESCRIPTION = """\
某单层钢结构厂房的一榀横向刚架：单跨 24 m，檐口高 7.5 m，屋脊比檐口高 1.2 m
（坡度约 1:10），柱脚铰接。柱用焊接 H700×250×10×16，斜梁用 H900×250×10×20。
材料 Q355：E = 206 GPa，泊松比 0.3，密度 7850 kg/m³。

荷载（沿斜梁的竖向线荷载）：
  恒载 D  8 kN/m 向下
  活载 L  5 kN/m 向下
  风吸 W  3 kN/m 向上
组合：1.3D + 1.5L、1.3D + 1.5W、1.3D + 1.5L + 0.9W。

请建模求解，然后回答：按 24 m 跨算挠跨比满不满足 L/400、哪个组合控制、
稳定和自振特性如何。若挠度不够，告诉我斜梁的 Iz 至少要做到多少。
"""

COLUMN = i_section("COLUMN", 0.700, 0.250, 0.010, 0.016)
RAFTER = i_section("RAFTER", 0.900, 0.250, 0.010, 0.020)
MATERIALS = [{"name": "Q355", "E": 2.06e11, "nu": 0.3, "density": 7850.0}]

SPAN, EAVE, RISE = 24.0, 7.5, 1.2
LIMIT_MM = SPAN / 400 * 1000            # L/400，单位 mm


def call(cid: str, name: str, **args) -> dict:
    return {"tool_calls": [{"id": cid, "name": name, "arguments": args}]}


def build_script(rafters: list[int]) -> list[dict]:
    """一个称职的模型面对上面那段描述，应当调出来的工具序列。"""
    def spans(w: float) -> list[dict]:
        return [{"member": m, "w": [0.0, 0.0, w]} for m in rafters]

    return [
        # 第一轮：建模 → 定荷载 → 求解
        call("t1", "define_materials_and_sections",
             materials=MATERIALS, sections=[COLUMN, RAFTER]),
        call("t2", "generate_portal_frame", spans=[SPAN], eave_height=EAVE,
             ridge_rise=RISE, column_section="COLUMN", rafter_section="RAFTER",
             material="Q355", base="pinned"),
        call("t3", "set_load_cases",
             cases=[{"name": "D", "member_loads": spans(-8e3)},
                    {"name": "L", "member_loads": spans(-5e3)},
                    {"name": "W", "member_loads": spans(+3e3)}],
             combos=[{"name": "1.3D+1.5L", "factors": {"D": 1.3, "L": 1.5}},
                     {"name": "1.3D+1.5W", "factors": {"D": 1.3, "W": 1.5}},
                     {"name": "1.3D+1.5L+0.9W",
                      "factors": {"D": 1.3, "L": 1.5, "W": 0.9}}]),
        call("t4", "solve_model"),
        {"content": "模型已建好并求解，三个组合都通过了静力平衡校核。"},

        # 第二轮：挠度与控制组合
        # 跨度必须显式传：斜梁是两根杆，杆长只有半跨，
        # 不传的话工具只会给"杆长/挠度"，那不是挠跨比
        call("t5", "query_results", what="max_deflection", span=SPAN),
        call("t6", "query_envelope", component="Mz"),
        call("t7", "query_envelope", component="N"),
        {"content": "挠度与控制组合见上。"},

        # 第三轮：稳定与自振
        call("t8", "buckling_analysis", num_modes=3),
        call("t9", "modal_analysis", num_modes=6),
        {"content": "稳定与自振特性见上。"},

        # 第四轮：梁截面要多大才够
        call("t10", "sweep", what="section_property", target="RAFTER", prop="Iz",
             # 把现有截面 2.4664e-3 也放进去，表里才看得出"差了多少"
             values=[5e-4, 1e-3, 2e-3, 2.4664e-3, 3e-3, 4e-3, 6e-3],
             metric="max_deflection", limit=LIMIT_MM),
        {"content": "扫描结果见上。"},

        # 第五轮：出报告
        call("t11", "write_report", fmt="both", filename="门式刚架分析报告"),
        {"content": "报告已生成。"},
    ]


def run() -> Conversation:
    """跑完整个算例，把每一轮的关键结论打出来。"""
    probe = Session()
    probe.define_materials_and_sections(MATERIALS, [COLUMN, RAFTER])
    gen = probe.generate_portal_frame(
        spans=[SPAN], eave_height=EAVE, ridge_rise=RISE,
        column_section="COLUMN", rafter_section="RAFTER",
        material="Q355", base="pinned")
    rafters = gen.payload["rafter_member_ids"]

    chat = Conversation(ScriptedProvider(build_script(rafters)), session=Session())
    for question in ("建模并求解", "挠度和控制组合怎么样",
                     "稳定和自振呢", "梁截面要多大才满足 L/400", "出个报告"):
        chat.ask(question)
    return chat


if __name__ == "__main__":
    chat = run()
    s = chat.session

    print(f"斜梁编号 {sorted(s.frame.members)[:2]}…　"
          f"节点 {len(s.frame.nodes)}　杆件 {len(s.frame.members)}")
    print()
    d = s.query_results(what="max_deflection", span=SPAN).payload
    print(f"最大杆件挠度 {d['magnitude_mm']:.2f} mm @ 杆件 {d['at_member']} "
          f"x={d['at_x_m']:.2f} m")
    print(f"挠跨比 L/{d['span_over_deflection']:.0f}"
          f"（按设计跨度 {SPAN:g} m；杆长只有 {d['member_length_m']:.2f} m，"
          f"不传跨度会算成 1/{d['member_length_over_deflection']:.0f}）")
    print(f"限值 L/400 = {LIMIT_MM:.0f} mm　→ "
          f"{'满足' if d['magnitude_mm'] <= LIMIT_MM else '不满足'}")
    print()
    for comp in ("Mz", "N"):
        e = s.query_envelope(component=comp).payload
        print(f"{comp} 最不利 {e['peak']:.2f} {e['unit']} @ 杆件 {e['at_member']} "
              f"x={e['at_x_m']:.2f} m　控制组合 {e['governing_case']}")
    print()
    b = s.buckling_analysis(num_modes=3).payload
    print(f"临界荷载因子 λ = {b['critical_factor']:.2f}　"
          f"最受压杆件 {b['most_compressed_member']}（{b['its_axial_kN']:.1f} kN）")
    m = s.modal_analysis(num_modes=3).payload
    print("自振频率 " + "、".join(f"{r['frequency_Hz']:.2f}" for r in m["modes"])
          + " Hz")
    print()
    sweep = s.sweep(what="section_property", target="RAFTER", prop="Iz",
                    values=[5e-4, 1e-3, 2e-3, 2.4664e-3, 3e-3, 4e-3, 6e-3],
                    metric="max_deflection", limit=LIMIT_MM).payload
    print("斜梁 Iz 扫描（限值 L/400 = %.0f mm）：" % LIMIT_MM)
    for row in sweep["rows"]:
        flag = "满足" if row["metric"] <= LIMIT_MM else "不满足"
        print(f"   Iz = {row['value']:.4e} m⁴ → {row['metric']:6.2f} mm  "
              f"L/{SPAN * 1000 / row['metric']:.0f}  {flag}")
    print(f"   最省取值 Iz = {sweep['smallest_sufficient']:.3e} m⁴")
    print()
    print("每轮调用的工具：")
    for k, row in enumerate(chat.transcript(), 1):
        print(f"  {k}. {row['user']}　→　" + " → ".join(row["tools"]))
