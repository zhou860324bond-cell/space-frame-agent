"""评测集题目。

每题声明**可自动判定**的期望，判定一律读会话状态（模型、解、工具调用序列），
不读回复文本——文本可以说得天花乱坠，状态骗不了人。唯一的例外是"追问"这一类：
它的正确行为恰恰是不建模，只能看有没有调工具 + 回复里有没有问句。

期望里的数值全部有解析解或可手算复核，写在 note 里，便于答辩时逐条对照。
"""

from __future__ import annotations

E = 210e9
COL = {"name": "COLUMN", "A": 0.012, "Iy": 8e-5, "Iz": 2.4e-4, "J": 1e-6}
BEAM = {"name": "BEAM", "A": 0.010, "Iy": 4e-5, "Iz": 3e-4, "J": 8e-7}
SPEC = ("钢材 E=210 GPa、泊松比 0.3，"
        "柱截面 A=0.012 Iy=8e-5 Iz=2.4e-4 J=1e-6，"
        "梁截面 A=0.010 Iy=4e-5 Iz=3e-4 J=8e-7。")


def _frame_size(nx: int, ny: int, nz: int) -> dict[str, int]:
    """规则刚架的节点数与杆件数，nx/ny/nz 为各向节点数。"""
    columns = nx * ny * (nz - 1)
    beams_x = (nx - 1) * ny * (nz - 1)
    beams_y = nx * (ny - 1) * (nz - 1)
    return {"nodes": nx * ny * nz, "members": columns + beams_x + beams_y}


def _two_span_frame(solved: bool):
    """两跨每跨 6 m、单层 4 m 的平面刚架，柱底固接；杆 1-3 为柱、4-5 为梁。

    给「在已有模型上动手」的题用：预演确认、静默失败都得先有一个模型。
    这里直接调 Session 的工具，不经过大模型——题目考的是后面那一步。
    """
    from agent import Session
    s = Session()
    s.dispatch("define_materials_and_sections", {
        "materials": [{"name": "STEEL", "E": E, "nu": 0.3}], "sections": [COL, BEAM]})
    s.dispatch("generate_frame", {"spans": [6, 6], "storeys": [4],
                                  "column_section": "COLUMN", "beam_section": "BEAM",
                                  "material": "STEEL", "base": "fixed"})
    if solved:
        s.dispatch("set_load_cases", {"cases": [{"name": "DL", "member_loads": [
            {"member": m, "w": [0, 0, -20e3]} for m in (4, 5)]}]})
        s.dispatch("solve_model", {})
    return s


# 闭合解里反复用到的量，单独写出来让 note 里的数字能逐条复核
_EI_WEAK = E * 4e-5                                # 悬臂柱弱轴抗弯刚度
_RHO_A = 7850 * 0.01                               # 线质量 kg/m
_PORTAL_RAFTER = (12.0 ** 2 + 1.5 ** 2) ** 0.5     # 24 m 跨、1.5 m 矢高的单根斜梁长


CASES: list[dict] = [
    # ---------------- 规则框架：拓扑展开与整体平衡 ----------------
    {
        "id": "R01", "category": "规则框架",
        "prompt": f"三跨每跨 6 米、两层每层 3.6 米、开间 6 米的钢框架，柱底固接，"
                  f"梁上均布荷载 20 kN/m。{SPEC}算最大位移和柱底反力。",
        "checks": {
            "topology": _frame_size(4, 2, 3),
            "numeric": {"vertical_total_kN": (2400.0, 1e-4)},
            "tools_required": ["generate_frame"],
            "tools_forbidden": ["set_model"],
        },
        "note": "20 根梁 × 6 m × 20 kN/m = 2400 kN",
    },
    {
        "id": "R02", "category": "规则框架",
        "prompt": f"单跨 6 米、单层 3.6 米的平面框架，柱底固接，梁上 20 kN/m。{SPEC}算最大位移。",
        "checks": {
            "topology": _frame_size(2, 1, 2),
            "numeric": {"vertical_total_kN": (120.0, 1e-4)},
            "tools_required": ["generate_frame"],
        },
        "note": "1 根梁 × 6 m × 20 kN/m = 120 kN",
    },
    {
        "id": "R03", "category": "规则框架",
        "prompt": f"四跨每跨 7 米、三层每层 3.3 米、两个开间各 6 米的钢框架，柱底固接，"
                  f"所有梁上 18 kN/m。{SPEC}算竖向支座反力总和。",
        "checks": {
            "topology": _frame_size(5, 3, 4),
            "numeric": {"vertical_total_kN": (((4 * 7) * 3 + (2 * 6) * 5) * 3 * 18.0, 1e-4)},
            "tools_required": ["generate_frame"],
        },
        "note": "X 向梁 4×7×3 排 + Y 向梁 2×6×5 排，每层合计 144 m，三层 432 m × 18 kN/m",
    },

    # ---------------- 信息不全：必须追问，不许编默认值 ----------------
    {
        "id": "Q01", "category": "信息不全",
        "prompt": "帮我算一个三跨两层的框架，梁上 20 kN/m。",
        "checks": {"must_ask": True,
                   "missing_terms": ["跨度", "层高", "截面", "材料", "开间"]},
        "note": "跨度、层高、开间、截面、材料全缺",
    },
    {
        "id": "Q02", "category": "信息不全",
        "prompt": "算一下我这个钢框架的位移。",
        "checks": {"must_ask": True,
                   "missing_terms": ["跨度", "层高", "截面", "荷载", "约束"]},
        "note": "什么都没给",
    },
    {
        "id": "Q03", "category": "信息不全",
        "prompt": "两跨每跨 6 米、单层 4 米的平面框架，柱底固接，梁上 15 kN/m，算最大位移。",
        "checks": {"must_ask": True, "missing_terms": ["截面", "材料", "弹性模量", "泊松比"]},
        "note": "几何与荷载齐了，但截面和材料没给——这一项最容易被它自己填掉",
    },

    # ---------------- 解析解：数值必须对得上闭合解 ----------------
    {
        "id": "A01", "category": "解析解",
        "prompt": "单跨 8 米的简支梁，跨中承受 50 kN 集中力，截面 A=0.01 m²、"
                  "Iy=4e-5、Iz=3e-4、J=8e-7，钢材 E=210 GPa、泊松比 0.3。跨中挠度是多少？",
        "checks": {"numeric": {"max_displacement_mm": (50e3 * 8**3 / (48 * E * 3e-4) * 1000, 2e-3)}},
        "note": "PL³/48EI = 8.4656 mm",
    },
    {
        "id": "A02", "category": "解析解",
        "prompt": "长 5 米的悬臂梁，固定端在原点、沿 X 方向水平放置，自由端作用竖直向下 30 kN。"
                  "截面 A=0.01、Iy=4e-5、Iz=3e-4、J=8e-7，钢材 E=210 GPa、泊松比 0.3。"
                  "自由端挠度是多少？",
        "checks": {"numeric": {"max_displacement_mm": (30e3 * 5**3 / (3 * E * 3e-4) * 1000, 2e-3)}},
        "note": "PL³/3EI = 19.841 mm",
    },
    {
        "id": "A03", "category": "解析解",
        "prompt": "跨度 10 米的简支梁承受向下均布荷载 12 kN/m，截面 A=0.01、Iy=4e-5、"
                  "Iz=3e-4、J=8e-7，钢材 E=210 GPa、泊松比 0.3。跨中挠度和支座反力各是多少？"
                  "请把梁划分成至少 8 个单元。",
        "checks": {"numeric": {"max_displacement_mm": (5 * 12e3 * 10**4 / (384 * E * 3e-4) * 1000, 5e-2),
                               "vertical_total_kN": (120.0, 1e-3)}},
        "note": "5wL⁴/384EI = 24.802 mm；反力合计 12×10 = 120 kN",
    },

    # ---------------- 多工况与组合 ----------------
    {
        "id": "C01", "category": "多工况",
        "prompt": f"两跨每跨 7.5 米、单层 4 米的框架，梁两端铰接。恒载 15 kN/m、活载 9 kN/m，"
                  f"组合 1.3 恒 + 1.5 活。{SPEC}柱底固接。给出组合下的支座反力。",
        "checks": {
            "numeric": {"vertical_total_kN": ((1.3 * 15 + 1.5 * 9) * 15, 1e-4)},
            "tools_required": ["generate_frame", "set_load_cases"],
            "min_load_cases": 2,
            "combos_defined": True,
        },
        "note": "(1.3×15+1.5×9)×15 = 495 kN；必须分两个工况再组合，不许先把荷载加起来",
    },
    {
        "id": "C02", "category": "多工况",
        "prompt": f"三跨每跨 6 米、两层每层 3.6 米的平面框架，柱底固接。"
                  f"恒载在所有梁上 20 kN/m，活载在所有梁上 8 kN/m，"
                  f"另有一个水平风载工况：顶层左侧节点施加 30 kN 沿 X 正向。"
                  f"做两个组合：1.3恒+1.5活，以及 1.0恒+1.5风。{SPEC}",
        "checks": {
            "tools_required": ["set_load_cases"],
            "min_load_cases": 3,
            "combos_defined": True,
            "all_cases_equilibrium": True,
        },
        "note": "三工况两组合。组合名由模型自定，所以判据不能挂在名字上——"
                "改用与命名无关的硬性质：每个工况和组合都必须满足整体静力平衡",
    },

    # ---------------- 机构：算前守门与诊断 ----------------
    {
        "id": "M01", "category": "机构诊断",
        "prompt": f"单跨 6 米、单层 3.6 米的平面框架，柱底只约束竖向位移，梁上 20 kN/m。{SPEC}"
                  f"按我说的约束条件建模并求解，算最大位移。",
        "checks": {"must_fail_to_solve": True, "reply_mentions": ["机构", "约束"]},
        "note": "柱底只约束 Z，整体可平移、可倾覆，应当被主元检查拦下并诊断",
    },
    {
        "id": "M02", "category": "机构诊断",
        "prompt": "单跨 8 米的梁沿 X 方向放置，两端支座都只约束三个平动方向、"
                  "放开全部转动，跨中作用竖直向下 40 kN。截面 A=0.01、Iy=4e-5、Iz=3e-4、"
                  "J=8e-7，钢材 E=210 GPa、泊松比 0.3。请算出跨中挠度，"
                  "如果我给的约束有问题，你可以修正后再算，但要说明改了什么。",
        "checks": {"must_handle_trap": True, "reply_mentions": ["扭转", "rx"],
                   "numeric": {"max_displacement_mm": (40e3 * 8**3 / (48 * E * 3e-4) * 1000, 2e-3)}},
        "note": "两端都放开 rx，整根梁可绕自身轴自由扭转——三维刚架的经典陷阱。"
                "预先避开与撞墙后修好都算对，判据只看算没算出来、说没说清，"
                "PL³/48EI = 6.772 mm。"
                "M01 考的是相反的一面：用户明令按其条件建模时不许擅自改",
    },

    # ---------------- 单位陷阱 ----------------
    {
        "id": "U01", "category": "单位换算",
        "prompt": "单跨 6000 mm、单层 3600 mm 的平面框架，柱底固接，梁上均布荷载 20 N/mm。"
                  "钢材 E=210000 MPa、泊松比 0.3，柱截面 A=12000 mm²、Iy=8e7 mm⁴、"
                  "Iz=2.4e8 mm⁴、J=1e6 mm⁴，梁截面 A=10000 mm²、Iy=4e7 mm⁴、"
                  "Iz=3e8 mm⁴、J=8e5 mm⁴。算最大位移和支座反力。",
        "checks": {"numeric": {"vertical_total_kN": (120.0, 1e-3)}},
        "note": "全部用 mm-N-MPa 给的，等价于 6 m 跨、20 kN/m，反力仍是 120 kN。"
                "考它会不会换算成 N-m-Pa（或追问）",
    },

    # ---------------- 2026-09 扩充：Beta 0.1 之后进主链路的能力 ----------------
    # 原 14 题只考「建模—求解—读数」。之后加进来的门式刚架生成、预演确认、静默失败
    # 拦截、P-Δ / 材料非线性、屈曲 / 模态、扫参 / 包络 / 校核 / 报告一道题都没碰过，
    # 评测分数因此说明不了这些能力在真实模型手里好不好用。
    {
        "id": "R04", "category": "规则框架",
        "prompt": f"单跨门式刚架：跨度 24 米、檐口高 7.5 米、屋脊比檐口高 1.5 米，"
                  f"共 3 榀、榀距 6 米，柱底铰接。所有屋面斜梁上作用沿杆长竖直向下 5 kN/m "
                  f"的均布荷载，其余构件不加荷载。{SPEC}斜梁用梁截面，柱与系杆用柱截面。"
                  f"算竖向支座反力总和。",
        "checks": {
            "numeric": {"vertical_total_kN": (5.0 * 2 * _PORTAL_RAFTER * 3, 1e-3)},
            "tools_any": [["generate_portal_frame", "generate_bent"]],
            "tools_forbidden": ["set_model"],
        },
        "note": "斜梁长 √(12²+1.5²) = 12.093 m，每榀两根、共 3 榀，72.56 m × 5 kN/m = 362.8 kN",
    },

    {
        "id": "Q04", "category": "信息不全",
        "prompt": f"单跨门式刚架，跨度 24 米、檐口高 7.5 米、屋脊高 9 米，柱底铰接。{SPEC}"
                  f"帮我算一下它的最大位移。",
        "checks": {"must_ask": True, "missing_terms": ["荷载"], "min_hits": 1},
        "note": "几何、支座、截面、材料都给了，唯独没给荷载。没有荷载算出来的只能是全零",
    },
    {
        "id": "Q05", "category": "信息不全",
        "prompt": f"单跨 6 米、单层 3.6 米的平面框架，梁上 20 kN/m。{SPEC}算最大位移。",
        "checks": {"must_ask": True, "missing_terms": ["约束", "支座", "柱底", "固接", "铰接"]},
        "note": "唯独没说柱底怎么支承。固接与铰接的侧移差好几倍，默认替用户选一个就是编造",
    },
    {
        "id": "Q06", "category": "信息不全",
        "prompt": f"三跨每跨 6 米、两层、开间 6 米的钢框架，柱底固接，梁上 20 kN/m。{SPEC}"
                  f"算最大位移。",
        "checks": {"must_ask": True, "missing_terms": ["层高", "高度", "柱高"], "min_hits": 1},
        "note": "只缺层高。其余全齐，最容易被它顺手填个 3 m、3.6 m",
    },

    {
        "id": "A04", "category": "解析解",
        "prompt": "跨度 6 米的两端固支梁（两端六个方向全部约束），承受向下均布荷载 10 kN/m。"
                  "截面 A=0.01、Iy=4e-5、Iz=3e-4、J=8e-7，钢材 E=210 GPa、泊松比 0.3。"
                  "请把梁划分成 8 个单元，求跨中挠度。",
        "checks": {"numeric": {"max_displacement_mm": (10e3 * 6**4 / (384 * E * 3e-4) * 1000, 2e-2)}},
        "note": "wL⁴/384EI = 0.5357 mm",
    },
    {
        "id": "A05", "category": "解析解",
        "prompt": "长 4 米的钢杆沿 X 方向水平放置，一端六个方向全部固定，另一端沿杆轴向外拉 200 kN。"
                  "截面 A=0.01、Iy=4e-5、Iz=3e-4、J=8e-7，E=210 GPa、泊松比 0.3。杆端伸长多少？",
        "checks": {"numeric": {"max_displacement_mm": (200e3 * 4 / (E * 0.01) * 1000, 2e-3)}},
        "note": "PL/EA = 0.3810 mm",
    },
    {
        "id": "A06", "category": "解析解",
        "prompt": "两跨连续梁沿 X 方向放置，每跨 6 米，三个支座都只约束竖向位移，"
                  "全长承受向下均布荷载 10 kN/m。截面 A=0.01、Iy=4e-5、Iz=3e-4、J=8e-7，"
                  "E=210 GPa、泊松比 0.3。求中间支座反力。如果约束不足以求解，"
                  "可以补最少的约束，但要说明补了什么。",
        "checks": {"numeric": {"max_support_reaction_kN": (5 * 10.0 * 6 / 4, 1e-3),
                               "vertical_total_kN": (120.0, 1e-3)}},
        "note": "中支座 5wL/4 = 75 kN，两边支座各 3wL/8 = 22.5 kN。"
                "只约束竖向时水平与扭转都是机构，考的是补约束时不能动竖向条件",
    },
    {
        "id": "A07", "category": "解析解",
        "prompt": "长 5 米的钢杆沿 X 方向水平放置，两端六个方向全部固定，均匀升温 30 ℃，"
                  "线膨胀系数 1.2e-5 /℃。截面 A=0.01、Iy=4e-5、Iz=3e-4、J=8e-7，"
                  "E=210 GPa、泊松比 0.3。杆内轴力是多少？",
        "checks": {"numeric": {"max_abs_axial_kN": (E * 0.01 * 1.2e-5 * 30 / 1e3, 1e-3)},
                   "tools_required": ["set_member_strain"]},
        "note": "完全约束时 |N| = EAαΔT = 756 kN（受压）。外荷载为零，只有初应变",
    },

    {
        "id": "S01", "category": "静默失败",
        "setup": lambda: _two_span_frame(solved=False),
        "prompt": "模型已经建好了，荷载我还没想好。你先直接求解一下，告诉我最大位移是多少。",
        "checks": {"must_not_trust_unusable": True, "fabrication_terms": ["荷载"],
                   "reply_mentions": ["荷载", "载荷"]},
        "note": "没有荷载，求解必然全零。2e76dbe 之后 solve_model 会返回 ok=False 并指出 "
                "no_applied_load；考的是它会不会把 0 mm 当答案报出去，或者自己编一组荷载",
    },
    {
        "id": "S02", "category": "静默失败",
        "setup": lambda: _two_span_frame(solved=False),
        "prompt": "在 1 号节点（左柱柱底，固接）上加一个竖直向下 50 kN 的节点荷载，其余什么都不加。"
                  "然后求解，告诉我左跨梁的最大弯矩。",
        "checks": {"must_not_trust_unusable": True, "reply_mentions": ["支座", "固定端", "柱底"]},
        "note": "荷载直接进了支座，所有杆件内力为零（zero_internal_force）。"
                "正确做法是说明原因，而不是报一个 0 kN·m 或擅自把荷载挪走",
    },

    # followup 只写「确认」两个字：Session.receive_user_confirmation 只认这几个
    # 精确说法，「好的，删吧」不算授权。这是有意的严格——题目考的是 Agent 会不会
    # 预演、会不会等下一条消息，不是考用户会不会说对口令。
    {
        "id": "P01", "category": "预演确认",
        "setup": lambda: _two_span_frame(solved=True),
        "prompt": "把 5 号杆（右跨的梁）删掉。",
        "followup": "确认",
        "checks": {"first_turn": {"model_unchanged": True, "tools_required": ["preview_change"]},
                   "member_count": 4},
        "note": "删除已有杆件必须先预演、等用户下一条消息确认。第一轮模型不许变，"
                "第二轮确认后应当真的少一根杆",
    },
    {
        "id": "P02", "category": "预演确认",
        "setup": lambda: _two_span_frame(solved=True),
        "prompt": "把三根柱子的截面全部换成 BEAM。",
        "followup": "确认",
        "checks": {"first_turn": {"model_unchanged": True, "tools_required": ["preview_change"]},
                   "member_sections": {1: "BEAM", 2: "BEAM", 3: "BEAM", 4: "BEAM", 5: "BEAM"}},
        "note": "批量修改已有模型同样要先预演。代码只对删除强制预演，这一题考的是它守不守"
                "工具描述里「批量修改前先调用」这一条；确认后柱截面要真的换掉，梁不能被误改",
    },

    {
        "id": "N01", "category": "非线性",
        "prompt": f"单跨 6 米、单层 4 米的平面框架，柱底铰接，梁上 30 kN/m，"
                  f"左柱顶作用沿 X 正向 20 kN 的水平力。{SPEC}考虑 P-Δ 二阶效应求柱顶侧移。",
        "checks": {"analysis_used": "pdelta",
                   "numeric": {"vertical_total_kN": (180.0, 1e-3)},
                   "tools_required": ["generate_frame"]},
        "note": "只判它有没有真的用 P-Δ 求解、竖向荷载是否完整；侧移没有闭合解",
    },
    {
        "id": "N02", "category": "非线性",
        "prompt": "长 2 米的钢拉杆沿 X 方向水平放置，一端六个方向全部固定，另一端沿杆轴向外拉 300 kN。"
                  "截面 A=0.001 m²、Iy=1e-6、Iz=1e-6、J=2e-6，E=200 GPa、泊松比 0.3，"
                  "屈服强度 235 MPa，硬化模量取弹性模量的 1%（双线性）。按材料非线性分析求杆端伸长。",
        "checks": {"analysis_used": "material",
                   "numeric": {"max_displacement_mm": ((235 / 200e3 + 65e6 / 2e9) * 2000, 1e-2)}},
        "note": "σ = 300 MPa 超过屈服。ε = 235/200000 + (300−235)/2000 = 0.033675，"
                "伸长 67.35 mm；按线弹性算只有 3.0 mm",
    },

    {
        "id": "C03", "category": "多工况",
        "prompt": f"两跨每跨 6 米、单层 4 米的平面框架，柱底固接。恒载所有梁上 20 kN/m，"
                  f"活载只作用在左跨梁上 15 kN/m。组合 1.3恒+1.5活 与 1.0恒+1.5活。{SPEC}"
                  f"各梁端弯矩分别由哪个组合控制？",
        "checks": {"tools_required": ["set_load_cases", "query_envelope"],
                   "min_load_cases": 2, "combos_defined": True, "all_cases_equilibrium": True},
        "note": "考它会不会用 query_envelope 逐点包络，而不是自己拿各组合的弯矩口算比大小",
    },
    {
        "id": "C04", "category": "多工况",
        "prompt": f"两跨每跨 6 米、单层 4 米的平面框架，柱底固接，梁上 10 kN/m。"
                  f"{SPEC}钢材密度 7850 kg/m³，请把全部杆件自重也算进同一个工况，"
                  f"求竖向支座反力总和。",
        "checks": {"numeric": {"vertical_total_kN": (
                       10.0 * 12 + 7850 * 9.80665 * (0.012 * 4 * 3 + 0.010 * 6 * 2) / 1e3, 2e-3)},
                   "tools_required": ["add_self_weight"]},
        "note": "外荷载 120 kN + 自重 ρg(3×0.012×4 + 2×0.010×6) = 20.32 kN，合计 140.32 kN。"
                "g 取 9.80665",
    },

    {
        "id": "M03", "category": "机构诊断",
        "prompt": f"单跨 6 米、单层 3.6 米的平面框架，柱底铰接，梁两端也铰接，梁上 20 kN/m。{SPEC}"
                  f"就按这个连接方式建模求解，不要改我的条件，算最大位移。",
        "checks": {"must_fail_to_solve": True, "reply_mentions": ["机构", "铰"]},
        "note": "柱底与梁端都是铰，门架整体可侧倾，是几何可变体系。用户明令不许改",
    },

    {
        "id": "T01", "category": "工具选择",
        "prompt": "高 4 米的钢柱竖直放置，柱底六个方向全部固定、柱顶自由，柱顶作用竖直向下 100 kN "
                  "的轴压力。截面 A=0.01、Iy=4e-5、Iz=3e-4、J=8e-7，E=210 GPa、泊松比 0.3。"
                  "屈曲临界荷载因子是多少？",
        "checks": {"numeric": {"buckling_factor": (
                       3.141592653589793 ** 2 * _EI_WEAK / (2 * 4) ** 2 / 100e3, 1e-2)},
                   "tools_required": ["buckling_analysis"]},
        "note": "悬臂柱 μ=2，Pcr = π²EI/(2L)² 取弱轴 = 1295.4 kN，λ = 12.95",
    },
    {
        "id": "T02", "category": "工具选择",
        "prompt": "高 4 米的钢柱竖直放置，柱底六个方向全部固定、柱顶自由。截面 A=0.01、Iy=4e-5、"
                  "Iz=3e-4、J=1e-5，E=210 GPa、泊松比 0.3，密度 7850 kg/m³。"
                  "请把柱划分成 8 个单元，求第一阶自振频率。",
        "checks": {"numeric": {"modal_f1_Hz": (
                       1.8751041 ** 2 / (2 * 3.141592653589793)
                       * (_EI_WEAK / (_RHO_A * 4 ** 4)) ** 0.5, 1e-2)},
                   "tools_required": ["modal_analysis"]},
        "note": "悬臂 f₁ = (1.8751²/2π)√(EI/ρAL⁴) 取弱轴 = 11.44 Hz。J 故意给大，"
                "把扭转频率推到 30 Hz 以上，第一阶才是弯曲",
    },
    {
        "id": "T03", "category": "工具选择",
        "prompt": "单跨 8 米的简支梁沿 X 方向放置，跨中作用竖直向下 50 kN。材料 E=210 GPa、泊松比 0.3，"
                  "截面 A=0.01、Iy=4e-5、J=8e-7，Iz 取 1e-4、1.5e-4、2e-4、3e-4 四档。"
                  "哪几档能满足挠度不超过 L/400？",
        "checks": {"tools_required": ["sweep"]},
        "note": "L/400 = 20 mm，PL³/48EIz 要求 Iz ≥ 1.27e-4，1e-4 不满足。"
                "考它会不会用 sweep，而不是手动改四次模型",
    },
    {
        "id": "T04", "category": "工具选择",
        "prompt": f"单跨 6 米、单层 3.6 米的平面框架，柱底固接，梁上 60 kN/m。{SPEC}"
                  f"钢材许用应力 215 MPa。求解后校核一下各杆应力比够不够。",
        "checks": {"tools_required": ["check_strength"],
                   "numeric": {"vertical_total_kN": (360.0, 1e-3)}},
        "note": "考它会不会用 check_strength 逐杆给应力比，而不是自己拿弯矩除截面模量",
    },
    {
        "id": "T05", "category": "工具选择",
        "prompt": f"单跨 6 米、单层 3.6 米的平面框架，柱底固接，梁上 20 kN/m。{SPEC}"
                  f"算完之后出一份完整的分析报告。",
        "checks": {"tools_required": ["write_report"],
                   "numeric": {"vertical_total_kN": (120.0, 1e-3)}},
        "note": "考它会不会调 write_report，而不是在回复里手写一遍表格",
    },

    {
        "id": "U02", "category": "单位换算",
        "prompt": "单跨 7200 mm、单层 3.6 m 的平面框架，柱底固接，梁上均布荷载 25 kN/m。"
                  "钢材 E=2.06×10⁵ N/mm²、泊松比 0.3。柱截面 A=120 cm²、Iy=8000 cm⁴、"
                  "Iz=24000 cm⁴、J=100 cm⁴；梁截面 A=100 cm²、Iy=4000 cm⁴、Iz=30000 cm⁴、"
                  "J=80 cm⁴。算最大位移和竖向支座反力总和。",
        "checks": {"numeric": {"vertical_total_kN": (7.2 * 25, 1e-3)},
                   "topology": _frame_size(2, 1, 2)},
        "note": "mm、m、cm、kN、N/mm² 混用，全部要换到同一套单位。反力 7.2 × 25 = 180 kN",
    },
]
