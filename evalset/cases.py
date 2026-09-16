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
]
