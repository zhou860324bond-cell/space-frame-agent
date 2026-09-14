"""对标算例的模型定义。两侧（本程序与 Abaqus）从同一份定义出发，保证模型完全一致。

单位制：N-m-Pa，与本程序一致。Abaqus 本身没有单位制，量纲一致性由我们保证。

单元选择很关键：
* **B33** 是 Abaqus 的三次梁单元，不计横向剪切变形，与本程序的 Euler-Bernoulli
  格式属于同一套理论——这才是苹果对苹果，误差应当落到数值精度量级。
* **B31** 是一点缩减积分的 Timoshenko 梁，含剪切变形与细长度补偿。拿它对标会
  留下系统性偏差，细长构件下很小、粗短构件下明显。两个都跑，正好说明差异来源。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

STEEL = {"name": "STEEL", "E": 2.1e11, "nu": 0.3}

# 对称截面：Iy = Iz，截面主轴方向搞反也看不出来（对照组）
SQUARE = {"name": "SQUARE", "A": 0.01, "Iy": 8.3333e-6, "Iz": 8.3333e-6, "J": 1.4e-5}
# 强弱轴悬殊的截面：主轴方向若搞反，误差会立刻放大到百分级（判别组）
STRONG = {"name": "STRONG", "A": 0.012, "Iy": 2.0e-5, "Iz": 4.0e-4, "J": 1.0e-6}
SLENDER_BEAM = {"name": "SLENDER", "A": 0.010, "Iy": 4.0e-5, "Iz": 3.0e-4, "J": 8.0e-7}


def _nodes(coords: list[tuple[float, float, float]]) -> list[dict]:
    return [{"id": k + 1, "x": float(x), "y": float(y), "z": float(z)}
            for k, (x, y, z) in enumerate(coords)]


def _members(pairs, section: str, material: str = "STEEL") -> list[dict]:
    return [{"id": k + 1, "i": i, "j": j, "section": section, "material": material}
            for k, (i, j) in enumerate(pairs)]


def cantilever_strong_axis() -> dict[str, Any]:
    """悬臂梁，端部竖向力。强弱轴悬殊的截面——主轴方向搞反会立刻暴露。"""
    n = 8
    L = 6.0
    coords = [(L * k / n, 0.0, 0.0) for k in range(n + 1)]
    return {
        "name": "cantilever_strong_axis",
        "units": "N-m-Pa",
        "materials": [STEEL], "sections": [STRONG],
        "nodes": _nodes(coords),
        "members": _members([(k + 1, k + 2) for k in range(n)], "STRONG"),
        "supports": [{"node": 1, "fix": [1, 1, 1, 1, 1, 1]}],
        "nodal_loads": [{"node": n + 1, "load": [0.0, 0.0, -50e3, 0.0, 0.0, 0.0]}],
        "note": "端部竖向力，理论挠度 PL^3/(3 E Iz)",
    }


def cantilever_weak_axis() -> dict[str, Any]:
    """同一根悬臂梁，改为侧向力——弯曲绕另一根主轴，与上一例互为交叉验证。"""
    model = cantilever_strong_axis()
    model["name"] = "cantilever_weak_axis"
    model["nodal_loads"] = [{"node": len(model["nodes"]),
                             "load": [0.0, -50e3, 0.0, 0.0, 0.0, 0.0]}]
    model["note"] = "侧向力，理论挠度 PL^3/(3 E Iy)"
    return model


def torsion_bar() -> dict[str, Any]:
    """扭转杆：只考 GJ，与弯曲完全解耦。"""
    L = 4.0
    return {
        "name": "torsion_bar",
        "units": "N-m-Pa",
        "materials": [STEEL], "sections": [SQUARE],
        "nodes": _nodes([(0.0, 0.0, 0.0), (L, 0.0, 0.0)]),
        "members": _members([(1, 2)], "SQUARE"),
        "supports": [{"node": 1, "fix": [1, 1, 1, 1, 1, 1]}],
        "nodal_loads": [{"node": 2, "load": [0.0, 0.0, 0.0, 20e3, 0.0, 0.0]}],
        "note": "端部扭矩，理论扭转角 T L /(G J)",
    }


def portal_frame() -> dict[str, Any]:
    """门式刚架：柱受弯、梁受均布荷载，含轴力与弯矩耦合。"""
    H, S = 4.0, 8.0
    nb = 4
    coords = [(0.0, 0.0, 0.0), (0.0, 0.0, H)]
    coords += [(S * k / nb, 0.0, H) for k in range(1, nb + 1)]
    coords += [(S, 0.0, 0.0)]
    pairs = [(1, 2)] + [(2 + k, 3 + k) for k in range(nb)] + [(len(coords), 2 + nb)]
    return {
        "name": "portal_frame",
        "units": "N-m-Pa",
        "materials": [STEEL], "sections": [SLENDER_BEAM],
        "nodes": _nodes(coords),
        "members": _members(pairs, "SLENDER"),
        "supports": [{"node": 1, "fix": [1, 1, 1, 1, 1, 1]},
                     {"node": len(coords), "fix": [1, 1, 1, 1, 1, 1]}],
        "member_loads": [{"member": k + 2, "w": [0.0, 0.0, -20e3]} for k in range(nb)],
        "nodal_loads": [{"node": 2, "load": [15e3, 0.0, 0.0, 0.0, 0.0, 0.0]}],
        "note": "梁上均布荷载 + 柱顶水平力，考等效节点荷载与内力回算",
    }


def space_frame() -> dict[str, Any]:
    """空间刚架：两跨一开间两层，考竖直杆件的局部坐标系与三维耦合。"""
    from generator import generate_frame
    model = generate_frame(spans=[6.0, 6.0], storeys=[3.6, 3.6], bays=[5.0],
                           column_section="STRONG", beam_section="SLENDER",
                           material="STEEL", beam_load=18e3)
    model["name"] = "space_frame"
    model["materials"] = [STEEL]
    model["sections"] = [STRONG, SLENDER_BEAM]
    model["note"] = "多层多跨空间刚架，柱为竖直杆件（参考向量退化分支）"
    return model


BENCHMARKS = [cantilever_strong_axis, cantilever_weak_axis,
              torsion_bar, portal_frame, space_frame]


def all_models() -> list[dict[str, Any]]:
    return [factory() for factory in BENCHMARKS]


def with_uniform_shear_areas(model: dict[str, Any], factor: float = 5.0 / 6.0
                             ) -> dict[str, Any]:
    """复制模型，并用 ``Ay=Az=factor*A`` 启用同参数 Timoshenko 对标。

    这是对标参数化工具，不声称 ``5A/6`` 对所有真实截面都是精确剪切面积。
    原始 B33 基线保持不变。
    """
    if not 0.0 < float(factor) <= 1.0:
        raise ValueError("剪切面积系数必须在 (0, 1] 内")
    out = deepcopy(model)
    for section in out.get("sections", []):
        section["Ay"] = float(factor) * float(section["A"])
        section["Az"] = float(factor) * float(section["A"])
    return out


def refine_uniform_members(model: dict[str, Any], divisions: int) -> dict[str, Any]:
    """把对标模型的每根杆等分，保持原节点、荷载强度与属性不变。

    这是 B31 网格收敛诊断用的离散变换，不是通用模型编译器。对标模型没有
    杆端释放、刚域偏移或跨间集中荷载；遇到这些需要端点语义的字段就明确拒绝，
    避免生成一个看似可算、实际已经改变物理含义的模型。
    """
    divisions = int(divisions)
    if divisions < 1:
        raise ValueError("divisions 必须至少为 1")
    out = deepcopy(model)
    if divisions == 1:
        return out
    if out.get("member_spans") or out.get("load_cases"):
        raise ValueError("细化诊断目前只支持顶层均布杆件荷载")
    forbidden = {"releases", "offset_i", "offset_j"}
    if any(forbidden & set(member) for member in out.get("members", [])):
        raise ValueError("带杆端释放或刚域偏移的模型不能用此诊断细化")

    nodes = {int(node["id"]): node for node in out["nodes"]}
    new_nodes = list(out["nodes"])
    next_node = max(nodes) + 1
    next_member = 1
    new_members = []
    mapping: dict[int, list[int]] = {}

    for member in out["members"]:
        start, end = nodes[int(member["i"])], nodes[int(member["j"])]
        chain = [int(member["i"])]
        for index in range(1, divisions):
            ratio = index / float(divisions)
            new_nodes.append({
                "id": next_node,
                "x": start["x"] + ratio * (end["x"] - start["x"]),
                "y": start["y"] + ratio * (end["y"] - start["y"]),
                "z": start["z"] + ratio * (end["z"] - start["z"]),
            })
            chain.append(next_node)
            next_node += 1
        chain.append(int(member["j"]))
        properties = {key: deepcopy(value) for key, value in member.items()
                      if key not in {"id", "i", "j"}}
        mapping[int(member["id"])] = []
        for i, j in zip(chain, chain[1:]):
            new_members.append({"id": next_member, "i": i, "j": j,
                                **deepcopy(properties)})
            mapping[int(member["id"])].append(next_member)
            next_member += 1

    new_loads = []
    for load in out.get("member_loads", []):
        for member_id in mapping[int(load["member"])]:
            item = deepcopy(load)
            item["member"] = member_id
            new_loads.append(item)
    out["nodes"] = new_nodes
    out["members"] = new_members
    if "member_loads" in out:
        out["member_loads"] = new_loads
    return out
