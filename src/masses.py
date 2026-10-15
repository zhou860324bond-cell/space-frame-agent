"""集中质量：节点附加质量，以及由荷载换算的重力荷载代表值。

杆件自身的质量来自材料密度，由 ``modal.consistent_mass`` 按一致质量阵装配；
真实建筑的大部分质量却不在杆件上——楼板、面层、设备、隔墙都只以荷载的形式
出现在模型里。不把它们算进质量，模态和反应谱算的只是「一副空骨架」的振动，
周期偏短、地震作用整体失真，而且不报错。

两条入口：

* 直接给节点质量（平动质量 + 可选的三个转动惯量），用于设备、配重；
* :func:`masses_from_loads`：按 GB 50011 的重力荷载代表值，把若干工况的
  **竖向向下**荷载乘组合值系数、除以 g 换成节点质量。杆上荷载按简支梁的
  杠杆关系分到两端节点。

单位：质量与密度同一套——N-m-Pa 下是 kg，N-mm-MPa 下是 t（吨）。
转动惯量是 质量·长度²。g 从 ``units`` 取，不写死。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Any

import numpy as np

from span_loads import SpanLoad, resultant
from units import system as unit_system

#: 自重荷载的标记，与 session_base.SELF_WEIGHT_NOTE 相同。在这里重复一份字面量，
#: 避免内核模块依赖会话层；由测试核对两者一致。
SELF_WEIGHT_NOTE = "self-weight"


def nodal_mass_table(payload: dict[str, Any]) -> dict[int, np.ndarray]:
    """把模型里的 ``masses`` 按节点求和：节点 → [m, Ixx, Iyy, Izz]。

    同一节点允许有多条（手工附加 + 荷载换算），物理上直接相加。
    """
    table: dict[int, np.ndarray] = {}
    for entry in payload.get("masses") or []:
        node = int(entry["node"])
        inertia = entry.get("inertia") or (0.0, 0.0, 0.0)
        row = np.array([float(entry["mass"]), *(float(v) for v in inertia)])
        table[node] = table.get(node, np.zeros(4)) + row
    return table


def _blocks(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """工况名 → 存放该工况荷载的字典。顶层单工况写法的名字是 default。"""
    out = {"default": payload}
    for case in payload.get("load_cases") or []:
        out[str(case["name"])] = case
    return out


def _member_length(payload: dict[str, Any], member: dict[str, Any]) -> float:
    """可变形段长度：节点间距再计入刚域偏移，与杆间荷载的位置 a、b 同一基准。"""
    nodes = {int(n["id"]): np.array([n["x"], n["y"], n["z"]], dtype=float)
             for n in payload["nodes"]}
    pi = nodes[int(member["i"])] + np.asarray(member.get("offset_i") or (0, 0, 0), float)
    pj = nodes[int(member["j"])] + np.asarray(member.get("offset_j") or (0, 0, 0), float)
    return float(np.linalg.norm(pj - pi))


def masses_from_loads(payload: dict[str, Any], factors: dict[str, float],
                      ) -> tuple[dict[int, float], dict[str, Any]]:
    """重力荷载代表值换成节点质量。返回 (节点 → 质量, 换算说明)。

    规则：

    * 只取全局 **−z**（向下）分量。向上的竖向分量与水平分量不是重力，忽略并计数；
    * 杆上荷载先求竖向合力与作用点（``span_loads.resultant``，六种杆间荷载
      一套算法），再按简支梁杠杆关系分到两端节点；
    * 标记为自重、且该杆材料有密度的荷载跳过——一致质量阵里已经有 ρA，
      再换一次就算了两遍。材料没有密度时自重荷载照常换算。
    """
    blocks = _blocks(payload)
    unknown = sorted(set(factors) - set(blocks))
    if unknown:
        raise ValueError(f"没有这些工况：{unknown}。已有：{sorted(blocks)}")
    g = unit_system(payload.get("units")).gravity
    members = {int(m["id"]): m for m in payload.get("members") or []}
    density = {m["name"]: float(m.get("density") or 0.0)
               for m in payload.get("materials") or []}

    weight: dict[int, float] = defaultdict(float)     # 节点 → 向下的力
    skipped_self_weight = ignored_upward = ignored_horizontal = 0

    def down(fz: float, factor: float) -> float:
        nonlocal ignored_upward
        if fz > 0.0:
            ignored_upward += 1
            return 0.0
        return -fz * factor

    def spread(member_id: int, load: SpanLoad, factor: float) -> None:
        nonlocal ignored_horizontal
        if load.kind == "moment":
            return                              # 力偶没有竖向合力，不是重量
        member = members[member_id]
        length = _member_length(payload, member)
        w1 = np.asarray(load.w1, dtype=float)
        if np.any(np.abs(w1[:2]) > 0.0) or np.any(np.abs(np.asarray(load.w2)[:2]) > 0.0):
            ignored_horizontal += 1
        vertical = replace(load, w1=(0.0, 0.0, float(load.w1[2])),
                           w2=(0.0, 0.0, float(load.w2[2])))
        total, xbar = resultant(vertical, length)
        force = down(float(total[2]), factor)
        if force == 0.0 or length <= 0.0:
            return
        share_j = min(max(xbar / length, 0.0), 1.0)
        weight[int(member["i"])] += force * (1.0 - share_j)
        weight[int(member["j"])] += force * share_j

    for name, factor in factors.items():
        block = blocks[name]
        factor = float(factor)
        for entry in block.get("nodal_loads") or []:
            load = entry["load"]
            if abs(load[0]) > 0.0 or abs(load[1]) > 0.0:
                ignored_horizontal += 1
            weight[int(entry["node"])] += down(float(load[2]), factor)
        for entry in block.get("member_loads") or []:
            spread(int(entry["member"]),
                   SpanLoad("uniform", tuple(float(v) for v in entry["w"])), factor)
        for entry in block.get("member_spans") or []:
            member = members[int(entry["member"])]
            if (entry.get("note") == SELF_WEIGHT_NOTE
                    and density.get(member["material"], 0.0) > 0.0):
                skipped_self_weight += 1
                continue
            spread(int(entry["member"]), SpanLoad.from_dict(entry), factor)

    masses = {node: w / g for node, w in sorted(weight.items()) if w > 0.0}
    report = {"factors": {k: float(v) for k, v in factors.items()},
              "gravity": g,
              "skipped_self_weight": skipped_self_weight,
              "ignored_upward": ignored_upward,
              "ignored_horizontal_components": ignored_horizontal}
    return masses, report
