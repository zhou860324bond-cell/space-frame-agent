"""物理模型到求解器分析模型的最小编译边界。

Domain IR 中的 ``members`` 代表用户创建的物理构件；``frame3d.Member`` 是求解器
消费的分析单元。杆间集中荷载或显式内节点会在这里触发剖分，GUI、Agent 和
求解器都只消费同一份 ``CompilationMap``，不各自猜映射关系。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from frame3d import Frame, LoadCase, Member, Node, member_endpoints
from model_io import from_dict, migrate_payload, validate_payload
from span_loads import POINT, TRAPEZOID, UNIFORM, SpanLoad

_SPLIT_REL_TOL = 1e-9
_SPLIT_ABS_TOL = 1e-9


class CompilationError(ValueError):
    """Domain IR 无法编译，并保留可直接展示给用户的诊断。"""

    def __init__(self, diagnostics: list[str] | tuple[str, ...]):
        self.diagnostics = tuple(diagnostics)
        super().__init__("模型编译失败：" + "；".join(self.diagnostics[:3]))


@dataclass(frozen=True)
class CompilationMap:
    """物理构件 ID 与分析单元 ID 的权威双向索引。"""

    physical_to_elements: dict[int, tuple[int, ...]]
    element_to_physical: dict[int, int]
    generated_nodes: dict[int, tuple[int, float]]

    def element_ids(self, physical_member_id: int) -> tuple[int, ...]:
        return self.physical_to_elements[int(physical_member_id)]

    def physical_member_id(self, analysis_element_id: int) -> int:
        return self.element_to_physical[int(analysis_element_id)]


@dataclass(frozen=True)
class CompiledModel:
    """一次可追溯的编译产物。"""

    source_ir: dict[str, Any]
    analysis_model: Frame
    mapping: CompilationMap
    diagnostics: tuple[str, ...] = ()


def compile_model(payload: dict[str, Any]) -> CompiledModel:
    """校验并编译 Domain IR，集中力位置和显式内节点成为分析切分点。"""
    errors = validate_payload(payload)
    if errors:
        raise CompilationError(errors)

    source = migrate_payload(payload)
    physical = from_dict(source)
    frame = deepcopy(physical)
    frame.nodes = dict(physical.nodes)
    frame.members = {}
    frame.load_cases = {
        name: LoadCase(name, nodal_loads=dict(case.nodal_loads),
                       settlements=dict(case.settlements))
        for name, case in physical.load_cases.items()
    }

    next_node_id = max(physical.nodes, default=0) + 1
    next_element_id = max(physical.members, default=0) + 1
    physical_to_elements: dict[int, tuple[int, ...]] = {}
    element_to_physical: dict[int, int] = {}
    generated_nodes: dict[int, tuple[int, float]] = {}
    diagnostics: list[str] = []

    for physical_id in sorted(physical.members):
        member = physical.members[physical_id]
        pi, pj = member_endpoints(physical, member)
        length = float(np.linalg.norm(pj - pi))
        tolerance = max(_SPLIT_ABS_TOL, length * _SPLIT_REL_TOL)
        point_positions = [
            float(load.a)
            for case in physical.load_cases.values()
            for load in case.member_spans.get(physical_id, ())
            if load.kind == POINT
        ]
        direction = pj - pi
        explicit_points: list[tuple[float, int]] = []
        for node_id, node in physical.nodes.items():
            if node_id in (member.i, member.j):
                continue
            offset = node.xyz - pi
            position = float(np.dot(offset, direction) / length)
            if position <= tolerance or length - position <= tolerance:
                continue
            projection = pi + (position / length) * direction
            if float(np.linalg.norm(node.xyz - projection)) <= tolerance:
                explicit_points.append((position, node_id))

        split_points: list[tuple[float, int | None]] = []
        for position, node_id in sorted(explicit_points):
            if not split_points or position - split_points[-1][0] > tolerance:
                split_points.append((position, node_id))
        for position in sorted(point_positions):
            if position <= tolerance or length - position <= tolerance:
                continue
            if not any(abs(position - existing) <= tolerance
                       for existing, _ in split_points):
                split_points.append((position, None))
        split_points.sort()

        if point_positions and (any(member.offset_i) or any(member.offset_j)):
            raise CompilationError([
                f"杆件 {physical_id} 同时使用刚域偏移和杆间集中力；"
                "当前自动剖分尚不支持复杂偏移，请先显式拆分构件"
            ])

        internal_node_ids: list[int] = []
        interior: list[float] = []
        for position, existing_node_id in split_points:
            ratio = position / length
            interior.append(position)
            if existing_node_id is None:
                xyz = pi + ratio * direction
                node_id = next_node_id
                next_node_id += 1
                frame.nodes[node_id] = Node(
                    node_id, *(float(value) for value in xyz))
                generated_nodes[node_id] = (physical_id, ratio)
            else:
                node_id = existing_node_id
            internal_node_ids.append(node_id)

        node_ids = [member.i, *internal_node_ids, member.j]
        element_ids = [physical_id]
        for _ in range(len(node_ids) - 2):
            element_ids.append(next_element_id)
            next_element_id += 1
        physical_to_elements[physical_id] = tuple(element_ids)
        for element_id in element_ids:
            element_to_physical[element_id] = physical_id

        for index, element_id in enumerate(element_ids):
            frame.members[element_id] = Member(
                element_id, node_ids[index], node_ids[index + 1],
                member.section, member.material, member.ref_vector,
                member.releases_i if index == 0 else (),
                member.releases_j if index == len(element_ids) - 1 else (),
                member.offset_i if index == 0 else (0.0, 0.0, 0.0),
                member.offset_j if index == len(element_ids) - 1 else (0.0, 0.0, 0.0),
            )

        boundaries = [0.0, *interior, length]
        for case_name, physical_case in physical.load_cases.items():
            analysis_case = frame.load_cases[case_name]
            if physical_id in physical_case.member_loads:
                for element_id in element_ids:
                    analysis_case.member_loads[element_id] = physical_case.member_loads[physical_id]

            # 初应变沿杆是常量，各段**原样继承**，不按段长分配——
            # 这正是把装配误差和温度都归一成应变而不是 Δl 的好处：
            # 剖分在这里不需要知道任何长度。
            if physical_id in physical_case.member_strains:
                for element_id in element_ids:
                    analysis_case.member_strains[element_id] = (
                        physical_case.member_strains[physical_id])

            for load in physical_case.member_spans.get(physical_id, ()):
                if load.kind == POINT:
                    position = float(load.a)
                    if position <= tolerance:
                        node_id = member.i
                    elif length - position <= tolerance:
                        node_id = member.j
                    else:
                        nearest = min(range(len(interior)),
                                      key=lambda item: abs(interior[item] - position))
                        node_id = internal_node_ids[nearest]
                    previous = analysis_case.nodal_loads.get(node_id, (0.0,) * 6)
                    analysis_case.nodal_loads[node_id] = tuple(
                        float(previous[index]) + (float(load.w1[index]) if index < 3 else 0.0)
                        for index in range(6))
                    continue

                if load.kind == UNIFORM:
                    for element_id in element_ids:
                        analysis_case.member_spans.setdefault(element_id, []).append(load)
                    continue

                if load.kind == TRAPEZOID:
                    start = np.asarray(load.w1, dtype=float)
                    change = np.asarray(load.w2, dtype=float) - start
                    for index, element_id in enumerate(element_ids):
                        w1 = start + change * (boundaries[index] / length)
                        w2 = start + change * (boundaries[index + 1] / length)
                        analysis_case.member_spans.setdefault(element_id, []).append(
                            SpanLoad(TRAPEZOID, tuple(w1), tuple(w2)))

        if len(element_ids) > 1:
            reasons = []
            if explicit_points:
                reasons.append("显式内节点")
            if any(tolerance < position < length - tolerance
                   for position in point_positions):
                reasons.append("集中荷载")
            diagnostics.append(
                f"物理构件 {physical_id} 因{'和'.join(reasons)}编译为 "
                f"{len(element_ids)} 个分析单元")

    _reject_duplicate_load_paths(frame, element_to_physical)

    mapping = CompilationMap(
        physical_to_elements=physical_to_elements,
        element_to_physical=element_to_physical,
        generated_nodes=generated_nodes,
    )
    return CompiledModel(source, frame, mapping, tuple(diagnostics))


def _reject_duplicate_load_paths(frame: Frame,
                                 element_to_physical: dict[int, int]) -> None:
    """编译产物里同一对节点不得被连接两次。

    ``validate_payload`` 里已经有这条语义规则（"杆件 X 与杆件 Y 连接同一对节点"），
    但它在 :func:`compile_model` 开头、**对编译前的物理 IR** 执行。剖分会造出
    校验本该拦下的东西：用户画了 1-2、2-3，又画了一根跨越两跨的 1-3，三者两两
    不同因而合法；1-3 被节点 2 剖分后，1-2 与 2-3 之间各自出现两条平行传力路径。

    这种模型条件数良好、静力平衡也满足，静默失败检测一项都不会触发，
    但刚度被成倍放大——两跨梁的实测结果是位移正好减半。所以在这里直接拒绝，
    而不是给一个算得出、看着正常、却不是用户画的那个结构的答案。
    """
    seen: dict[tuple[int, int], int] = {}
    conflicts: list[str] = []
    for element_id in sorted(frame.members):
        element = frame.members[element_id]
        key = (min(element.i, element.j), max(element.i, element.j))
        if key in seen:
            first = element_to_physical.get(seen[key], seen[key])
            second = element_to_physical.get(element_id, element_id)
            if first == second:                     # 同一构件的链，不该发生
                continue
            conflicts.append(
                f"物理构件 {second} 与物理构件 {first} 在节点 {key[0]}-{key[1]} 之间"
                "重复传力")
        else:
            seen[key] = element_id
    if conflicts:
        raise CompilationError([
            *conflicts,
            "同一对节点之间出现了多于一条传力路径，结构刚度会被成倍放大。"
            "通常是一根构件跨越了已经存在的构件；请删除跨越的那根，"
            "或把它改成不与已有构件重叠",
        ])
