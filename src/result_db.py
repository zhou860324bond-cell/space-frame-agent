"""统一结果数据库的最小实现。

层次固定为 Analysis(ResultDB) → Step → Frame → FieldOutput。线性静力中每个
荷载工况或组合是一项 Step，当前只有最终一帧。所有字段都携带位置、坐标系、
单位与平均规则，查询、绘图和报告可逐步迁移到同一数据合同。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from frame3d import CaseResult, Frame, Solution

RESULT_DB_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FieldOutput:
    name: str
    components: tuple[str, ...]
    location: str
    coordinate_system: str
    unit: str
    averaging: str
    values: dict[int, np.ndarray]

    def value(self, object_id: int) -> np.ndarray:
        return self.values[int(object_id)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "components": list(self.components),
            "location": self.location,
            "coordinate_system": self.coordinate_system,
            "unit": self.unit,
            "averaging": self.averaging,
            "values": {str(key): np.asarray(value, dtype=float).tolist()
                       for key, value in sorted(self.values.items())},
        }


@dataclass(frozen=True)
class ResultFrame:
    index: int
    description: str
    fields: dict[str, FieldOutput]

    def field(self, name: str) -> FieldOutput:
        return self.fields[name]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "description": self.description,
            "fields": {name: field.to_dict()
                       for name, field in sorted(self.fields.items())},
        }


@dataclass(frozen=True)
class ResultStep:
    name: str
    procedure: str
    frames: tuple[ResultFrame, ...]

    @property
    def final_frame(self) -> ResultFrame:
        return self.frames[-1]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "procedure": self.procedure,
                "frames": [frame.to_dict() for frame in self.frames]}


@dataclass(frozen=True)
class ResultDB:
    schema_version: int
    metadata: dict[str, Any]
    steps: dict[str, ResultStep]
    physical_to_elements: dict[int, tuple[int, ...]]

    def field(self, step: str, name: str) -> FieldOutput:
        return self.steps[step].final_frame.field(name)

    def physical_member_value(self, step: str, name: str,
                              physical_member_id: int) -> np.ndarray:
        """取物理构件两端值，不对内部分析节点做平均。"""
        element_ids = self.physical_to_elements[int(physical_member_id)]
        field = self.field(step, name)
        return np.array([field.value(element_ids[0])[0],
                         field.value(element_ids[-1])[1]], dtype=float)

    def solution_view(self, model: Frame):
        """从 Result DB 重建供既有后处理公式读取的只读式结果视图。"""
        results: dict[str, CaseResult] = {}
        for name in self.steps:
            displacement = np.zeros(model.num_dofs)
            reaction = np.zeros(model.num_dofs)
            for node_id in model.order():
                dofs = model.node_dofs(node_id)
                displacement[dofs[:3]] = self.field(name, "U").value(node_id)
                displacement[dofs[3:]] = self.field(name, "UR").value(node_id)
                reaction[dofs[:3]] = self.field(name, "RF").value(node_id)
                reaction[dofs[3:]] = self.field(name, "RM").value(node_id)
            member_forces: dict[int, np.ndarray] = {}
            for element_id in model.members:
                forces = np.zeros(12)
                for index, component in enumerate(("N", "Vy", "Vz", "T", "My", "Mz")):
                    ends = self.field(name, component).value(element_id)
                    forces[index], forces[index + 6] = ends
                member_forces[element_id] = forces
            results[name] = CaseResult(name, displacement, reaction, member_forces)
        primary = str(self.metadata["primary_step"])
        return _SolutionView(results, primary,
                             {"type": self.metadata["analysis_type"]})

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "metadata": dict(self.metadata),
            "mapping": {str(key): list(value)
                        for key, value in sorted(self.physical_to_elements.items())},
            "steps": {name: step.to_dict()
                      for name, step in sorted(self.steps.items())},
        }


def _model_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _nodal_values(model: Frame, vector: np.ndarray,
                  start: int) -> dict[int, np.ndarray]:
    return {node_id: np.asarray(vector[model.node_dofs(node_id)[start:start + 3]],
                                dtype=float).copy()
            for node_id in model.order()}


def _member_end_values(result, component: int) -> dict[int, np.ndarray]:
    return {element_id: np.array([forces[component], forces[component + 6]],
                                 dtype=float)
            for element_id, forces in result.member_forces.items()}


class _SolutionView:
    def __init__(self, results: dict[str, CaseResult], primary: str,
                 analysis: dict[str, Any]):
        self._results = results
        self.primary = primary
        self.analysis = analysis

    def __getitem__(self, name: str) -> CaseResult:
        return self._results[name]

    def all_results(self) -> dict[str, CaseResult]:
        return dict(self._results)


def from_solution(payload: dict[str, Any], model: Frame,
                  solution: Solution, mapping) -> ResultDB:
    """把求解器结果复制到独立、可序列化的 Result DB。"""
    length_unit = "m" if model.units == "N-m-Pa" else "mm"
    force_unit = "N"
    moment_unit = f"N·{length_unit}"
    procedure = str(solution.analysis.get("type", "linear_static"))
    steps: dict[str, ResultStep] = {}

    for name, result in solution.all_results().items():
        fields = {
            "U": FieldOutput("U", ("U1", "U2", "U3"), "NODE", "GLOBAL",
                             length_unit, "none", _nodal_values(model, result.U, 0)),
            "UR": FieldOutput("UR", ("UR1", "UR2", "UR3"), "NODE", "GLOBAL",
                              "rad", "none", _nodal_values(model, result.U, 3)),
            "RF": FieldOutput("RF", ("RF1", "RF2", "RF3"), "NODE", "GLOBAL",
                              force_unit, "none", _nodal_values(model, result.R, 0)),
            "RM": FieldOutput("RM", ("RM1", "RM2", "RM3"), "NODE", "GLOBAL",
                              moment_unit, "none", _nodal_values(model, result.R, 3)),
        }
        for component, index, unit in (
                ("N", 0, force_unit), ("Vy", 1, force_unit),
                ("Vz", 2, force_unit), ("T", 3, moment_unit),
                ("My", 4, moment_unit), ("Mz", 5, moment_unit)):
            fields[component] = FieldOutput(
                component, ("i", "j"), "ELEMENT_END", "LOCAL", unit,
                "none", _member_end_values(result, index))
        frame = ResultFrame(0, "final", fields)
        steps[name] = ResultStep(name, procedure, (frame,))

    return ResultDB(
        RESULT_DB_SCHEMA_VERSION,
        {
            "analysis_type": procedure,
            "primary_step": solution.primary,
            "model_hash": _model_hash(payload),
            "model_schema_version": payload.get("schema_version", 0),
            "model_units": model.units,
            "solver": "frame3d",
            "status": "COMPLETED",
            "physical_members": len(mapping.physical_to_elements),
            "analysis_elements": len(mapping.element_to_physical),
            "generated_analysis_nodes": len(mapping.generated_nodes),
        },
        steps,
        dict(mapping.physical_to_elements),
    )
