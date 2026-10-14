"""Session 的建模对象域：截面指派、边界条件、荷载、约束的创建与管理。

对应 Abaqus/CAE 的「Create … / … Manager」：每类对象都是同一组操作——创建、
编辑、复制、改名、抑制、删除。界面的管理器、模型树右键菜单和 Agent 都调
这里，不各自改逐条目字段。对象的语义与派生规则见 ``features.py``。

区域写法与 Abaqus 的「选区域或用集合」一致：集合名（活引用，集合变了对象
跟着变）、编号列表，或 ``{"set": ...}`` / ``{"nodes": [...]}`` /
``{"members": [...]}``。

每次修改都在模型副本上试做、再生成、校验，**只拦这次修改引入的新错误**：
几何刚建好、属性还没指派时，原本就有的「未指派」不该挡住创建边界条件。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

import features as _features
from session_base import ToolResult, _records

_LABEL = {"section_assignments": "截面指派", "beam_orientations": "梁方向",
          "bcs": "边界条件", "loads": "荷载", "constraints": "约束"}
#: 工具参数里的类别名 → features 里的键
KIND_ALIASES = {"section_assignment": "section_assignments",
                "section_assignments": "section_assignments",
                "beam_orientation": "beam_orientations",
                "beam_orientations": "beam_orientations",
                "bc": "bcs", "bcs": "bcs", "boundary_condition": "bcs",
                "load": "loads", "loads": "loads",
                "constraint": "constraints", "constraints": "constraints"}

#: Abaqus 边界条件预设（Encastre / Pinned / 对称）。顺序 ux uy uz rx ry rz。
BC_PRESETS = {
    "ENCASTRE": [1, 1, 1, 1, 1, 1],
    "PINNED": [1, 1, 1, 0, 0, 0],
    "XSYMM": [1, 0, 0, 0, 1, 1],
    "YSYMM": [0, 1, 0, 1, 0, 1],
    "ZSYMM": [0, 0, 1, 1, 1, 0],
}


def _region(value: Any, what: str) -> dict[str, Any]:
    """把工具参数里的区域写法规范成 features 的区域字典。"""
    if isinstance(value, dict):
        return {k: v for k, v in value.items() if k in ("set", "nodes", "members")}
    if isinstance(value, str):
        return {"set": value}
    if isinstance(value, int):
        return {what: [int(value)]}
    return {what: sorted({int(v) for v in value or []})}


class FeaturesMixin:
    """建模对象的创建与管理。见模块 docstring。"""

    # --- 内部 ---

    def _feature_kind(self, kind: str) -> str:
        key = KIND_ALIASES.get(str(kind))
        if key is None:
            raise ValueError(f"对象类别只能是 {sorted(set(KIND_ALIASES))}，收到 {kind!r}")
        return key

    def _feature_list(self, model: dict, kind: str) -> list[dict[str, Any]]:
        return model.setdefault("features", {}).setdefault(kind, [])

    def _find_feature(self, model: dict, kind: str, name: str) -> dict[str, Any] | None:
        return next((f for f in (model.get("features") or {}).get(kind) or []
                     if f["name"] == name), None)

    def _commit_features(self, candidate: dict, summary: dict[str, Any]) -> ToolResult:
        """副本上再生成并校验；只拒绝这次修改新引入的错误。"""
        from model_io import validate_editing_payload
        # 空的类别不留在模型里，保持文件干净
        feats = candidate.get("features") or {}
        for key in list(feats):
            if not feats[key]:
                del feats[key]
        if not feats:
            candidate.pop("features", None)
        _features.regenerate(candidate)
        before = set(validate_editing_payload(self.model))
        errors = [e for e in validate_editing_payload(candidate) if e not in before]
        if errors:
            return ToolResult(False, {"errors": errors,
                                      "hint": "改动未写入，原模型保持不变。"})
        self.model = candidate
        self._invalidate()
        return ToolResult(True, summary)

    def _region_summary(self, model: dict, kind: str, feature: dict) -> dict[str, Any]:
        what = _features.region_kind(kind, feature)
        ids, problem = _features.region_ids(model, feature["region"], what)
        return {"region": feature["region"], what: ids, **({"problem": problem} if problem else {})}

    # --- 创建 ---

    @_records
    def create_section_assignment(self, name: str, region, section: str,
                                  material: str) -> ToolResult:
        """截面指派（Property 模块 Assign Section）：把截面与材料指派到一组杆件。"""
        feature = {"name": str(name).strip(), "region": _region(region, "members"),
                   "section": str(section), "material": str(material)}
        return self._create_feature("section_assignments", feature)

    @_records
    def create_beam_orientation(self, name: str, region, n1: list[float]) -> ToolResult:
        """梁方向（Property 模块 Assign Beam Orientation）：给一组杆件指定截面局部 y 轴的近似方向。

        ``n1`` 是全局分量，只看方向；它与杆轴平行时截面朝向无法确定，会被拒绝。
        """
        vector = [float(v) for v in n1]
        if len(vector) != 3 or not all(np.isfinite(vector)) or not any(vector):
            return ToolResult(False, {"error": "n1 必须是三个有限数，且不全为零"})
        feature = {"name": str(name).strip(), "region": _region(region, "members"),
                   "n1": vector}
        return self._create_feature("beam_orientations", feature)

    @_records
    def create_boundary_condition(self, name: str, region, fix: list[int] | None = None,
                                  preset: str | None = None,
                                  spring: list[float] | None = None,
                                  step: str = _features.INITIAL) -> ToolResult:
        """边界条件（Load 模块 Create Boundary Condition）。

        ``preset`` 是 Abaqus 的 ENCASTRE / PINNED / XSYMM / YSYMM / ZSYMM；
        也可直接给 ``fix``（六个 0/1）。``spring`` 是弹性支承刚度，同一方向
        不能既固定又给弹簧。``step`` 缺省为 Initial。
        """
        if preset is not None:
            key = str(preset).upper()
            if key not in BC_PRESETS:
                return ToolResult(False, {"error": f"preset 只能是 {sorted(BC_PRESETS)}"})
            if fix is not None and list(fix) != BC_PRESETS[key]:
                return ToolResult(False, {"error": f"preset={key} 与 fix={list(fix)} 不一致，"
                                                   "只给其中一个"})
            fix = BC_PRESETS[key]
        if fix is None:
            return ToolResult(False, {"error": "要给 fix（六个 0/1）或 preset",
                                      "presets": sorted(BC_PRESETS)})
        if len(fix) != 6 or any(int(v) not in (0, 1) for v in fix):
            return ToolResult(False, {"error": "fix 必须是六个 0 或 1，顺序 ux uy uz rx ry rz"})
        feature: dict[str, Any] = {"name": str(name).strip(),
                                   "region": _region(region, "nodes"),
                                   "fix": [int(v) for v in fix]}
        if step and step != _features.INITIAL:
            feature["step"] = str(step)
        if spring is not None:
            values = [float(v) for v in spring]
            if len(values) != 6 or not all(np.isfinite(v) and v >= 0 for v in values):
                return ToolResult(False, {"error": "spring 必须是六个非负有限数"})
            if any(values):
                feature["spring"] = values
        return self._create_feature("bcs", feature)

    @_records
    def create_load(self, name: str, case: str, load_type: str, region,
                    force: list[float] | None = None, moment: list[float] | None = None,
                    w: list[float] | None = None, g: list[float] | None = None) -> ToolResult:
        """荷载（Load 模块 Create Load）：集中力、力矩、线荷载或重力，施加在区域上。

        集中力 ``force`` 与力矩 ``moment`` 作用在节点区域；线荷载 ``w``（全局
        分量，力/长度）与重力 ``g``（加速度，全局分量）作用在杆件区域。
        数值单位跟模型的单位制。
        """
        load_type = str(load_type)
        if load_type not in _features.LOAD_TYPES:
            return ToolResult(False, {"error": f"load_type 只能是 {list(_features.LOAD_TYPES)}"})
        field = {"concentrated_force": "force", "moment": "moment", "line_load": "w",
                 "gravity": "g"}[load_type]
        value = {"force": force, "moment": moment, "w": w, "g": g}[field]
        if value is None:
            return ToolResult(False, {"error": f"{load_type} 要给 {field}（三个全局分量）"})
        vector = [float(v) for v in value]
        if len(vector) != 3 or not all(np.isfinite(vector)):
            return ToolResult(False, {"error": f"{field} 必须是三个有限数"})
        feature = {"name": str(name).strip(), "case": str(case), "type": load_type,
                   field: vector}
        feature["region"] = _region(region, _features.region_kind("loads", feature))
        return self._create_feature("loads", feature)

    @_records
    def create_constraint(self, name: str, constraint_type: str, control: int, region,
                          dofs: list[int] | None = None) -> ToolResult:
        """约束（Interaction 模块 Create Constraint）：从节点区域跟随主节点 ``control``。

        ``constraint_type``：beam（MPC BEAM 刚性连杆）、tie（重合节点全部自由度相等）、
        pin（重合节点只连平动，即铰）、coupling（运动耦合，``dofs`` 六个 0/1 选被
        约束的自由度；只选 ux uy rz 就是水平刚性楼板）。
        """
        constraint_type = str(constraint_type)
        if constraint_type not in _features.CONSTRAINT_TYPES:
            return ToolResult(False, {"error": f"constraint_type 只能是 {list(_features.CONSTRAINT_TYPES)}"})
        feature: dict[str, Any] = {"name": str(name).strip(), "type": constraint_type,
                                   "control": int(control), "region": _region(region, "nodes")}
        if constraint_type == "coupling":
            mask = [1] * 6 if dofs is None else [int(v) for v in dofs]
            if len(mask) != 6 or any(v not in (0, 1) for v in mask) or not any(mask):
                return ToolResult(False, {"error": "dofs 必须是六个 0 或 1（ux uy uz rx ry rz），至少一个 1"})
            feature["dofs"] = mask
        elif dofs is not None:
            return ToolResult(False, {"error": f"{constraint_type} 的自由度是固定的，只有 coupling 可以给 dofs"})
        result = self._create_feature("constraints", feature)
        if result.ok:
            loose = self._undriven_control_dofs(int(control))
            if loose:
                result.payload["warning"] = (
                    f"主节点 {control} 没有连杆件，它的 {loose} 方向没有被任何东西带动，"
                    "求解时刚度为零；请给主节点在这些方向加边界条件（与 Abaqus 参考点相同）")
        return result

    def _undriven_control_dofs(self, control: int) -> list[str]:
        """主节点上刚度为零的自由度（悬空参考点常见）。只读诊断，失败时返回空。"""
        if any(int(control) in (int(m["i"]), int(m["j"])) for m in self.model.get("members") or []):
            return []
        try:
            from frame3d import LOCAL_DOF_NAMES, assemble, dof_basis
            from model_io import from_dict
            frame = from_dict(self.model)
            if any(not frame.sections.get(m.section) or not frame.materials.get(m.material)
                   for m in frame.members.values()):
                return []
            basis = dof_basis(frame)
            diagonal = np.abs(basis.reduce(assemble(frame)[0]).diagonal())
        except Exception:                      # noqa: BLE001  诊断失败不影响创建结果
            return []
        column = {int(d): k for k, d in enumerate(basis.free)}
        dofs = frame.node_dofs(int(control))
        scale = float(diagonal.max(initial=0.0)) or 1.0
        return [LOCAL_DOF_NAMES[k] for k in range(6)
                if dofs[k] in column and diagonal[column[dofs[k]]] <= 1e-12 * scale]

    def constraint_links(self) -> list[tuple[str, list[float], list[list[float]]]]:
        """视口画约束用：(名称, 主节点坐标, [从节点坐标])，只含生效的约束。"""
        xyz = {int(n["id"]): [float(n["x"]), float(n["y"]), float(n["z"])]
               for n in self.model.get("nodes") or []}
        return [(c["name"], xyz[c["master"]], [xyz[s] for s in c["slaves"] if s in xyz])
                for c in _features.resolved_constraints(self.model) if c["master"] in xyz]

    def _create_feature(self, kind: str, feature: dict[str, Any]) -> ToolResult:
        if not feature["name"]:
            return ToolResult(False, {"error": f"{_LABEL[kind]}名称不能为空"})
        if self._find_feature(self.model, kind, feature["name"]) is not None:
            return ToolResult(False, {"error": f"已有名为「{feature['name']}」的{_LABEL[kind]}，"
                                               "请换一个名字或编辑原对象"})
        candidate = deepcopy(self.model)
        self._feature_list(candidate, kind).append(feature)
        return self._commit_features(candidate, {
            "kind": kind, "name": feature["name"],
            **self._region_summary(candidate, kind, feature)})

    # --- 管理器操作 ---

    @_records
    def edit_feature(self, kind: str, name: str, changes: dict[str, Any]) -> ToolResult:
        """编辑对象的参数或区域。``changes`` 只写要改的键（region 用区域写法），
        值为 None 表示去掉该键。"""
        return self._edit_feature(kind, name, changes)

    def _edit_feature(self, kind: str, name: str, changes: dict[str, Any]) -> ToolResult:
        try:
            kind = self._feature_kind(kind)
        except ValueError as exc:
            return ToolResult(False, {"error": str(exc)})
        candidate = deepcopy(self.model)
        feature = self._find_feature(candidate, kind, name)
        if feature is None:
            return ToolResult(False, {"error": f"没有名为「{name}」的{_LABEL[kind]}"})
        if "name" in changes:
            return ToolResult(False, {"error": "改名请用 rename_feature"})
        for key, value in dict(changes).items():
            if key == "region":
                feature["region"] = _region(value, _features.region_kind(kind, feature))
            elif value is None:
                feature.pop(key, None)
            else:
                feature[key] = value
        return self._commit_features(candidate, {
            "kind": kind, "name": name, "changed": sorted(changes),
            **self._region_summary(candidate, kind, feature)})

    @_records
    def rename_feature(self, kind: str, name: str, new_name: str) -> ToolResult:
        try:
            kind = self._feature_kind(kind)
        except ValueError as exc:
            return ToolResult(False, {"error": str(exc)})
        new_name = str(new_name).strip()
        if not new_name:
            return ToolResult(False, {"error": "新名称不能为空"})
        candidate = deepcopy(self.model)
        feature = self._find_feature(candidate, kind, name)
        if feature is None:
            return ToolResult(False, {"error": f"没有名为「{name}」的{_LABEL[kind]}"})
        if self._find_feature(candidate, kind, new_name) is not None:
            return ToolResult(False, {"error": f"已有名为「{new_name}」的{_LABEL[kind]}"})
        feature["name"] = new_name
        return self._commit_features(candidate, {"kind": kind, "name": new_name, "was": name})

    @_records
    def copy_feature(self, kind: str, name: str, new_name: str) -> ToolResult:
        """复制对象（Abaqus 管理器的 Copy）。副本与原对象同参数、同区域。"""
        try:
            kind = self._feature_kind(kind)
        except ValueError as exc:
            return ToolResult(False, {"error": str(exc)})
        source = self._find_feature(self.model, kind, name)
        if source is None:
            return ToolResult(False, {"error": f"没有名为「{name}」的{_LABEL[kind]}"})
        if kind == "bcs" and not source.get("suppressed"):
            # 同区域、同自由度的两个边界条件必然过约束；副本先抑制，改好区域再启用
            copy = {**deepcopy(source), "name": str(new_name).strip(), "suppressed": True}
        else:
            copy = {**deepcopy(source), "name": str(new_name).strip()}
        return self._create_feature(kind, copy)

    @_records
    def suppress_feature(self, kind: str, name: str, suppressed: bool = True) -> ToolResult:
        """抑制 / 恢复对象（Abaqus 的 Suppress / Resume）：抑制的对象不参与计算。"""
        return self._edit_feature(kind, name, {"suppressed": True if suppressed else None})

    @_records
    def delete_feature(self, kind: str, name: str) -> ToolResult:
        try:
            kind = self._feature_kind(kind)
        except ValueError as exc:
            return ToolResult(False, {"error": str(exc)})
        candidate = deepcopy(self.model)
        items = self._feature_list(candidate, kind)
        kept = [f for f in items if f["name"] != name]
        if len(kept) == len(items):
            return ToolResult(False, {"error": f"没有名为「{name}」的{_LABEL[kind]}",
                                      "available": [f["name"] for f in items]})
        candidate["features"][kind] = kept
        return self._commit_features(candidate, {"kind": kind, "deleted": name})

    def list_features(self, kind: str | None = None) -> ToolResult:
        """列出对象：名字、区域、展开后的编号、是否抑制、派生问题。"""
        kinds = [self._feature_kind(kind)] if kind else list(_features.KINDS)
        out: dict[str, list[dict[str, Any]]] = {}
        for key in kinds:
            rows = []
            for feature in (self.model.get("features") or {}).get(key) or []:
                rows.append({**{k: v for k, v in feature.items() if k != "region"},
                             **self._region_summary(self.model, key, feature),
                             "suppressed": bool(feature.get("suppressed"))})
            out[key] = rows
        return ToolResult(True, {"features": out})
