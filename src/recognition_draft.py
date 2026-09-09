"""多模态识别草稿：图片理解与正式 Domain IR 之间的人机确认层。"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from draft import FrameDraft

DRAFT_FORMAT = "space-frame-recognition-draft/v1"
V2_DRAFT_FORMAT = "space-frame-recognition-draft/v2"


def migrate_to_v2(payload: dict[str, Any], *, image_hash: str = "",
                  source_path: str = "") -> dict[str, Any]:
    """Compatibility entry point; the existing v1 UI remains unchanged."""
    from multimodal_workflow import migrate_v1_payload
    return migrate_v1_payload(payload, image_hash=image_hash, source_path=source_path)


@dataclass
class RecognitionDraft:
    """允许缺属性、缺支座和未知尺度，但不允许坏拓扑。"""

    model: dict[str, Any]
    scale_status: str
    scale_evidence: str = ""
    questions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    entities: list[dict[str, Any]] = field(default_factory=list)
    source_image: str = ""

    @classmethod
    def from_payload(cls, payload: dict[str, Any],
                     source_image: str | Path = "") -> "RecognitionDraft":
        """读取 v1 草稿；同时兼容旧版直接返回完整模型的响应。"""
        if not isinstance(payload, dict):
            raise ValueError("识别结果必须是 JSON 对象")
        if payload.get("format") == DRAFT_FORMAT:
            model = deepcopy(payload.get("model"))
            scale = payload.get("scale") or {}
            status = str(scale.get("status", "unknown"))
            evidence = str(scale.get("evidence", ""))
            questions = payload.get("questions") or []
            warnings = payload.get("warnings") or []
            entities = payload.get("entities") or []
        elif "nodes" in payload and "members" in payload:
            # 旧提示词没有留下尺度证据，不能把坐标值直接当作米。
            model = deepcopy(payload)
            status, evidence = "unknown", ""
            questions = ["旧版识别结果没有尺度证据，请标定一根杆件的实际长度"]
            warnings, entities = ["旧版识别响应没有实体置信度"], []
        else:
            raise ValueError(f"识别结果缺少 format={DRAFT_FORMAT!r} 和 model")

        if not isinstance(model, dict):
            raise ValueError("识别草稿的 model 必须是 JSON 对象")
        if status not in ("confirmed", "unknown"):
            raise ValueError("scale.status 只能是 confirmed 或 unknown")
        if not isinstance(questions, list) or not all(isinstance(x, str) for x in questions):
            raise ValueError("questions 必须是字符串列表")
        if not isinstance(warnings, list) or not all(isinstance(x, str) for x in warnings):
            raise ValueError("warnings 必须是字符串列表")
        if not isinstance(entities, list) or not all(isinstance(x, dict) for x in entities):
            raise ValueError("entities 必须是对象列表")

        model.setdefault("units", "N-m-Pa")
        model.setdefault("materials", [])
        model.setdefault("sections", [])
        model.setdefault("supports", [])
        model.setdefault("load_cases", [])
        for member in model.get("members") or []:
            if isinstance(member, dict):
                member.setdefault("section", "")
                member.setdefault("material", "")

        draft = cls(model, status, evidence, list(questions),
                    list(warnings), deepcopy(entities), str(source_image))
        errors = draft.validation_errors()
        if errors:
            raise ValueError("识别草稿无效：" + "；".join(errors))
        if status == "unknown" and not any("尺度" in item for item in draft.questions):
            draft.questions.append("图中缺少可确认的尺度，请标定一根杆件的实际长度")
        return draft

    def validation_errors(self) -> list[str]:
        errors: list[str] = []
        nodes = self.model.get("nodes")
        members = self.model.get("members")
        if not isinstance(nodes, list) or len(nodes) < 2:
            errors.append("至少需要识别两个节点")
            return errors
        if not isinstance(members, list) or not members:
            errors.append("至少需要识别一根杆件")
            return errors

        node_ids: list[int] = []
        coordinates: dict[int, tuple[float, float, float]] = {}
        for node in nodes:
            try:
                node_id = int(node["id"])
                xyz = tuple(float(node[key]) for key in ("x", "y", "z"))
            except (KeyError, TypeError, ValueError):
                errors.append("节点必须包含整数 id 和数值 x/y/z")
                continue
            if not all(math.isfinite(value) for value in xyz):
                errors.append(f"节点 {node_id} 坐标必须是有限数")
            node_ids.append(node_id)
            coordinates[node_id] = xyz
        if len(set(node_ids)) != len(node_ids):
            errors.append("节点 id 重复")

        known = set(node_ids)
        member_ids: list[int] = []
        edges: set[tuple[int, int]] = set()
        for member in members:
            try:
                member_id = int(member["id"])
                i, j = int(member["i"]), int(member["j"])
            except (KeyError, TypeError, ValueError):
                errors.append("杆件必须包含整数 id/i/j")
                continue
            member_ids.append(member_id)
            if i not in known or j not in known:
                errors.append(f"杆件 {member_id} 引用了不存在的节点")
                continue
            if i == j or math.dist(coordinates[i], coordinates[j]) <= 1e-12:
                errors.append(f"杆件 {member_id} 长度为零")
            edge = (min(i, j), max(i, j))
            if edge in edges:
                errors.append(f"节点 {i} 与 {j} 之间存在重复杆件")
            edges.add(edge)
        if len(set(member_ids)) != len(member_ids):
            errors.append("杆件 id 重复")
        known_members = set(member_ids)

        for support in self.model.get("supports") or []:
            try:
                node_id = int(support["node"])
                fix = list(support["fix"])
            except (KeyError, TypeError, ValueError):
                errors.append("支座必须包含节点编号和 fix")
                continue
            if node_id not in known:
                errors.append(f"支座引用了不存在的节点 {node_id}")
            if len(fix) != 6 or any(value not in (0, 1) for value in fix):
                errors.append(f"节点 {node_id} 的支座 fix 必须是六个 0 或 1")

        def finite_vector(entry: dict[str, Any], key: str, size: int,
                          label: str) -> None:
            try:
                values = [float(value) for value in entry[key]]
            except (KeyError, TypeError, ValueError):
                errors.append(f"{label} 的 {key} 必须是 {size} 个数")
                return
            if len(values) != size or not all(math.isfinite(value) for value in values):
                errors.append(f"{label} 的 {key} 必须是 {size} 个有限数")

        def validate_loads(container: Any, label: str) -> None:
            if not isinstance(container, dict):
                errors.append(f"{label or '模型'}荷载容器必须是对象")
                return
            for entry in container.get("nodal_loads") or []:
                if not isinstance(entry, dict):
                    errors.append(f"{label}节点荷载必须是对象")
                    continue
                try:
                    node_id = int(entry.get("node", -1))
                except (TypeError, ValueError):
                    node_id = -1
                if node_id not in known:
                    errors.append(f"{label}节点荷载引用了不存在的节点 {node_id}")
                finite_vector(entry, "load", 6, f"{label}节点荷载")
            for entry in container.get("settlements") or []:
                if not isinstance(entry, dict):
                    errors.append(f"{label}给定位移必须是对象")
                    continue
                try:
                    node_id = int(entry.get("node", -1))
                except (TypeError, ValueError):
                    node_id = -1
                if node_id not in known:
                    errors.append(f"{label}给定位移引用了不存在的节点 {node_id}")
                finite_vector(entry, "d", 6, f"{label}给定位移")
            for key, vector_key in (("member_loads", "w"),
                                    ("member_spans", "w1")):
                for entry in container.get(key) or []:
                    if not isinstance(entry, dict):
                        errors.append(f"{label}杆件荷载必须是对象")
                        continue
                    try:
                        member_id = int(entry.get("member", -1))
                    except (TypeError, ValueError):
                        member_id = -1
                    if member_id not in known_members:
                        errors.append(f"{label}杆件荷载引用了不存在的杆件 {member_id}")
                    finite_vector(entry, vector_key, 3, f"{label}杆件荷载")

        validate_loads(self.model, "")
        for case in self.model.get("load_cases") or []:
            case_name = case.get("name", "未命名") if isinstance(case, dict) else "未命名"
            validate_loads(case, f"工况 {case_name} 的")

        entity_keys: set[tuple[str, str]] = set()
        for entity in self.entities:
            kind = str(entity.get("kind", ""))
            entity_id = str(entity.get("id", ""))
            key = kind, entity_id
            if key in entity_keys:
                errors.append(f"识别实体 {kind} {entity_id} 重复")
            entity_keys.add(key)
            if kind == "node":
                try:
                    if int(entity_id) not in known:
                        errors.append(f"节点实体引用了不存在的节点 {entity_id}")
                except ValueError:
                    errors.append("节点实体 id 必须是整数")
            elif kind == "member":
                try:
                    if int(entity_id) not in known_members:
                        errors.append(f"杆件实体引用了不存在的杆件 {entity_id}")
                except ValueError:
                    errors.append("杆件实体 id 必须是整数")

            confidence = entity.get("confidence")
            if confidence is not None:
                try:
                    value = float(confidence)
                except (TypeError, ValueError):
                    errors.append("实体 confidence 必须是 0 到 1 的数字")
                    continue
                if not 0.0 <= value <= 1.0:
                    errors.append("实体 confidence 必须介于 0 和 1")
            geometry = entity.get("image_geometry") or {}
            if not isinstance(geometry, dict):
                errors.append("实体 image_geometry 必须是对象")
                continue
            coordinate_groups = []
            if "point" in geometry:
                coordinate_groups.append(geometry["point"])
            if "line" in geometry:
                line = geometry["line"]
                coordinate_groups.extend(line if isinstance(line, list) else [line])
            if "bbox" in geometry:
                bbox = geometry["bbox"]
                if isinstance(bbox, list) and len(bbox) == 4:
                    coordinate_groups.extend((bbox[:2], bbox[2:]))
                else:
                    coordinate_groups.append(bbox)
            for point in coordinate_groups:
                if (not isinstance(point, (list, tuple)) or len(point) != 2):
                    errors.append(f"识别实体 {kind} {entity_id} 的图片坐标格式无效")
                    continue
                try:
                    values = [float(value) for value in point]
                except (TypeError, ValueError):
                    errors.append(f"识别实体 {kind} {entity_id} 的图片坐标必须是数字")
                    continue
                if not all(math.isfinite(value) and 0.0 <= value <= 1.0
                           for value in values):
                    errors.append(f"识别实体 {kind} {entity_id} 的图片坐标必须介于 0 和 1")
            target = entity.get("target")
            if target is not None and not isinstance(target, dict):
                errors.append(f"识别实体 {kind} {entity_id} 的 target 必须是对象")
            elif isinstance(target, dict) and "node" in target:
                try:
                    target_node = int(target["node"])
                except (TypeError, ValueError):
                    errors.append(f"识别实体 {kind} {entity_id} 的目标节点必须是整数")
                else:
                    if target_node not in known:
                        errors.append(
                            f"识别实体 {kind} {entity_id} 引用了不存在的节点 {target_node}")
        return errors

    @property
    def ready_to_load(self) -> bool:
        return self.scale_status == "confirmed" and not self.validation_errors()

    def _node_entity(self, node_id: int) -> dict[str, Any] | None:
        return next((entity for entity in self.entities
                     if entity.get("kind") == "node"
                     and str(entity.get("id")) == str(node_id)), None)

    @staticmethod
    def _image_point(entity: dict[str, Any] | None) -> tuple[float, float] | None:
        point = (entity or {}).get("image_geometry", {}).get("point")
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return None
        try:
            values = float(point[0]), float(point[1])
        except (TypeError, ValueError):
            return None
        return values if all(0.0 <= value <= 1.0 for value in values) else None

    def _fallback_image_scale(self) -> float:
        """估算模型单位/图片归一化距离，只用于拖动二维识别节点。"""
        nodes = {int(node["id"]): node for node in self.model["nodes"]}
        scales: list[float] = []
        for member in self.model["members"]:
            a, b = nodes[int(member["i"])], nodes[int(member["j"])]
            pa = self._image_point(self._node_entity(int(member["i"])))
            pb = self._image_point(self._node_entity(int(member["j"])))
            if pa is None or pb is None:
                continue
            image_length = math.dist(pa, pb)
            if image_length > 1e-12:
                model_length = math.dist(
                    tuple(float(a[key]) for key in ("x", "z")),
                    tuple(float(b[key]) for key in ("x", "z")))
                if model_length > 1e-12:
                    scales.append(model_length / image_length)
        return sum(scales) / len(scales) if scales else 1.0

    def move_node_image(self, node_id: int, u: float, v: float) -> None:
        """在原图中拖动节点，并按当前二维轴向比例同步模型 x/z 坐标。"""
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in (u, v)):
            raise ValueError("图片坐标必须是 0 到 1 的有限数")
        node = next((item for item in self.model["nodes"]
                     if int(item["id"]) == int(node_id)), None)
        entity = self._node_entity(node_id)
        old_point = self._image_point(entity)
        if node is None or entity is None or old_point is None:
            raise ValueError(f"节点 {node_id} 缺少可编辑的图片定位")

        located = []
        for item in self.model["nodes"]:
            point = self._image_point(self._node_entity(int(item["id"])))
            if point is not None:
                located.append((item, point))
        fallback = self._fallback_image_scale()
        image_u = [point[0] for _, point in located]
        image_v = [point[1] for _, point in located]
        model_x = [float(item["x"]) for item, _ in located]
        model_z = [float(item["z"]) for item, _ in located]
        u_span, v_span = max(image_u) - min(image_u), max(image_v) - min(image_v)
        x_span, z_span = max(model_x) - min(model_x), max(model_z) - min(model_z)
        scale_x = x_span / u_span if u_span > 1e-12 and x_span > 1e-12 else fallback
        scale_z = z_span / v_span if v_span > 1e-12 and z_span > 1e-12 else fallback

        before = deepcopy((self.model, self.entities))
        node["x"] = float(node["x"]) + (u - old_point[0]) * scale_x
        node["z"] = float(node["z"]) - (v - old_point[1]) * scale_z
        entity.setdefault("image_geometry", {})["point"] = [u, v]
        for attached in self.entities:
            if (attached.get("source") == "user"
                    and self._target_node(attached) == int(node_id)
                    and attached.get("kind") in ("support", "load")):
                bbox = self._node_bbox(node_id)
                if bbox is not None:
                    attached.setdefault("image_geometry", {})["bbox"] = bbox
        self._sync_member_lines()
        errors = self.validation_errors()
        if errors:
            self.model, self.entities = before
            raise ValueError("拖动后草稿无效：" + "；".join(errors))

    def _sync_member_lines(self) -> None:
        for member in self.model["members"]:
            start = self._image_point(self._node_entity(int(member["i"])))
            end = self._image_point(self._node_entity(int(member["j"])))
            entity = next((item for item in self.entities
                           if item.get("kind") == "member"
                           and str(item.get("id")) == str(member["id"])), None)
            if entity is not None and start is not None and end is not None:
                entity.setdefault("image_geometry", {})["line"] = [list(start), list(end)]

    def add_member(self, i: int, j: int) -> int:
        """在两个已有识别节点之间补一根杆件。"""
        known = {int(node["id"]) for node in self.model["nodes"]}
        if i not in known or j not in known:
            raise ValueError("补画杆件只能连接已有节点")
        if i == j:
            raise ValueError("杆件两端不能是同一节点")
        if any({int(item["i"]), int(item["j"])} == {int(i), int(j)}
               for item in self.model["members"]):
            raise ValueError(f"节点 {i} 与 {j} 之间已有杆件")
        member_id = max((int(item["id"]) for item in self.model["members"]), default=0) + 1
        self.model["members"].append({
            "id": member_id, "i": int(i), "j": int(j),
            "section": "", "material": "",
        })
        start = self._image_point(self._node_entity(i))
        end = self._image_point(self._node_entity(j))
        geometry = {"line": [list(start), list(end)]} if start and end else {}
        self.entities.append({
            "kind": "member", "id": member_id, "confidence": 1.0,
            "image_geometry": geometry, "source": "user",
        })
        return member_id

    def remove_member(self, member_id: int) -> None:
        """删除误识别杆件；仍被荷载引用时拒绝，避免悄悄丢荷载。"""
        if not any(int(item["id"]) == int(member_id) for item in self.model["members"]):
            raise ValueError(f"识别草稿中没有杆件 {member_id}")
        if len(self.model["members"]) == 1:
            raise ValueError("识别草稿至少需要保留一根杆件")
        containers = [self.model, *(self.model.get("load_cases") or [])]
        for container in containers:
            if not isinstance(container, dict):
                continue
            loads = [*(container.get("member_loads") or []),
                     *(container.get("member_spans") or [])]
            if any(isinstance(load, dict)
                   and int(load.get("member", -1)) == int(member_id) for load in loads):
                raise ValueError(f"杆件 {member_id} 仍被荷载引用，请先处理荷载")
        self.model["members"] = [item for item in self.model["members"]
                                 if int(item["id"]) != int(member_id)]
        self.entities = [item for item in self.entities
                         if not (item.get("kind") == "member"
                                 and str(item.get("id")) == str(member_id))]

    def _node_bbox(self, node_id: int) -> list[float] | None:
        point = self._image_point(self._node_entity(node_id))
        if point is None:
            return None
        radius = 0.035
        return [max(0.0, point[0] - radius), max(0.0, point[1] - radius),
                min(1.0, point[0] + radius), min(1.0, point[1] + radius)]

    @staticmethod
    def _target_node(entity: dict[str, Any]) -> int | None:
        try:
            return int((entity.get("target") or {})["node"])
        except (KeyError, TypeError, ValueError):
            return None

    def set_support(self, node_id: int, fix: list[int], name: str = "") -> None:
        """新增或修改节点支座；全零约束应使用 remove_support 明确删除。"""
        known = {int(node["id"]) for node in self.model["nodes"]}
        if int(node_id) not in known:
            raise ValueError(f"支座只能设置在已有节点，未找到节点 {node_id}")
        values = list(fix)
        if len(values) != 6 or any(value not in (0, 1) for value in values):
            raise ValueError("支座 fix 必须是六个 0 或 1")
        if not any(values):
            raise ValueError("全零约束不是支座，请使用删除支座")
        support = next((item for item in self.model["supports"]
                        if int(item["node"]) == int(node_id)), None)
        payload = {"node": int(node_id), "fix": values}
        if name.strip():
            payload["name"] = name.strip()
        if support is None:
            self.model["supports"].append(payload)
        else:
            support.clear()
            support.update(payload)

        entity = next((item for item in self.entities
                       if item.get("kind") == "support"
                       and self._target_node(item) == int(node_id)), None)
        geometry = {}
        bbox = self._node_bbox(node_id)
        if bbox is not None:
            geometry["bbox"] = bbox
        if entity is None:
            self.entities.append({
                "kind": "support", "id": f"S{node_id}", "confidence": 1.0,
                "target": {"node": int(node_id)},
                "image_geometry": geometry, "source": "user",
            })
        else:
            entity.update({"confidence": 1.0, "target": {"node": int(node_id)},
                           "image_geometry": geometry, "source": "user"})

    def remove_support(self, node_id: int) -> None:
        if not any(int(item["node"]) == int(node_id)
                   for item in self.model["supports"]):
            raise ValueError(f"节点 {node_id} 没有支座")
        self.model["supports"] = [item for item in self.model["supports"]
                                  if int(item["node"]) != int(node_id)]
        self.entities = [item for item in self.entities
                         if not (item.get("kind") == "support"
                                 and self._target_node(item) == int(node_id))]

    def add_load_case(self, name: str) -> None:
        case_name = name.strip()
        if not case_name:
            raise ValueError("工况名称不能为空")
        if any(str(case.get("name", "")) == case_name
               for case in self.model["load_cases"]):
            raise ValueError(f"工况 {case_name} 已存在")
        self.model["load_cases"].append({"name": case_name, "nodal_loads": []})

    def set_nodal_load(self, case_name: str, node_id: int, load: list[float],
                       name: str) -> None:
        """在已有工况中新增或按名称更新一个节点荷载。"""
        known = {int(node["id"]) for node in self.model["nodes"]}
        if int(node_id) not in known:
            raise ValueError(f"节点荷载引用了不存在的节点 {node_id}")
        load_name = name.strip()
        if not load_name:
            raise ValueError("节点荷载名称不能为空")
        try:
            values = [float(value) for value in load]
        except (TypeError, ValueError):
            raise ValueError("节点荷载必须是六个数") from None
        if len(values) != 6 or not all(math.isfinite(value) for value in values):
            raise ValueError("节点荷载必须是六个有限数")
        if not any(abs(value) > 0.0 for value in values):
            raise ValueError("节点荷载不能全为零，请使用删除荷载")
        case = next((item for item in self.model["load_cases"]
                     if str(item.get("name", "")) == case_name), None)
        if case is None:
            raise ValueError(f"工况 {case_name} 不存在，请先新建工况")
        entries = case.setdefault("nodal_loads", [])
        entry = next((item for item in entries
                      if str(item.get("name", "")) == load_name), None)
        payload = {"name": load_name, "node": int(node_id), "load": values}
        if entry is None:
            entries.append(payload)
        else:
            entry.clear()
            entry.update(payload)

        entity_id = f"{case_name}:{load_name}"
        entity = next((item for item in self.entities
                       if item.get("kind") == "load"
                       and str(item.get("id")) == entity_id), None)
        geometry = {}
        bbox = self._node_bbox(node_id)
        if bbox is not None:
            geometry["bbox"] = bbox
        entity_payload = {
            "kind": "load", "id": entity_id, "confidence": 1.0,
            "target": {"case": case_name, "node": int(node_id),
                       "name": load_name},
            "image_geometry": geometry, "source": "user",
        }
        if entity is None:
            self.entities.append(entity_payload)
        else:
            entity.clear()
            entity.update(entity_payload)

    def remove_nodal_load(self, case_name: str, name: str) -> None:
        case = next((item for item in self.model["load_cases"]
                     if str(item.get("name", "")) == case_name), None)
        if case is None:
            raise ValueError(f"工况 {case_name} 不存在")
        entries = case.get("nodal_loads") or []
        if not any(str(item.get("name", "")) == name for item in entries):
            raise ValueError(f"工况 {case_name} 中没有节点荷载 {name}")
        case["nodal_loads"] = [item for item in entries
                               if str(item.get("name", "")) != name]
        entity_id = f"{case_name}:{name}"
        self.entities = [item for item in self.entities
                         if not (item.get("kind") == "load"
                                 and str(item.get("id")) == entity_id)]

    def calibrate(self, member_id: int, actual_length_m: float) -> None:
        """用一根已识别杆件的真实长度，把相对坐标统一换算成米。"""
        if not math.isfinite(actual_length_m) or actual_length_m <= 0:
            raise ValueError("实际长度必须是大于 0 的有限数")
        member = next((item for item in self.model["members"]
                       if int(item["id"]) == int(member_id)), None)
        if member is None:
            raise ValueError(f"识别草稿中没有杆件 {member_id}")
        nodes = {int(item["id"]): item for item in self.model["nodes"]}
        a, b = nodes[int(member["i"])], nodes[int(member["j"])]
        current = math.dist(tuple(float(a[k]) for k in ("x", "y", "z")),
                            tuple(float(b[k]) for k in ("x", "y", "z")))
        if current <= 1e-12:
            raise ValueError(f"杆件 {member_id} 的识别长度为零，无法标定")
        factor = actual_length_m / current
        for node in self.model["nodes"]:
            for key in ("x", "y", "z"):
                node[key] = float(node[key]) * factor
        self.model["units"] = "N-m-Pa"
        self.scale_status = "confirmed"
        self.scale_evidence = f"用户标定杆件 {member_id} = {actual_length_m:g} m"

    def to_frame_draft(self) -> FrameDraft:
        if not self.ready_to_load:
            raise ValueError("尺度尚未确认，不能加载识别草稿")
        return FrameDraft.from_model(self.model)
