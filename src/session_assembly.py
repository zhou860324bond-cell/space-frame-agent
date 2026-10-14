"""Session 的 Assembly 模块：部件实例、线性阵列、环形阵列（Abaqus 的 Instance / Pattern）。

做的是 Abaqus 的**独立实例**（Independent instance）：选一组杆件当作部件，按平移、
旋转放一份或阵列多份，副本是模型里实实在在的节点与杆件，可以单独改、单独加荷载。
每份副本记成一个有名字的实例，整份删除时连同它新建的节点、带过去的属性一起撤掉。

不做非独立实例（改部件、各实例跟着变）：那要求派生节点的编号稳定，而集合、荷载、
边界条件都按编号引用节点，部件一改编号就变，引用会悄悄指错。

语义与 Abaqus 一致的几条：

* 截面指派、梁方向属于部件，随副本走：为副本建同参数的对象（旋转副本的 n1 跟着转）；
* 荷载与边界条件属于装配，不随副本复制——新柱脚要另加边界条件，结果里列出新节点；
* 副本与已有结构坐标重合的节点直接共用（相当于 Merge），重合的杆件跳过并报告。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

import features as _features
from session_base import ToolResult, _records
from session_features import _region


def rotation_matrix(axis, angle_deg: float) -> np.ndarray:
    """绕单位轴 axis 转 angle_deg 度的旋转矩阵（右手定则，Rodrigues 公式）。"""
    a = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(a))
    if a.shape != (3,) or not np.isfinite(norm) or norm == 0.0:
        raise ValueError("旋转轴要是三个有限数，且不全为零")
    k = a / norm
    t = np.radians(float(angle_deg))
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(t) * K + (1.0 - np.cos(t)) * (K @ K)


def _vector(value, label: str) -> np.ndarray:
    v = np.asarray(value if value is not None else (0.0, 0.0, 0.0), dtype=float)
    if v.shape != (3,) or not np.all(np.isfinite(v)):
        raise ValueError(f"{label}要是三个有限数")
    return v


def _unique(name: str, taken: set[str]) -> str:
    if name not in taken:
        return name
    k = 2
    while f"{name}-{k}" in taken:
        k += 1
    return f"{name}-{k}"


class AssemblyMixin:
    """Assembly 模块工具。见模块 docstring。"""

    # --- 公用 ---

    def _instances(self, model: dict) -> list[dict[str, Any]]:
        return model.setdefault("assembly", {}).setdefault("instances", [])

    def _instance_source(self, region) -> tuple[dict, list[int]]:
        reg = _region(region, "members")
        ids, problem = _features.region_ids(self.model, reg, "members")
        if problem:
            raise ValueError(f"部件区域：{problem}")
        return reg, ids

    def _place_copy(self, model: dict, ids: list[int], R: np.ndarray, point: np.ndarray,
                    shift: np.ndarray) -> tuple[dict[int, int], list[int], list[int]]:
        """把杆件 ids 按 x' = R(x − p) + p + t 复制一份，就地写进 model。

        返回 (原杆件 → 副本杆件, 新建节点, 因与已有杆件重合而跳过的原杆件)。
        """
        from scipy.spatial import cKDTree

        tol = 1e-6 / self.units.length_to_m
        nodes = {int(n["id"]): n for n in model["nodes"]}
        members = {int(m["id"]): m for m in model["members"]}
        known_ids = list(nodes)
        tree = cKDTree(np.array([[n["x"], n["y"], n["z"]] for n in nodes.values()], float))
        next_node = max(nodes) + 1
        fresh: list[tuple[int, np.ndarray]] = []
        mapping: dict[int, int] = {}
        for nid in sorted({int(members[m][end]) for m in ids for end in ("i", "j")}):
            x = np.array([nodes[nid][k] for k in "xyz"], dtype=float)
            y = R @ (x - point) + point + shift
            dist, idx = tree.query(y)
            hit = known_ids[int(idx)] if dist <= tol else next(
                (k for k, xyz in fresh if float(np.linalg.norm(xyz - y)) <= tol), None)
            if hit is None:
                hit = next_node
                next_node += 1
                model["nodes"].append({"id": hit, "x": float(y[0]), "y": float(y[1]),
                                       "z": float(y[2])})
                fresh.append((hit, y))
            mapping[nid] = hit
        seen = {(min(int(m["i"]), int(m["j"])), max(int(m["i"]), int(m["j"])))
                for m in model["members"]}
        next_member = max(members) + 1
        copies: dict[int, int] = {}
        skipped: list[int] = []
        for mid in ids:
            source = members[mid]
            i, j = mapping[int(source["i"])], mapping[int(source["j"])]
            key = (min(i, j), max(i, j))
            if i == j or key in seen:
                skipped.append(mid)
                continue
            seen.add(key)
            entry = {k: deepcopy(v) for k, v in source.items()
                     if k not in ("id", "i", "j", "assignment", "orientation")}
            if source.get("assignment"):
                entry["section"], entry["material"] = "", ""   # 由带过去的截面指派填
            if source.get("orientation"):
                entry.pop("ref_vector", None)                  # 由带过去的梁方向填
            for key_ in ("ref_vector", "offset_i", "offset_j"):
                if key_ in entry:
                    entry[key_] = [float(v) for v in R @ np.asarray(entry[key_], float)]
            entry.update({"id": next_member, "i": i, "j": j})
            model["members"].append(entry)
            copies[mid] = next_member
            next_member += 1
        return copies, [k for k, _ in fresh], skipped

    def _carry_features(self, model: dict, label: str,
                        placed: list[tuple[np.ndarray, dict[int, int]]]) -> list[dict[str, str]]:
        """截面指派与梁方向属于部件：给副本建同参数的对象，旋转副本的 n1 跟着转。"""
        created: list[dict[str, str]] = []
        feats = model.setdefault("features", {})
        for kind in ("section_assignments", "beam_orientations"):
            taken = {f["name"] for f in feats.get(kind) or []}
            for feature in [f for f in feats.get(kind) or [] if not f.get("suppressed")]:
                ids, problem = _features.region_ids(model, feature["region"], "members")
                if problem:
                    continue
                if kind == "section_assignments":
                    groups = [(None, sorted(c[o] for _R, c in placed for o in ids if o in c))]
                else:
                    # 每份副本的转角可能不同，n1 各转各的；纯平移的可以合成一个对象
                    same = all(np.allclose(R, np.eye(3)) for R, _c in placed)
                    groups = ([(np.eye(3), sorted(c[o] for _R, c in placed for o in ids if o in c))]
                              if same else [(R, sorted(c[o] for o in ids if o in c))
                                            for R, c in placed])
                for R, members in groups:
                    if not members:
                        continue
                    name = _unique(f"{feature['name']}-{label}", taken)
                    taken.add(name)
                    copy = {k: deepcopy(v) for k, v in feature.items() if k != "region"}
                    copy.update({"name": name, "region": {"members": members}})
                    if R is not None:
                        copy["n1"] = [float(v) for v in R @ np.asarray(feature["n1"], float)]
                    feats.setdefault(kind, []).append(copy)
                    created.append({"kind": kind, "name": name})
        return created

    def _pattern(self, label: str, region, placements: list[tuple[str, np.ndarray, np.ndarray,
                                                                   np.ndarray, dict]],
                 pattern: str | None) -> ToolResult:
        """放若干份副本并记成实例；placements 是 [(实例名, R, 点, 平移, 记录的变换)]。"""
        try:
            reg, ids = self._instance_source(region)
        except ValueError as exc:
            return ToolResult(False, {"error": str(exc)})
        taken = {i["name"] for i in (self.model.get("assembly") or {}).get("instances") or []}
        clash = sorted({p[0] for p in placements} & taken)
        if clash:
            return ToolResult(False, {"error": f"已有名为 {clash} 的实例，请换一个名字"})
        candidate = deepcopy(self.model)
        placed, records, new_nodes, skipped = [], [], [], {}
        for name, R, point, shift, transform in placements:
            copies, fresh, skip = self._place_copy(candidate, ids, R, point, shift)
            placed.append((R, copies))
            new_nodes += fresh
            if skip:
                skipped[name] = skip
            records.append({"name": name, **({"pattern": pattern} if pattern else {}),
                            "source": reg, "transform": transform,
                            "members": sorted(copies.values()), "nodes": fresh})
        carried = self._carry_features(candidate, label, placed)
        for record in records:
            record["features"] = carried
        self._instances(candidate).extend(records)
        summary = {"instances": [r["name"] for r in records],
                   "members": [m for r in records for m in r["members"]],
                   "new_nodes": new_nodes, "carried_features": carried,
                   "note": "荷载与边界条件不随实例复制（与 Abaqus 相同）；新节点上的支座要另加。"}
        if skipped:
            summary["skipped_overlapping"] = skipped
        return self._commit_features(candidate, summary)

    # --- 工具 ---

    @_records
    def create_instance(self, name: str, region, translate: list[float] | None = None,
                        rotate_axis: list[float] | None = None, rotate_angle: float = 0.0,
                        rotate_point: list[float] | None = None) -> ToolResult:
        """实例（Create Instance + Translate / Rotate）：把一组杆件复制一份放到新位置。

        先绕过 ``rotate_point``、方向 ``rotate_axis`` 的轴转 ``rotate_angle`` 度，再平移
        ``translate``。区域可以是集合名或杆件编号。
        """
        try:
            shift = _vector(translate, "平移")
            point = _vector(rotate_point, "旋转中心")
            R = rotation_matrix(rotate_axis, rotate_angle) if rotate_angle else np.eye(3)
        except ValueError as exc:
            return ToolResult(False, {"error": str(exc)})
        if not rotate_angle and not shift.any():
            return ToolResult(False, {"error": "平移与转角都是零，副本会与原杆件完全重合"})
        transform: dict[str, Any] = {"translate": shift.tolist()}
        if rotate_angle:
            transform["rotate"] = {"axis": list(map(float, rotate_axis)), "angle_deg": float(rotate_angle),
                                   "point": point.tolist()}
        return self._pattern(str(name).strip(), region,
                             [(str(name).strip(), R, point, shift, transform)], None)

    @_records
    def linear_pattern(self, name: str, region, direction1: list[float], spacing1: float,
                       number1: int, direction2: list[float] | None = None,
                       spacing2: float = 0.0, number2: int = 1) -> ToolResult:
        """线性阵列（Linear Pattern）：沿一个或两个方向等距复制，number 含原件。

        沿 x 每 6 m 一跨共 4 跨：direction1=[1,0,0]、spacing1=6、number1=4，得到 3 份副本。
        """
        try:
            d1 = _vector(direction1, "方向 1")
            d2 = _vector(direction2 if direction2 is not None else (0, 0, 0), "方向 2")
        except ValueError as exc:
            return ToolResult(False, {"error": str(exc)})
        if int(number1) < 1 or int(number2) < 1 or int(number1) * int(number2) < 2:
            return ToolResult(False, {"error": "number1 × number2 至少为 2（含原件）"})
        if not np.linalg.norm(d1) or (int(number2) > 1 and not np.linalg.norm(d2)):
            return ToolResult(False, {"error": "阵列方向不能是零向量"})
        u1 = d1 / np.linalg.norm(d1)
        u2 = d2 / np.linalg.norm(d2) if np.linalg.norm(d2) else d2
        placements = []
        for a in range(int(number1)):
            for b in range(int(number2)):
                if a == 0 and b == 0:
                    continue
                shift = a * float(spacing1) * u1 + b * float(spacing2) * u2
                placements.append((f"{name}-lin-{a + 1}-{b + 1}", np.eye(3), np.zeros(3), shift,
                                   {"translate": shift.tolist()}))
        return self._pattern(str(name).strip(), region, placements, str(name).strip())

    @_records
    def radial_pattern(self, name: str, region, number: int, angle_deg: float,
                       axis: list[float], point: list[float] | None = None) -> ToolResult:
        """环形阵列（Radial Pattern）：绕轴每隔 ``angle_deg`` 度复制一份，number 含原件。"""
        if int(number) < 2:
            return ToolResult(False, {"error": "number 至少为 2（含原件）"})
        try:
            center = _vector(point, "旋转中心")
            rotation_matrix(axis, 1.0)
        except ValueError as exc:
            return ToolResult(False, {"error": str(exc)})
        placements = []
        for k in range(1, int(number)):
            angle = k * float(angle_deg)
            placements.append((f"{name}-rad-{k + 1}", rotation_matrix(axis, angle), center,
                               np.zeros(3),
                               {"translate": [0.0, 0.0, 0.0],
                                "rotate": {"axis": list(map(float, axis)), "angle_deg": angle,
                                           "point": center.tolist()}}))
        return self._pattern(str(name).strip(), region, placements, str(name).strip())

    @_records
    def delete_instance(self, name: str) -> ToolResult:
        """删除实例（或整个阵列）：撤掉它新建的杆件、不再被用到的节点，以及带过去的属性对象。"""
        from changes import Change, apply as apply_change

        records = [r for r in (self.model.get("assembly") or {}).get("instances") or []
                   if name in (r["name"], r.get("pattern"))]
        if not records:
            names = [r["name"] for r in (self.model.get("assembly") or {}).get("instances") or []]
            return ToolResult(False, {"error": f"没有名为「{name}」的实例或阵列", "available": names})
        candidate = deepcopy(self.model)
        drop = {m for r in records for m in r["members"]}
        candidate["members"] = [m for m in candidate["members"] if int(m["id"]) not in drop]
        gone = {r["name"] for r in records}
        candidate["assembly"]["instances"] = [r for r in candidate["assembly"]["instances"]
                                              if r["name"] not in gone]
        report = apply_change(candidate, Change.GEOMETRY, dropped_members=drop)
        # 带过去的属性对象：区域被剪空的一并删掉，没剪空的（别的实例还在用）留着
        feats = candidate.get("features") or {}
        for item in {(f["kind"], f["name"]) for r in records for f in r.get("features") or []}:
            kind, fname = item
            feature = next((f for f in feats.get(kind) or [] if f["name"] == fname), None)
            if feature is not None and not (feature["region"].get("members") or []):
                feats[kind] = [f for f in feats[kind] if f["name"] != fname]
        if not candidate["assembly"]["instances"]:
            candidate.pop("assembly")
        return self._commit_features(candidate, {
            "deleted": sorted(gone), "removed_members": sorted(drop),
            "removed_orphan_nodes": report.get("removed_orphan_nodes", [])})

    def list_instances(self) -> ToolResult:
        rows = [{"name": r["name"], "pattern": r.get("pattern"), "source": r["source"],
                 "transform": r["transform"], "members": len(r["members"]),
                 "new_nodes": len(r["nodes"])}
                for r in (self.model.get("assembly") or {}).get("instances") or []]
        return ToolResult(True, {"instances": rows})
