"""Session 的 Part 模块工具：分割杆件、基准点、测量（Abaqus 的 Partition / Datum / Query）。

* 分割杆件（Partition Edge）：在杆件上按比例或距离加一个顶点，之后可以把别的杆件
  连到这个点上。物理杆件仍是一根——编译时显式内节点会自动把它剖成两个分析杆段
  （见 model_compiler），所以杆上的荷载、杆端释放、截面指派都不用拆，也不会拆错。
* 基准点（Datum Point）：只用于定位与吸附的参考点，不参与计算。可按坐标、两节点中点
  或相对某节点的偏移创建；建节点时靠近基准点就吸附上去。
* 测量（Query）：两点距离与分量、两杆夹角、杆长。只读。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from session_base import ToolResult, _records


class PartMixin:
    """Part 模块工具。见模块 docstring。"""

    def _node_xyz(self, node_id: int) -> np.ndarray | None:
        node = next((n for n in self.model.get("nodes") or [] if int(n["id"]) == int(node_id)),
                    None)
        return None if node is None else np.array([node["x"], node["y"], node["z"]], float)

    @_records
    def partition_member(self, member_id: int, ratio: float | None = None,
                         distance: float | None = None) -> ToolResult:
        """在杆件上加一个顶点：``ratio`` 是从 i 端量起的比例（0~1），或 ``distance`` 是长度。"""
        member = next((m for m in self.model.get("members") or []
                       if int(m["id"]) == int(member_id)), None)
        if member is None:
            return ToolResult(False, {"error": f"杆件 {member_id} 不存在"})
        pi, pj = self._node_xyz(member["i"]), self._node_xyz(member["j"])
        length = float(np.linalg.norm(pj - pi))
        if (ratio is None) == (distance is None):
            return ToolResult(False, {"error": "ratio 与 distance 二选一",
                                      "example": {"member_id": member_id, "ratio": 0.5}})
        t = float(ratio) if ratio is not None else float(distance) / length
        if not np.isfinite(t) or not 0.0 < t < 1.0:
            return ToolResult(False, {"error": f"分割点要在杆件内部（比例 0~1 之间，不含端点），收到 {t:g}"})
        point = pi + t * (pj - pi)
        added = self.add_nodes.__wrapped__(self, [point.tolist()])
        if not added.ok:
            return added
        node_id = added.payload["input_node_ids"][0]
        return ToolResult(True, {
            "member": int(member_id), "node": node_id, "ratio": round(t, 9),
            "point": [round(float(v), 9) for v in point],
            "note": "物理杆件仍是一根，求解时在这个顶点自动剖分；可以把别的杆件连到这个节点上。"})

    # --- 基准点 ---

    @_records
    def create_datum_point(self, name: str, xyz: list[float] | None = None,
                           midpoint_of: list[int] | None = None,
                           offset_from: int | None = None,
                           offset: list[float] | None = None) -> ToolResult:
        """基准点：给坐标，或两节点中点（midpoint_of），或某节点加偏移（offset_from + offset）。"""
        name = str(name).strip()
        if not name:
            return ToolResult(False, {"error": "基准点名称不能为空"})
        given = [xyz is not None, midpoint_of is not None, offset_from is not None]
        if sum(given) != 1:
            return ToolResult(False, {"error": "xyz、midpoint_of、offset_from 三种给法选一种"})
        if xyz is not None:
            point = np.asarray(xyz, float)
        elif midpoint_of is not None:
            ends = [self._node_xyz(k) for k in midpoint_of]
            if len(ends) != 2 or any(p is None for p in ends):
                return ToolResult(False, {"error": "midpoint_of 要给两个存在的节点编号"})
            point = 0.5 * (ends[0] + ends[1])
        else:
            base = self._node_xyz(offset_from)
            if base is None:
                return ToolResult(False, {"error": f"节点 {offset_from} 不存在"})
            point = base + np.asarray(offset or (0.0, 0.0, 0.0), float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            return ToolResult(False, {"error": "基准点坐标必须是三个有限数"})
        datums = dict(self.model.get("datums") or {})
        points = [p for p in datums.get("points") or [] if p["name"] != name]
        replaced = len(points) != len(datums.get("points") or [])
        points.append({"name": name, "xyz": [float(v) for v in point]})
        datums["points"] = points
        self.model["datums"] = datums
        return ToolResult(True, {"name": name, "xyz": [round(float(v), 9) for v in point],
                                 "replaced": replaced,
                                 "note": "基准点只用于定位与吸附，不参与计算。"})

    @_records
    def delete_datum_point(self, name: str) -> ToolResult:
        datums = dict(self.model.get("datums") or {})
        points = datums.get("points") or []
        kept = [p for p in points if p["name"] != name]
        if len(kept) == len(points):
            return ToolResult(False, {"error": f"没有名为「{name}」的基准点",
                                      "available": [p["name"] for p in points]})
        if kept:
            datums["points"] = kept
            self.model["datums"] = datums
        else:
            self.model.pop("datums", None)
        return ToolResult(True, {"deleted": name})

    # --- 测量 ---

    def measure(self, nodes: list[int] | None = None,
                members: list[int] | None = None) -> ToolResult:
        """测量：两个节点给距离与分量；两根杆件给夹角；一根杆件给长度。只读。"""
        length_unit = "mm" if self.model.get("units") == "N-mm-MPa" else "m"
        if nodes:
            if len(nodes) != 2:
                return ToolResult(False, {"error": "测距离要恰好两个节点"})
            a, b = (self._node_xyz(k) for k in nodes)
            if a is None or b is None:
                return ToolResult(False, {"error": f"节点 {nodes} 中有不存在的"})
            d = b - a
            return ToolResult(True, {"nodes": list(nodes), "distance": float(np.linalg.norm(d)),
                                     "components": [float(v) for v in d], "unit": length_unit})
        if members:
            vectors = []
            for mid in members:
                m = next((x for x in self.model.get("members") or []
                          if int(x["id"]) == int(mid)), None)
                if m is None:
                    return ToolResult(False, {"error": f"杆件 {mid} 不存在"})
                vectors.append(self._node_xyz(m["j"]) - self._node_xyz(m["i"]))
            if len(vectors) == 1:
                return ToolResult(True, {"members": list(members),
                                         "length": float(np.linalg.norm(vectors[0])),
                                         "unit": length_unit})
            if len(vectors) != 2:
                return ToolResult(False, {"error": "测夹角要恰好两根杆件"})
            u, v = vectors
            cos = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
            angle = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
            return ToolResult(True, {"members": list(members), "angle_deg": angle,
                                     "note": "按两杆各自 i→j 方向的夹角，0~180°"})
        return ToolResult(False, {"error": "给两个节点测距离，或一两根杆件测长度、夹角"})

    def datum_points(self) -> dict[str, Any]:
        return {p["name"]: p["xyz"] for p in (self.model.get("datums") or {}).get("points") or []}
