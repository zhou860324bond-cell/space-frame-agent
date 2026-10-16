"""荷载定位与结果查询共用的原始坐标几何校验。"""

from __future__ import annotations

import numpy as np


def parse_coordinate(text: str) -> np.ndarray:
    """读取三个有限全局坐标，允许逗号、中文逗号或空格分隔。"""
    try:
        point = np.asarray([float(v) for v in
                            text.replace("，", " ").replace(",", " ").split()])
    except ValueError:
        raise ValueError("坐标必须是三个数字，例如 1, 2, 3；请检查 X、Y、Z。") from None
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError("坐标必须包含三个有限数字；请检查 X、Y、Z。")
    return point


def station_on_segment(point, start, end) -> float:
    """严格定位中心线上的点；不把离杆点或越界点静默吸附到杆端。"""
    point, start, end = (np.asarray(value, dtype=float)
                         for value in (point, start, end))
    if any(value.shape != (3,) or not np.all(np.isfinite(value))
           for value in (point, start, end)):
        raise ValueError("坐标必须包含三个有限数字。")
    axis = end - start
    length = float(np.linalg.norm(axis))
    if length == 0:
        raise ValueError("杆件长度为零；请修正两端节点坐标。")
    tolerance = max(length * 1e-8, np.finfo(float).eps *
                    max(float(np.max(np.abs(start))), length) * 32)
    station = float(np.dot(point - start, axis) / length)
    projected = start + station / length * axis
    if (station < -tolerance or station > length + tolerance
            or np.linalg.norm(point - projected) > tolerance):
        raise ValueError("坐标不在所选杆件中心线范围内；请检查位置或选择对应杆件。")
    return float(np.clip(station, 0, length))


def model_member_ends(model: dict, member_id: int):
    nodes = {int(node["id"]): node for node in model.get("nodes", [])}
    member = next((item for item in model.get("members", [])
                   if int(item["id"]) == member_id), None)
    if member is None:
        raise ValueError(f"杆件 {member_id} 不存在；请选择有效杆件。")
    # 位置与会话荷载接口一致：沿用户创建杆件的 i→j 轴量取。
    return tuple(np.asarray([nodes[int(member[end])][key] for key in "xyz"], float)
                 for end in ("i", "j"))
