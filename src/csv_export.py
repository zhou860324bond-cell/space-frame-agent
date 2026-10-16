"""完整模型及已生成结果的单文件 CSV 导出，不触发新的求解。"""

from __future__ import annotations

import csv
import io
import json

import numpy as np

ANALYSIS_MODULES = ("modal", "buckling", "envelope", "deflection", "spectrum",
                    "strength", "symmetry", "numbering", "generic", "solid_joint",
                    "section_optimizer")


def export_csv(session, *, model=None, analyses=None, solid=None) -> str:
    """用模块、JSON 路径和值保留嵌套内容、空模块和每个工况的结果。"""
    from model_tables import from_tables, to_tables

    payload = session.model if model is None else model
    same_model = (payload == session.model or
                  payload == from_tables(session.model, to_tables(session.model)))
    if same_model:
        payload = session.model
    buf = io.StringIO(newline="")
    writer = csv.writer(buf, lineterminator="\r\n")
    writer.writerow(("module", "path", "value", "unit", "status"))

    def emit(module, value, path="", unit="", status="AVAILABLE"):
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, dict) and value:
            unit = str(value.get("unit", unit))
            for key, child in value.items():
                token = str(key).replace("~", "~0").replace("/", "~1")
                emit(module, child, f"{path}/{token}", unit, status)
        elif isinstance(value, (list, tuple)) and value:
            for index, child in enumerate(value):
                emit(module, child, f"{path}/{index}", unit, status)
        else:
            writer.writerow((module, path or "/",
                             json.dumps(value, ensure_ascii=False), unit, status))

    for key in dict.fromkeys((*payload, "materials", "sections", "nodes", "members",
                             "supports", "load_cases", "combos", "steps", "amplitudes",
                             "sets")):
        emit(f"model.{key}", payload.get(key, []), unit=payload.get("units", "N-m-Pa"))
    # 表格草稿尚未应用时只能导出草稿输入，不能将旧模型结果混入。
    valid = same_model and session.result_error() is None
    if valid:
        emit("results", session.result_db.to_dict(), status="COMPLETED")
        if session.solve_summary:
            emit("solve_summary", session.solve_summary.payload, status="COMPLETED")
    else:
        emit("results", "模型已修改或尚未求解；请应用模型并求解。", status="NOT_AVAILABLE")
    key = session._model_key()
    available = {name: data for name, (fingerprint, data) in (analyses or {}).items()
                 if same_model and fingerprint == key}
    for name in dict.fromkeys((*ANALYSIS_MODULES, *available)):
        if name in available:
            emit(f"analysis.{name}", available[name], status="COMPLETED")
        else:
            emit(f"analysis.{name}", "当前模型尚无该模块的已完成结果。", status="NOT_AVAILABLE")
    if solid is not None and same_model:
        emit("solid.points", solid.points, unit="mm", status="COMPLETED")
        emit("solid.cells", solid.cells, status="COMPLETED")
        for association in ("point", "cell"):
            for name, values in getattr(solid, f"{association}_data").items():
                emit(f"solid.{association}.{name}", values, status="COMPLETED")
    else:
        emit("solid", "当前未打开有效实体结果。", status="NOT_AVAILABLE")
    return "\ufeff" + buf.getvalue()
