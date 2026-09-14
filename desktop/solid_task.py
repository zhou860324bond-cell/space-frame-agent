"""节点实体双后端任务的纯编排；不碰 Qt，也不直接调用求解器。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def run_backend_pair(
        run_backend: Callable[[str, list[float]], Any],
        plans: dict[str, list[float]], *,
        cached: dict[str, dict] | None = None,
        is_cancelled: Callable[[], bool] | None = None,
        report: Callable[[str, dict, bool], None] | None = None) -> dict:
    """依次运行 native/Abaqus；失败或取消时返回已经完成的阶段。"""
    completed = dict(cached or {})
    cancelled = is_cancelled or (lambda: False)
    notify = report or (lambda _name, _data, _ok: None)
    for index, backend in enumerate(("native", "abaqus"), start=1):
        if backend in completed:
            notify("cached", {"backend": backend, "stage": index, "total": 2}, True)
            continue
        if cancelled():
            return {"ok": False, "cancelled": True, "completed": completed,
                    "failed_backend": None,
                    "error": "已在安全阶段边界取消，未启动后续求解器。"}
        notify("started", {"backend": backend, "stage": index, "total": 2}, True)
        try:
            result = run_backend(backend, list(plans[backend]))
        except Exception as exc:  # 后端异常也必须带回已经完成的阶段
            return {"ok": False, "cancelled": False, "completed": completed,
                    "failed_backend": backend,
                    "error": f"{type(exc).__name__}: {exc}"}
        if not bool(getattr(result, "ok", False)):
            payload = getattr(result, "payload", {}) or {}
            return {"ok": False, "cancelled": False, "completed": completed,
                    "failed_backend": backend,
                    "error": str(payload.get("error") or
                                 f"{backend} 阶段未返回可用结果")}
        completed[backend] = dict(result.payload)
        notify("completed", {"backend": backend, "stage": index, "total": 2}, True)
    return {"ok": True, "cancelled": False, "completed": completed,
            "failed_backend": None, "error": None}
