"""后台任务在增量、样本和网格阶段之间协作停止，不强制终止线程。"""

from __future__ import annotations

from contextlib import contextmanager
from threading import Event, local


class TaskCancelled(Exception):
    """当前任务已在安全检查点停止。"""


_state = local()


@contextmanager
def task_scope(cancel_event: Event):
    """取消标志只属于当前线程，避免影响其他窗口或普通脚本。"""
    previous = getattr(_state, "cancel_event", None)
    _state.cancel_event = cancel_event
    try:
        yield
    finally:
        _state.cancel_event = previous


def checkpoint() -> None:
    """矩阵分解等不可中断步骤完成后，再在这里响应停止请求。"""
    event = getattr(_state, "cancel_event", None)
    if event is not None and event.is_set():
        raise TaskCancelled("任务已停止；未完成的计算结果未提交。")
