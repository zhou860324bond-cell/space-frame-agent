"""真实后台任务收尾：工作对象销毁线程、待投递结果与线程资源释放。"""

from threading import Event, get_ident
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QEvent, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication              # noqa: E402
from shiboken6 import isValid                           # noqa: E402

from desktop.worker import Runner                      # noqa: E402
from task_control import checkpoint                    # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize("outcome", ["success", "failure", "cancelled"])
def test_completed_job_is_destroyed_in_its_worker_thread(app, outcome):
    """防止工作对象在线程停止后才延迟删除，滞留到后台垃圾回收造成原生退出。"""
    release = Event()
    entered = Event()
    worker_ident = []
    destroyed_on = []
    results, failures = [], []

    def work():
        worker_ident.append(get_ident())
        entered.set()
        assert release.wait(5)
        checkpoint()
        if outcome == "failure":
            raise ValueError("验证失败路径的资源释放")
        return 42

    runner = Runner()
    assert runner.submit(work, on_done=results.append,
                         on_failed=lambda kind, message: failures.append(kind))
    thread, job = runner._thread, runner._job
    job.destroyed.connect(lambda: destroyed_on.append(get_ident()),
                          Qt.ConnectionType.DirectConnection)
    try:
        assert entered.wait(5)
        if outcome == "cancelled":
            assert runner.cancel()
    finally:
        release.set()
        assert runner.wait(10000)
    assert not isValid(job)
    assert destroyed_on == worker_ident and worker_ident != [get_ident()]
    assert results == ([42] if outcome == "success" else [])
    assert failures == ([] if outcome == "success" else [
        "ValueError" if outcome == "failure" else "TaskCancelled"])
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not isValid(thread)


def test_stopped_thread_keeps_runner_busy_until_result_is_delivered(app):
    """防止线程先停止但结果仍在事件队列时，新任务覆盖旧结果回调和资源引用。"""
    release = Event()
    results = []
    runner = Runner()
    assert runner.submit(lambda: release.wait(5) and 42, on_done=results.append)
    thread, job = runner._thread, runner._job
    job.done.connect(thread.quit, Qt.ConnectionType.DirectConnection)
    try:
        release.set()
        assert thread.wait(5000)
        assert runner.busy
        assert not runner.submit(lambda: 99)
    finally:
        assert runner.wait(10000)
    assert results == [42]


def test_window_cleanup_does_not_destroy_a_running_child_thread():
    """防止应力比异步任务未结束时，测试收尾忽略拒绝关闭并强制销毁运行中的线程。"""
    script = textwrap.dedent('''
        import sys
        from threading import Event
        sys.path[:0] = [".", "src", "tests"]
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication
        from shiboken6 import isValid
        from conftest import _close_windows
        from desktop.main_window import MainWindow
        from task_control import checkpoint

        app = QApplication([])
        window = MainWindow()
        window.show()
        entered, release = Event(), Event()
        def work():
            entered.set()
            assert release.wait(5)
            checkpoint()
        assert window.runner.submit(work)
        job = window.runner._job
        assert entered.wait(5)
        QTimer.singleShot(30, release.set)
        cleanup = _close_windows.__wrapped__()
        next(cleanup)
        try:
            next(cleanup)
        except StopIteration:
            pass
        assert not isValid(window) and not isValid(job)
    ''')
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONUTF8="1")
    got = subprocess.run([sys.executable, "-c", script],
                         cwd=Path(__file__).resolve().parents[1], env=env,
                         capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert got.returncode == 0, got.stdout + got.stderr


def test_closing_window_cancels_deferred_utilization_check():
    """防止切到应力比后立即关闭，零延迟回调在已关闭窗口上启动新线程并原生退出。"""
    script = textwrap.dedent('''
        import sys
        sys.path[:0] = [".", "src", "tests"]
        from PySide6.QtCore import QCoreApplication, QEvent
        from PySide6.QtWidgets import QApplication
        from shiboken6 import isValid
        from test_desktop_window import built, solved
        from desktop.main_window import MainWindow

        app = QApplication([])
        window = solved(MainWindow(built()))
        submissions = []
        submit = window.runner.submit
        def record(*args, **kwargs):
            submissions.append(True)
            return submit(*args, **kwargs)
        window.runner.submit = record
        window.set_mode("应力比")
        assert window.close()
        app.processEvents()
        assert submissions == [], "关闭后仍启动了待执行的应力比验算"
        assert not window.runner.busy
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        assert not isValid(window)
    ''')
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONUTF8="1")
    got = subprocess.run([sys.executable, "-c", script],
                         cwd=Path(__file__).resolve().parents[1], env=env,
                         capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert got.returncode == 0, got.stdout + got.stderr
