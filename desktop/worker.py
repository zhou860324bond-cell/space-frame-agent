"""把耗时的活儿挪出界面线程。

Qt 的界面事件循环在主线程。**任何在主线程里跑几秒的东西都会让窗口变成
"未响应"**——用户看到的是程序死了，而不是程序在算。

这里有两类活儿必须挪走：

* **求解**——几百个自由度是毫秒级，但网架、扫参、模态一上来就是秒级；
* **大模型调用**——网络往返，几秒起步，而且可能超时。

做法是最朴素的一种：一个 `QThread` 跑一个可调用对象，跑完发信号。
不用线程池，因为这两类活儿都是"用户点一下、跑一件"，并发度天然是 1，
而且**必须是 1**：两个求解同时改同一个 Session 会把模型改烂。
`Runner.busy` 就是这道闸。

---

**连接类型这件事，是这个模块唯一真正难的地方，也踩过一次。**

Qt 的 `AutoConnection` 按**接收者的线程归属**决定是直接调用还是投递事件。
而 lambda 不是 QObject，判断不出归属，于是退化成直接连接——
信号发出的那一刻，槽就在**工作线程里**跑起来了。后果有两种，都很难查：

* 收尾函数在工作线程里执行 `thread.wait()`，**线程在等自己**，直接 abort；
* 业务回调在工作线程里碰控件，撞上主线程的渲染，随机崩。

所以这里的规矩是：**所有跨线程的连接，接收者必须是主线程里的 QObject，
并且显式写 QueuedConnection。** 用户传进来的回调不直接连信号，
而是存下来，由 Runner 自己的槽在主线程里调。
"""

from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import QObject, Qt, QThread, Signal


class _Job(QObject):
    """在工作线程里跑一个可调用对象。"""

    done = Signal(object)                 # 正常结束，带返回值
    failed = Signal(str, str)             # 异常：(类型名, 消息)
    progress = Signal(str, object, bool)  # 中途进度：(名称, 参数, 是否成功)

    def __init__(self, fn: Callable[..., Any], wants_report: bool):
        super().__init__()
        self._fn = fn
        self._wants_report = wants_report

    def run(self) -> None:
        try:
            # 有 `report` 形参的可调用对象，就把进度出口交给它。
            # **进度必须走信号，不能让工作线程直接回调界面**——
            # 那就是跨线程碰控件，之前已经因此崩过一次
            result = (self._fn(report=self.progress.emit) if self._wants_report
                      else self._fn())
        except Exception as exc:          # noqa: BLE001
            # 工作线程里的异常不会自动冒到主线程。不抓住的话，
            # 界面就只是"永远转圈"，比报错更难查
            self.failed.emit(type(exc).__name__, str(exc))
            return
        self.done.emit(result)


class Runner(QObject):
    """一次只跑一件事的后台执行器。

    用法：

        runner.submit(session.solve_model,
                      on_done=self._solved, on_failed=self._oops)

    回调**保证在主线程里执行**，所以里面可以放心碰控件。

    `busy` 为真时再提交会被拒绝并返回 False——调用方据此把按钮置灰，
    而不是把请求排队。排队在这里是错的：用户连点两次求解，
    他要的是"算一次"，不是"算两次"。
    """

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._thread: QThread | None = None
        self._job: _Job | None = None
        self._on_done: Callable[[Any], None] | None = None
        self._on_failed: Callable[[str, str], None] | None = None
        self._on_progress: Callable[[str, Any, bool], None] | None = None

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def submit(self, fn: Callable[..., Any],
               on_done: Callable[[Any], None] | None = None,
               on_failed: Callable[[str, str], None] | None = None,
               on_progress: Callable[[str, Any, bool], None] | None = None
               ) -> bool:
        """提交一件活儿。

        `on_progress` 给了的话，`fn` 会收到一个 `report` 关键字参数——
        它是一个可以在工作线程里安全调用的出口，内部走信号投递，
        **所以 `on_progress` 本身仍然在主线程里执行**，可以放心碰控件。
        """
        if self.busy:
            return False

        thread = QThread()
        job = _Job(fn, wants_report=on_progress is not None)
        job.moveToThread(thread)
        thread.started.connect(job.run)

        # 接收者是 self —— 一个主线程里的 QObject，再显式指定 QueuedConnection。
        # 两者缺一不可：少了接收者上下文，Qt 判断不出线程归属；
        # 少了 QueuedConnection，跨线程时行为依赖 Qt 的推断
        job.done.connect(self._handle_done, Qt.ConnectionType.QueuedConnection)
        job.failed.connect(self._handle_failed, Qt.ConnectionType.QueuedConnection)
        job.progress.connect(self._handle_progress,
                             Qt.ConnectionType.QueuedConnection)

        self._thread, self._job = thread, job
        self._on_done, self._on_failed = on_done, on_failed
        self._on_progress = on_progress
        thread.start()
        return True

    # --- 以下几个槽都在主线程里执行 ---

    def _handle_progress(self, name: str, args: Any, ok: bool) -> None:
        if self._on_progress is not None:
            self._on_progress(name, args, ok)

    def _handle_done(self, result: Any) -> None:
        callback = self._on_done
        self._finish()                    # 先收尾：回调里可能接着提交下一件
        if callback is not None:
            callback(result)

    def _handle_failed(self, kind: str, message: str) -> None:
        callback = self._on_failed
        self._finish()
        if callback is not None:
            callback(kind, message)

    def _finish(self) -> None:
        thread, job = self._thread, self._job
        self._thread = self._job = None
        self._on_done = self._on_failed = self._on_progress = None
        if thread is not None:
            thread.quit()
            thread.wait(5000)             # 主线程等工作线程，方向是对的
        if job is not None:
            job.deleteLater()

    def wait(self, msec: int = 30000) -> bool:
        """等当前这件事跑完并且回调也走完。**测试里用，界面流程不该调。**

        必须转事件循环：跨线程的信号是投递到主线程事件队列的，
        不转就永远收不到，`busy` 也就永远为真。
        """
        from PySide6.QtCore import QCoreApplication, QDeadlineTimer

        deadline = QDeadlineTimer(msec)
        while self.busy and not deadline.hasExpired():
            QCoreApplication.processEvents()
            thread = self._thread
            if thread is not None:
                thread.wait(10)
        QCoreApplication.processEvents()
        return not self.busy
