"""桌面端测试的公共收尾。

每个用例都要造一个 `MainWindow`，而每个窗口都持有一个**活的 VTK 渲染窗口**。
不关就一直攒着，攒到一定数量，进程退出时会崩在 VTK 的析构里——
表现为整批测试跑完之后一个 `Fatal Python error: Aborted`，
而每个用例单独跑都是绿的。这种"只在一起跑才崩"的现象最容易被误判成偶发。

顺带一提：这**不只是测试的问题**。真机上点关闭按钮走的是同一条路，
所以 `MainWindow.closeEvent` 里也做了同样的收尾。
"""

from __future__ import annotations

import functools
import subprocess
import sys

import pytest


# 探测脚本单独拎出来，是因为它必须在**子进程**里跑，见下。
_PROBE = """
import vtk
w = vtk.vtkRenderWindow()
w.SetOffScreenRendering(1)
w.SetSize(16, 16)
w.AddRenderer(vtk.vtkRenderer())
w.Render()
w.Finalize()
"""


@functools.lru_cache(maxsize=1)
def opengl_available() -> bool:
    """检测本机是否有可用的 OpenGL 渲染环境。

    远程桌面、虚拟机、没装显卡驱动的机器上，VTK/PyVista 依赖全齐也画不出
    东西。这种环境下涉及真实渲染的测试应该 skip，而不是让进程崩溃。

    **必须在子进程里探测。** 原先是在本进程里 try/except 包一个 vtkRenderWindow
    再 Render()，想法没错，但兜不住：在没有 GPU 的机器上 Render() 不抛 Python
    异常，而是直接原生崩溃——`except Exception` 对 access violation 无能为力。
    GitHub 的 windows runner 上就是这样，整个 pytest 在收集阶段被带走：

        Windows fatal exception: access violation
        Current thread ... tests/conftest.py in opengl_available
        Segmentation fault    pytest -q     → exit code 139

    子进程崩了只是父进程读到一个非零返回码，正好就是"不可用"。这也是探测
    任何"可能把进程带走"的东西时唯一靠得住的做法。

    结果缓存一次：这个函数在好几个测试模块的 import 期被调用，每次开一个
    子进程太慢。
    """
    try:
        done = subprocess.run([sys.executable, "-c", _PROBE],
                              capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


@pytest.fixture(autouse=True)
def _close_windows():
    """每个用例跑完，关掉它开出来的所有顶层窗口。"""
    yield
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        return                                  # 没装 Qt 的环境不需要收尾
    app = QApplication.instance()
    if app is None:
        return
    for widget in list(app.topLevelWidgets()):
        if widget.__class__.__name__ == "MainWindow":
            widget.close()
            widget.deleteLater()
    app.processEvents()
