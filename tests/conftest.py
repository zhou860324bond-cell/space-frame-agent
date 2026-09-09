"""桌面端测试的公共收尾。

每个用例都要造一个 `MainWindow`，而每个窗口都持有一个**活的 VTK 渲染窗口**。
不关就一直攒着，攒到一定数量，进程退出时会崩在 VTK 的析构里——
表现为整批测试跑完之后一个 `Fatal Python error: Aborted`，
而每个用例单独跑都是绿的。这种"只在一起跑才崩"的现象最容易被误判成偶发。

顺带一提：这**不只是测试的问题**。真机上点关闭按钮走的是同一条路，
所以 `MainWindow.closeEvent` 里也做了同样的收尾。
"""

from __future__ import annotations

import pytest


def opengl_available() -> bool:
    """检测本机是否有可用的 OpenGL 渲染环境。

    远程桌面、虚拟机、没装显卡驱动的机器上，VTK/PyVista 依赖全齐也画不出
    东西——创建渲染窗口时会抛 'failed to get valid pixel format' 或
    'Failed to initialize OpenGL functions!'。这种环境下涉及真实渲染的
    测试应该 skip，而不是让进程崩溃。

    检测方法：尝试创建一个最小的 VTK 渲染窗口并渲染。任何异常都算不可用。
    """
    try:
        import vtk
        ren_win = vtk.vtkRenderWindow()
        ren_win.SetOffScreenRendering(1)
        ren_win.SetSize(16, 16)
        renderer = vtk.vtkRenderer()
        ren_win.AddRenderer(renderer)
        # 真正调一次 Render 才会触发 OpenGL 上下文初始化；只建窗口不渲染
        # 在某些环境下会"假成功"，doctor 自检误报"离屏正常"就是这个原因。
        ren_win.Render()
        ren_win.Finalize()
        return True
    except Exception:
        return False


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
