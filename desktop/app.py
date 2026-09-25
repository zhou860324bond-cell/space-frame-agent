"""桌面端入口。

    python -m desktop.app
    python -m desktop.app --doctor      # 启动不了时先跑这个

**QT_API 必须在任何 Qt 相关 import 之前钉死。** `pyvistaqt` 走 `qtpy` 选绑定，
而 `qtpy` 默认按 PyQt5 → PySide2 → PyQt6 → PySide6 的顺序探测：机器上装过
PyQt5（用 matplotlib 的人常有）它就会选 PyQt5，而我们的代码直接 import
PySide6——一个进程里两套 Qt 绑定，轻则符号冲突，重则直接闪退，
而且报错信息基本看不出根源。这是"桌面端打不开"最常见的一种原因。

无头环境（CI、截图）里设 QT_QPA_PLATFORM=offscreen 即可，不用改代码。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 这一行必须在 import 任何 Qt / pyvistaqt 之前
os.environ.setdefault("QT_API", "pyside6")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _normalise_application_font(app) -> float:
    """确保继承给所有控件的是有效的 point-size 字体。

    Windows 高 DPI/自定义系统字体有时只设置 pixelSize；这时 Qt 的
    ``pointSize()`` 合法地返回 -1。部分 Qt/第三方控件会把这个返回值再传给
    ``setPointSize``，于是控制台出现 ``Point size <= 0 (-1)``。在创建任何
    主窗口组件前转成等价的磅值，既消除警告，也避免局部控件字号异常。
    """
    font = app.font()
    point_size = float(font.pointSizeF())
    if point_size > 0:
        return point_size

    pixel_size = int(font.pixelSize())
    screen = app.primaryScreen()
    dpi = float(screen.logicalDotsPerInch()) if screen is not None else 96.0
    if pixel_size > 0 and dpi > 0:
        point_size = pixel_size * 72.0 / dpi
    else:
        point_size = 9.0
    point_size = max(1.0, point_size)
    font.setPointSizeF(point_size)
    app.setFont(font)
    return point_size


def _load_cjk_font(app) -> str:
    """Register a reliable CJK font when Qt cannot see system fallback fonts."""
    from PySide6.QtGui import QFont, QFontDatabase

    candidates = (
        ("C:/Windows/Fonts/msyh.ttc", "Microsoft YaHei UI"),
        ("C:/Windows/Fonts/Deng.ttf", "DengXian"),
        ("C:/Windows/Fonts/simhei.ttf", "SimHei"),
        ("C:/Windows/Fonts/SourceHanSansSC-Normal.otf", "Source Han Sans SC"),
    )
    family = ""
    available = set(QFontDatabase.families())
    for path, expected in candidates:
        if expected in available:
            family = expected
            break
        if not Path(path).is_file():
            continue
        font_id = QFontDatabase.addApplicationFont(path)
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            family = families[0]
            break
    if family:
        current = app.font()
        font = QFont(family)
        font.setPointSizeF(max(1.0, current.pointSizeF()))
        app.setFont(font)
    return family


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--doctor" in argv or "--check" in argv:
        from desktop.doctor import report
        return report()

    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:                      # noqa: BLE001
        # 启动失败时最没用的就是一句 ImportError。直接转去自检，把话说清楚
        print(f"无法加载 PySide6：{type(exc).__name__}: {exc}\n", file=sys.stderr)
        from desktop.doctor import report
        return report()

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("空间刚架智能计算")
    _normalise_application_font(app)
    _load_cjk_font(app)

    # 字体必须先规范化，再创建功能区、PyVistaQt 和 Matplotlib 控件；否则
    # 它们可能已经复制走 pointSize == -1 的系统字体。
    from desktop import qt_style, theme
    from desktop.main_window import MainWindow

    # 样式表管颜色边框，画不了形状——勾选框的"勾"、单选钮的圆、数字框的
    # 箭头都得自绘。**必须在 setStyleSheet 之前装**：QProxyStyle 换掉之后
    # Qt 会重新解析样式表，顺序反了的话样式表只作用在旧 style 上。
    qt_style.install(app)
    app.setStyleSheet(theme.STYLESHEET)

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
