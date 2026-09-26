"""用固定示例模型重拍使用手册的五张真实界面图。

需要 Windows 图形会话和 OpenGL；只生成文档截图，不执行演示验收。
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT, ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from render_smoke import build_session  # noqa: E402

# 手册图在 Word 里按页宽排版，1600 px 足够清楚；再大只会让 docx 膨胀
MAX_WIDTH = 1600


def shrink(path: Path) -> None:
    """缩到 MAX_WIDTH 宽并转 256 色调色板 PNG。

    高分屏整窗截图是 2500+ px 真彩色，每张 600–840 KB；压完 230–310 KB，
    文字仍清楚。`tests/test_docs_current.py` 守着体积上限。
    """
    from PIL import Image

    with Image.open(path) as image:
        picture = image.convert("RGB")
    if picture.width > MAX_WIDTH:
        height = round(picture.height * MAX_WIDTH / picture.width)
        picture = picture.resize((MAX_WIDTH, height), Image.Resampling.LANCZOS)
    picture.quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(
        path, "PNG", optimize=True)


def main() -> None:
    from PySide6.QtWidgets import QApplication

    from desktop import app as desktop_app
    from desktop import qt_style, scene, theme
    from desktop.main_window import MainWindow

    output = ROOT / "docs" / "手册图"
    output.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    desktop_app._normalise_application_font(app)
    desktop_app._load_cjk_font(app)
    qt_style.install(app)
    app.setStyleSheet(theme.STYLESHEET)

    def settle() -> None:
        for _ in range(5):
            app.processEvents()

    def capture(window, name: str) -> None:
        """抓真实窗口，包含 VTK 原生子窗口和覆盖其上的 Qt 抽屉。"""
        settle()
        print(f"{name}: 窗口 {window.width()}×{window.height()}，"
              f"结果抽屉 {window.bottom_drawer.is_open()}")
        window.export_screenshot(str(output / name))
        shrink(output / name)

    with tempfile.TemporaryDirectory(prefix="framelab-manual-") as scratch:
        os.environ["FRAMELAB_AUTOSAVE_DIR"] = str(Path(scratch) / "autosave")
        os.environ["FRAMELAB_SETTINGS_FILE"] = str(Path(scratch) / "settings.ini")

        empty = MainWindow()
        empty.resize(1440, 900)
        empty.move(0, 0)
        empty.show()
        settle()
        capture(empty, "01-启动界面.png")
        empty.close()

        window = MainWindow(build_session())
        window.resize(1440, 900)
        window.move(0, 0)
        window.show()
        settle()
        window.solve()
        window.runner.wait(60000)
        settle()
        if window.result is None or not window.result.ok:
            raise RuntimeError("示例模型求解失败，不能生成结果截图")
        window.view_fit()

        window._show_ribbon_page("结果")
        window.set_mode("模型")
        settle()
        capture(window, "02-结果页功能区.png")

        window.run_strength_check()
        window.runner.wait(60000)
        window.results_dock.show()
        window.bottom_drawer.set_extent(600)
        result_area = window.bottom_drawer.stack.currentWidget()
        result_area.verticalScrollBar().setValue(180)
        settle()
        capture(window, "03-强度验算结果.png")

        window.results_dock.hide()
        window.component = "M"
        window.set_mode("云图")
        window.view_fit()
        settle()
        capture(window, "04-弯矩云图.png")

        window.component = scene.STRESS
        window.set_mode("云图")
        window.view_fit()
        settle()
        capture(window, "05-应力云图.png")
        window.close()


if __name__ == "__main__":
    os.environ.pop("QT_QPA_PLATFORM", None)
    os.environ.setdefault("QT_SCALE_FACTOR", "1")
    main()
