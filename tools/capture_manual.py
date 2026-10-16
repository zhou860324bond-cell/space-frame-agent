"""用固定示例模型重拍使用手册的六张真实界面图。

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
        window.ribbon.set_collapsed(False)
        window.set_mode("模型")
        settle()
        capture(window, "02-结果页功能区.png")
        window.ribbon.set_collapsed(True)

        window.run_strength_check()
        window.runner.wait(60000)
        window.results_dock.show()
        result_area = window.results_dock._container
        result_area.verticalScrollBar().setValue(0)
        window.view_fit()
        settle()
        # 明细默认独立显示，手册图直接取同一个真实结果窗口。
        window.results_dock._window.grab().save(str(output / "03-强度验算结果.png"))

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

        # 06：离线演示的一轮真实对话与模型并排。离线演示的意图为预设脚本，
        # 工具调用与求解真实执行；不使用任何在线密钥。
        os.environ["FRAMELAB_SETTINGS_FILE"] = str(Path(scratch) / "chat-settings.ini")
        chat_window = MainWindow()
        chat_window.resize(1440, 900)
        chat_window.move(0, 0)
        chat_window.show()
        chat_window.chat_dock.show()
        chat = chat_window.chat
        chat.model_selector.setCurrentIndex(chat.model_selector.findData("offline"))
        settle()
        chat.send("单跨 24 米门式刚架，檐口高 7.5 米，柱脚铰接，Q355，屋面恒载 8 kN/m，建模求解并看看挠跨比。")
        chat_window.runner.wait(60000)
        settle()
        chat_window.view_fit()
        chat.view.scroll_to_bottom()
        capture(chat_window, "06-Agent工作台.png")
        chat_window.close()


if __name__ == "__main__":
    os.environ.pop("QT_QPA_PLATFORM", None)
    os.environ.setdefault("QT_SCALE_FACTOR", "1")
    main()
