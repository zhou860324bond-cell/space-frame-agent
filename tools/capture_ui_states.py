"""按固定示例模型拍一组界面状态图，用于界面改动前后的对比。

需要 Windows 图形会话和 OpenGL。用法：

    python tools/capture_ui_states.py results/ui_polish/before

每个状态在 1440×900 与 1366×768 两种窗口尺寸下各拍一张；
相机统一走生产「适应窗口」入口，不手动调整，便于暴露真实的适应窗口行为。
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT, ROOT / "src", ROOT / "tools"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from render_smoke import build_session  # noqa: E402

SIZES = ((1440, 900), (1366, 768))


def main(output: Path) -> None:
    from PySide6.QtWidgets import QApplication

    from desktop import app as desktop_app
    from desktop import qt_style, scene, theme
    from desktop.main_window import MainWindow

    output.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    desktop_app._normalise_application_font(app)
    desktop_app._load_cjk_font(app)
    qt_style.install(app)
    app.setStyleSheet(theme.STYLESHEET)

    def settle() -> None:
        for _ in range(8):
            app.processEvents()

    with tempfile.TemporaryDirectory(prefix="framelab-ui-") as scratch:
        os.environ["FRAMELAB_AUTOSAVE_DIR"] = str(Path(scratch) / "autosave")
        os.environ["FRAMELAB_SETTINGS_FILE"] = str(Path(scratch) / "settings.ini")

        for width, height in SIZES:
            tag = f"{width}x{height}"

            empty = MainWindow()
            empty.resize(width, height)
            empty.move(0, 0)
            empty.show()
            settle()
            empty.export_screenshot(str(output / f"{tag}-01-空模型.png"))
            empty.close()

            window = MainWindow(build_session())
            window.resize(width, height)
            window.move(0, 0)
            window.show()
            settle()
            window.view_fit()
            settle()
            window.export_screenshot(str(output / f"{tag}-02-模型.png"))

            window.solve()
            window.runner.wait(60000)
            settle()
            if window.result is None or not window.result.ok:
                raise RuntimeError("示例模型求解失败，不能生成结果截图")
            window.ribbon.set_collapsed(True)

            states = (
                ("03-变形", "变形", None),
                ("04-弯矩云图", "云图", "M"),
                ("05-应力云图", "云图", scene.STRESS),
                ("06-内力图", "内力图", "M"),
                ("07-应力比", "应力比", None),
            )
            for name, mode, component in states:
                if component is not None:
                    window.component = component
                window.set_mode(mode)
                window.runner.wait(60000)         # 应力比在后台算
                settle()
                window.view_fit()
                settle()
                window.export_screenshot(str(output / f"{tag}-{name}.png"))
            window.close()
            settle()


if __name__ == "__main__":
    os.environ.pop("QT_QPA_PLATFORM", None)
    os.environ.setdefault("QT_SCALE_FACTOR", "1")
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "ui_states"
    main(target)
