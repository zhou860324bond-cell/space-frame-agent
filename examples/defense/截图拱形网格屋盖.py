"""从软件真实主窗口导出拱形屋盖截图，包含实际后台求解与初设对比。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def capture(output: Path) -> None:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    private = ROOT / "results/defense_vault_ui"
    private.mkdir(parents=True, exist_ok=True)
    os.environ["FRAMELAB_AUTOSAVE_DIR"] = str(private / "autosave")
    os.environ["FRAMELAB_SETTINGS_FILE"] = str(private / "desktop.ini")
    os.environ["QT_QPA_PLATFORM"] = "windows"
    os.environ["QT_SCALE_FACTOR"] = "1.25"
    import capsule
    capsule.DEFAULT_CAPSULE_DIR = private / "capsules"
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QCoreApplication, QEvent, Qt
    from PySide6.QtTest import QTest
    from agent import Session
    from desktop import app as desktop_app, qt_style, theme, scene
    from desktop.main_window import MainWindow

    app = QApplication([])
    desktop_app._normalise_application_font(app)
    desktop_app._load_cjk_font(app)
    qt_style.install(app)
    app.setStyleSheet(theme.STYLESHEET)
    window = MainWindow(Session())
    checks, screenshots = [], []

    def check(name, passed):
        checks.append({"name": name, "ok": bool(passed)})
        assert passed, name
        print(name, flush=True)

    def settle():
        for _ in range(4):
            app.processEvents()

    def shot(name, view="iso"):
        window.left_drawer.close()
        window.bottom_drawer.close()
        QTest.keyClick(window, Qt.Key_Escape)
        if view == "front":
            window.view_front()
        elif view == "top":
            window.view_top()
        else:
            window.view_iso()
        window.viewport.plotter.camera.zoom(.78)
        settle()
        path = output / (name + ".png")
        window.export_screenshot(str(path))
        check("截图 " + name, path.is_file())
        screenshots.append({"file": path.name, "mode": window.mode, "case": window.case,
                            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "source": "MainWindow.export_screenshot：真实 Qt 窗口和 VTK 渲染视口"})

    try:
        dpr = window.devicePixelRatioF()
        window.resize(round(1920 / dpr), round(1080 / dpr))
        window.show()
        settle()
        window.ribbon.set_collapsed(True)
        model_path = ROOT / "examples/defense/拱形网格屋盖.json"
        check("打开调整版模型", window._load_model_path(model_path))
        check("239 节点与 622 杆件", len(window.session.model["nodes"]) == 239 and len(window.session.model["members"]) == 622)
        check("18 个固定柱脚", len(window.session.model["supports"]) == 18)
        check("模型校验", window.session.validate_model().ok)
        window.set_mode("模型")
        window.viewport.show_model(window.session.preview_frame(), case=None, loads=False)
        shot("01_整体模型")
        shot("02_拱形正视", "front")
        shot("03_网格俯视", "top")
        window.solve()
        check("真实后台求解完成", window.runner.wait(60000))
        settle()
        check("十个工况及组合", len(window.session.solution.all_results()) == 10)
        for i in range(window.ribbon.count()):
            if window.ribbon.tabText(i) == "结果":
                window.ribbon.setCurrentIndex(i)
                break
        window.quickbar.cases.setCurrentText("ULS-GW")
        window.set_mode("变形")
        shot("04_调整版变形")
        window._set_component("N")
        shot("05_轴力云图")
        window._set_component("M")
        shot("06_弯矩云图")
        window._set_component(scene.STRESS)
        shot("07_正应力云图")
        window.show_utilization(interactive=False)
        check("调整版应力比任务完成", window.runner.wait(60000))
        settle()
        check("调整版验算项通过", window._utilization[1]["ok"])
        shot("08_调整版应力比")
        window.quickbar.cases.setCurrentText("ULS-ASYM")
        window.set_mode("变形")
        shot("09_半跨不对称荷载变形")
        model_path = ROOT / "examples/defense/拱形网格屋盖_初设.json"
        check("打开初设模型", window._load_model_path(model_path))
        window.solve()
        check("初设真实后台求解", window.runner.wait(60000))
        settle()
        window.quickbar.cases.setCurrentText("ULS-GW")
        window.show_utilization(interactive=False)
        check("初设应力比任务完成", window.runner.wait(60000))
        settle()
        check("初设三根超限可复现", window._utilization[1]["failed_members"] == [73, 97, 121])
        shot("10_初设三根杆件超限")
    finally:
        (private / "验收.json").write_text(json.dumps({"checks": checks, "screenshots": screenshots},
                                                    ensure_ascii=False, indent=2), encoding="utf-8")
        window.close()
        window.deleteLater()
        app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "examples/defense/拱形网格屋盖截图")
    capture(parser.parse_args().output)
