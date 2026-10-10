"""用真实计时记录模型树重建、窗口刷新和视口重画，输出可复核的 JSON。"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--repeat", type=int, default=12)
    args = parser.parse_args()
    if args.repeat < 2:
        parser.error("repeat 必须至少为 2")
    if args.headless:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6 import __version__ as qt_version
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QApplication
    from desktop.main_window import MainWindow
    from tools.render_smoke import build_session

    with tempfile.TemporaryDirectory() as directory:
        # 性能测量不读写用户的自动保存和界面设置。
        os.environ["FRAMELAB_AUTOSAVE_DIR"] = str(Path(directory) / "autosave")
        os.environ["FRAMELAB_SETTINGS_FILE"] = str(Path(directory) / "desktop.ini")
        import capsule
        capsule.DEFAULT_CAPSULE_DIR = Path(directory) / "capsules"
        app = QApplication.instance() or QApplication([])
        report = {"python": platform.python_version(), "platform": platform.platform(),
                  "pyside": qt_version, "render": not args.headless,
                  "repeat": args.repeat, "scenarios": []}
        for large in (False, True):
            session = build_session()
            if large:
                session.generate_frame(spans=[6.0] * 8, storeys=[3.6] * 5, bays=[6.0] * 5,
                                       beam_load=3e4, column_section="COL",
                                       beam_section="BEAM", material="Q355")
            window = MainWindow(session)
            window.show()
            app.processEvents()
            scenario = {"nodes": len(session.model["nodes"]),
                        "members": len(session.model["members"]), "timings_ms": {}}
            counter = 0

            def search():
                nonlocal counter
                counter += 1
                window.tree_panel.search.setText(f"节点 {counter % len(session.model['nodes']) + 1}")

            def locate():
                nonlocal counter
                counter += 1
                window._on_tree_action("locate_object", ("node", counter % len(session.model['nodes']) + 1))

            for name, action in (("tree_rebuild", lambda: window.tree.rebuild(session)),
                                 ("refresh", window.refresh), ("redraw", window.redraw),
                                 ("tree_search", search), ("object_location", locate)):
                action()
                values = []
                for _ in range(args.repeat):
                    started = time.perf_counter()
                    action()
                    app.processEvents()
                    values.append((time.perf_counter() - started) * 1000)
                ordered = sorted(values)
                scenario["timings_ms"][name] = {
                    "median": statistics.median(values),
                    "p95": ordered[math.ceil(.95 * len(ordered)) - 1],
                    "samples": values}
            report["scenarios"].append(scenario)
            window.close()
            window.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            app.processEvents()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
