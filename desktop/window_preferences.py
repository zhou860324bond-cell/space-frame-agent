"""主窗口的持久化设置；测试窗口不写用户设置。"""

from __future__ import annotations

import json
import os

from PySide6.QtCore import QSettings

from . import scene


def _settings() -> QSettings:
    """设置入口；测试通过 FRAMELAB_SETTINGS_FILE 隔离真实设置。"""
    path = os.environ.get("FRAMELAB_SETTINGS_FILE")
    if path:
        return QSettings(path, QSettings.Format.IniFormat)
    return QSettings("SpaceFrameAgent", "Desktop")


class WindowPreferencesMixin:
    # 只有程序入口恢复过设置的窗口才写回；测试窗口不能覆盖用户设置。
    _persist_preferences = False

    def restore_preferences(self) -> None:
        """恢复窗口、抽屉、云图选项与内力分量。"""
        self._persist_preferences = True
        settings = _settings()
        geometry = settings.value("window/geometry")
        if geometry:
            self.restoreGeometry(geometry)
        for side, drawer in self.drawers.drawers.items():
            extent = settings.value(f"drawers/{side}")
            try:
                if extent is not None:
                    drawer.extent = int(extent)
            except (TypeError, ValueError):
                pass
        try:
            options = json.loads(settings.value("results/display", "{}") or "{}")
        except (TypeError, ValueError):
            options = {}
        if isinstance(options, dict) and options:
            self.results.apply_options(options)
        component = settings.value("results/component")
        if component in {"M", "V", "N", "Vy", "Vz", "T", "My", "Mz", scene.STRESS}:
            self.component = component
        self.drawers.relayout()

    def _save_preferences(self) -> None:
        if not self._persist_preferences:
            return
        settings = _settings()
        settings.setValue("window/geometry", self.saveGeometry())
        for side, drawer in self.drawers.drawers.items():
            settings.setValue(f"drawers/{side}", int(drawer.extent))
        settings.setValue("results/display",
                          json.dumps(self.results.display_options()))
        settings.setValue("results/component", self.component)
