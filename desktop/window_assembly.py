"""主窗口的 Assembly 命令：创建实例、线性阵列、环形阵列、实例管理器。

流程与 Abaqus 相同的部分：先选部件（在视口选杆件或用集合），再填平移 / 旋转或阵列参数。
副本是独立实例，语义见 src/session_assembly.py。
"""

from __future__ import annotations

from PySide6.QtWidgets import QDialog, QMessageBox

from .assembly_dialogs import InstanceDialog, InstanceManagerDialog
from .object_dialogs import next_name


class WindowAssemblyMixin:
    """Assembly 命令。见模块 docstring。"""

    def _instance_names(self) -> set[str]:
        records = (self.session.model.get("assembly") or {}).get("instances") or []
        return {r["name"] for r in records} | {r.get("pattern") for r in records if r.get("pattern")}

    def _start_instance(self, mode: str, label: str) -> None:
        if not self.session.model.get("members"):
            self.statusBar().showMessage("还没有杆件：请先建立要复制的部件。", 5000)
            return
        self.region_pick.start("member", f"{label}：选部件杆件",
                               lambda region: self._finish_instance(mode, region),
                               self._initial_region("member"))

    def create_instance_ui(self) -> None:
        self._start_instance("instance", "创建实例")

    def linear_pattern_ui(self) -> None:
        self._start_instance("linear", "线性阵列")

    def radial_pattern_ui(self) -> None:
        self._start_instance("radial", "环形阵列")

    def _finish_instance(self, mode: str, region: dict) -> None:
        text = (f"集合「{region['set']}」" if "set" in region
                else f"{len(region.get('members') or [])} 根杆件")
        prefix = {"instance": "Instance", "linear": "Pattern", "radial": "Pattern"}[mode]
        dialog = InstanceDialog(mode, next_name(prefix, self._instance_names()), text, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        values = dialog.values()
        tool = {"instance": self.session.create_instance, "linear": self.session.linear_pattern,
                "radial": self.session.radial_pattern}[mode]
        result = tool(region=region, **values)
        done = f"已创建「{values['name']}」：{len(result.payload.get('members') or [])} 根杆件。"
        if self._object_result(result, done):
            skipped = result.payload.get("skipped_overlapping")
            note = "边界条件与荷载不随实例复制；新节点上的支座请另加。"
            if skipped:
                note = f"{sum(len(v) for v in skipped.values())} 根与已有杆件重合，已跳过。" + note
            self.statusBar().showMessage(done + note, 12000)

    def open_instance_manager(self) -> None:
        InstanceManagerDialog(self).exec()
        self.refresh()

    def delete_instance_ui(self, name: str) -> None:
        if QMessageBox.question(self, "删除实例", f"删除「{name}」？它新建的杆件与节点会一并撤掉。"
                                ) != QMessageBox.StandardButton.Yes:
            return
        self._object_result(self.session.delete_instance(name), f"已删除「{name}」。")
