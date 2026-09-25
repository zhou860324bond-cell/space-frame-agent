"""主窗口命令注册与分派。"""

from __future__ import annotations

import inspect
import traceback

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QShortcut
from PySide6.QtWidgets import QMessageBox

from . import commands, icons


def _takes_command(handler) -> bool:
    """处理函数是否接收 Command 参数。"""
    try:
        return bool(inspect.signature(handler).parameters)
    except (TypeError, ValueError):
        return False


class WindowCommandsMixin:
    def _build_actions(self) -> None:
        """从命令表创建菜单、功能区和快捷键共用的 QAction。"""
        self.actions_by_name: dict[str, QAction] = {}
        for cmd in commands.COMMANDS:
            act = QAction(icons.icon(cmd.icon), cmd.label, self)
            act.setObjectName(cmd.name)
            act.setToolTip(cmd.tip)
            act.setStatusTip(cmd.tip)
            if cmd.shortcut:
                act.setShortcut(cmd.shortcut)
            if cmd.checkable:
                act.setCheckable(True)
            # 点击时再取主窗口方法，避免动作表依赖窗口构造顺序。
            act.triggered.connect(lambda _=False, c=cmd: self._run_command(c))
            self.actions_by_name[cmd.name] = act
            self.addAction(act)

        sc_esc = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        sc_esc.activated.connect(self.cancel_interaction)

        self._sync_selection_actions()
        self.act_new = self.actions_by_name["new"]
        self.act_solve = self.actions_by_name["solve"]
        self.act_generate = self.actions_by_name["frame"]
        self.act_report = self.actions_by_name["report"]
        self.act_chat = self.actions_by_name["chat"]
        self.act_chat.setChecked(False)
        self.actions_by_name["load_labels"].setChecked(self.viewport.load_labels)
        self.actions_by_name["lang"].setChecked(False)

        self.mode_actions = {"模型": self.actions_by_name["model"],
                             "分析网格": self.actions_by_name["analysis_mesh"],
                             "变形": self.actions_by_name["deformed"],
                             "云图": self.actions_by_name["contour"],
                             "内力图": self.actions_by_name["force_diagram"],
                             "应力比": self.actions_by_name["utilization"],
                             "模态": self.actions_by_name["modal"]}
        group = QActionGroup(self)
        group.setExclusive(True)
        for act in self.mode_actions.values():
            group.addAction(act)
        self.mode_actions[self.mode].setChecked(True)

        self.pick_actions = {"node": self.actions_by_name["pick_node"],
                             "member": self.actions_by_name["pick_member"]}
        picks = QActionGroup(self)
        picks.setExclusive(True)
        picks.setExclusionPolicy(
            QActionGroup.ExclusionPolicy.ExclusiveOptional)
        for act in self.pick_actions.values():
            picks.addAction(act)

    def _run_command(self, cmd) -> None:
        """分派命令；缺处理函数或执行异常时在界面上说明原因。"""
        handler = getattr(self, cmd.handler, None)
        if handler is None:
            QMessageBox.information(self, cmd.label,
                                    f"「{cmd.label}」还没接上（缺 {cmd.handler}）。")
            return
        # 换命令即放弃之前待拾取的命令，避免稍后点击对象时误弹旧对话框。
        self._pending_command = None
        try:
            handler(cmd) if _takes_command(handler) else handler()
        except Exception as e:  # noqa: BLE001  界面边界：异常显示给用户，不让主窗口退出
            QMessageBox.critical(self, f"{cmd.label} 失败",
                                 f"错误：{type(e).__name__}: {str(e)}")
            traceback.print_exc()
