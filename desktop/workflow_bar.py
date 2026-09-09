"""Persistent UI projection of the deterministic backend workflow state."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from workflow import WorkflowPhase, inspect_workflow


def draft_target(status) -> tuple[str, str, str]:
    """Return the visible workflow stage, ribbon page and human next step.

    Validation remains authoritative.  Counts only turn its schema-oriented
    messages into a useful UI destination; they never decide that a model is
    valid.
    """
    counts = status.counts
    if counts["nodes"] < 2:
        return "model", "建模", f"当前 {counts['nodes']} 个节点；至少创建 2 个节点"
    if counts["members"] < 1:
        return "model", "建模", "节点已创建；下一步连接至少 1 根杆件"
    if counts["materials"] < 1:
        return "define", "属性", "杆系几何已建立；下一步定义材料"
    if counts["sections"] < 1:
        return "define", "属性", "材料已定义；下一步定义并指派截面"
    if counts["supports"] < 1:
        return "define", "载荷", "属性已就绪；下一步施加支座约束"
    if counts["load_cases"] < 1:
        return "define", "载荷", "约束已设置；下一步创建荷载工况"
    return "define", "建模", "模型仍有校验问题；请查看下方问题并修正"


class WorkflowBar(QWidget):
    """Show what is complete, what blocks progress, and the next real action."""

    stage_requested = Signal(str)
    STAGES = (
        ("model", "1  建模"),
        ("define", "2  定义"),
        ("validate", "3  校验"),
        ("solve", "4  求解"),
        ("results", "5  结果"),
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("workflowBar")
        row = QHBoxLayout(self)
        row.setContentsMargins(10, 4, 10, 4)
        row.setSpacing(4)
        label = QLabel("分析流程", self)
        label.setProperty("workflow", "caption")
        row.addWidget(label)
        self.buttons: dict[str, QPushButton] = {}
        for stage, text in self.STAGES:
            button = QPushButton(text, self)
            button.setProperty("workflowStage", stage)
            button.setProperty("workflowState", "pending")
            button.clicked.connect(
                lambda _=False, value=stage: self.stage_requested.emit(value))
            row.addWidget(button)
            self.buttons[stage] = button
        self.guidance = QLabel("", self)
        self.guidance.setProperty("workflow", "guidance")
        self.guidance.setMinimumWidth(260)
        row.addWidget(self.guidance, 1)
        self.next_button = QPushButton("开始建模", self)
        self.next_button.setProperty("role", "primary")
        self.next_button.clicked.connect(self._request_next)
        row.addWidget(self.next_button)
        self._next_stage = "model"
        self.status = None

    @staticmethod
    def _set_state(button: QPushButton, state: str, suffix: str = "") -> None:
        base = next(text for stage, text in WorkflowBar.STAGES
                    if stage == button.property("workflowStage"))
        button.setText(base + suffix)
        button.setProperty("workflowState", state)
        button.style().unpolish(button)
        button.style().polish(button)

    def update_state(self, session) -> None:
        status = inspect_workflow(session)
        self.status = status
        phase = status.phase
        states = {stage: "pending" for stage, _ in self.STAGES}
        suffixes = {stage: "" for stage, _ in self.STAGES}
        if phase is WorkflowPhase.EMPTY:
            states["model"] = "active"
            self._next_stage = "model"
            message, action = "尚未创建几何模型", "开始建模"
        elif phase is WorkflowPhase.DRAFT:
            active_stage, _page, guidance = draft_target(status)
            states[active_stage] = "active"
            if active_stage == "define":
                states["model"] = "done"
                suffixes["model"] = "  ✓"
            states["validate"] = "blocked"
            suffixes["validate"] = f"  !{len(status.validation_errors)}"
            self._next_stage = active_stage
            message = guidance
            action = "修正模型"
        elif phase is WorkflowPhase.READY:
            for stage in ("model", "define", "validate"):
                states[stage], suffixes[stage] = "done", "  ✓"
            states["solve"] = "active"
            self._next_stage = "solve"
            message, action = "模型已通过确定性校验，可以提交分析", "开始求解"
        else:
            for stage, _ in self.STAGES:
                states[stage], suffixes[stage] = "done", "  ✓"
            states["results"] = "active"
            self._next_stage = "results"
            message, action = "求解完成，可查看变形、内力与报告", "查看结果"
        for stage, button in self.buttons.items():
            self._set_state(button, states[stage], suffixes[stage])
        self.guidance.setText(message)
        self.guidance.setToolTip("\n".join(status.validation_errors))
        self.next_button.setText(action)

    def _request_next(self) -> None:
        self.stage_requested.emit(self._next_stage)
