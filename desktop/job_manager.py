"""作业管理器（Abaqus 的 Job Manager）：创建、数据检查、写 inp、提交、监控、结果、终止。

原来按 F5 直接求解，没有「作业」这一层：算过什么、用的什么设置、出了什么警告，
只在状态栏闪一下就没了；改了模型之后旧结果是不是还对应当前模型，也只能靠记忆。

作业只记录「怎么算、算得怎样」：类型（独立工况或声明的分析步）、状态、提交与完成
时间、消息（平衡校核、静默失败检查、各分析步的频率或临界因子、错误），以及提交时
的模型指纹。作业不写进模型——写进去的话一提交就改了模型指纹，结果反而失效。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from PySide6.QtWidgets import (QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout,
                               QHeaderView, QLineEdit, QListWidget, QPushButton,
                               QTableWidget, QTableWidgetItem, QVBoxLayout)

from .dialog_styles import DialogHeader, button_box, style_dialog

#: 作业类型 → (显示名, 求解方式)
JOB_TYPES = (("独立工况 · 线性静力", ("cases", "linear")),
             ("独立工况 · P-Delta 二阶弹性", ("cases", "pdelta")),
             ("独立工况 · 轴向材料非线性", ("cases", "material")),
             ("声明的分析步（含频率、屈曲步）", ("steps", None)))


@dataclass
class Job:
    name: str
    kind: str                       # "cases" 或 "steps"
    analysis: str | None = None     # 独立工况时的分析类型
    status: str = "未提交"
    submitted: str = ""
    finished: str = ""
    fingerprint: bytes | None = None
    messages: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        for text, (kind, analysis) in JOB_TYPES:
            if (kind, analysis) == (self.kind, self.analysis):
                return text
        return self.kind


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def describe_result(job: Job, result) -> list[str]:
    """把一次求解的返回整理成作业消息：平衡、静默失败、各步结果、错误。"""
    payload: dict[str, Any] = result.payload if result is not None else {}
    if result is None or not result.ok:
        errors = payload.get("errors") or [payload.get("error") or "求解失败"]
        detail = payload.get("detail")
        lines = [f"错误：{e}" for e in errors]
        if isinstance(detail, dict):
            lines += [f"详情：{detail.get('error') or detail.get('errors')}"]
        return lines
    lines: list[str] = []
    if job.kind == "steps":
        for row in payload.get("steps") or []:
            if "frequencies_Hz" in row:
                text = "、".join(f"{f:.4g}" for f in row["frequencies_Hz"][:6])
                lines.append(f"分析步 {row['step']}（频率）：{text} Hz")
            elif "critical_factor" in row:
                lines.append(f"分析步 {row['step']}（屈曲）：临界因子 λ = {row['critical_factor']:.4g}")
            else:
                ok = "通过" if row.get("equilibrium_ok") else "未通过"
                lines.append(f"分析步 {row['step']}（{row['analysis']}）：最大位移 "
                             f"{row.get('max_displacement_mm')} mm，平衡校核{ok}")
        if payload.get("limitation"):
            lines.append(f"说明：{payload['limitation']}")
        return lines
    for case, entry in (payload.get("cases") or {}).items():
        ok = "通过" if entry.get("equilibrium_ok") else "未通过"
        lines.append(f"工况 {case}：最大位移 {entry.get('max_displacement_mm')} mm，平衡校核{ok}")
        if entry.get("warning"):
            lines.append(f"警告（{case}）：{entry['warning']}")
    findings = (payload.get("silent_failures") or {}).get("findings") or []
    bad = [f for f in findings if f.get("status") != "pass"]
    if findings:
        lines.append(f"静默失败检查：{len(findings) - len(bad)}/{len(findings)} 项通过")
        lines += [f"异常：{f['name']}——{f.get('message', '')}" for f in bad]
    return lines


class CreateJobDialog(QDialog):
    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("创建作业")
        layout = QVBoxLayout(self)
        layout.addWidget(DialogHeader("创建作业", "作业记录求解方式；提交后在管理器里监控状态与消息。"))
        form = QFormLayout()
        self.name = QLineEdit(name, self)
        self.kind = QComboBox(self)
        for text, value in JOB_TYPES:
            self.kind.addItem(text, value)
        form.addRow("名称", self.name)
        form.addRow("类型", self.kind)
        layout.addLayout(form)
        box = button_box(self)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        layout.addWidget(box)
        style_dialog(self, 380, 200)

    def values(self) -> tuple[str, str, str | None]:
        kind, analysis = self.kind.currentData()
        return self.name.text().strip(), kind, analysis


class JobManagerDialog(QDialog):
    """作业列表与操作；非模态，提交后可以一边看状态一边继续操作。"""

    COLUMNS = ("名称", "类型", "状态", "提交", "完成")

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle("作业管理器")
        layout = QVBoxLayout(self)
        layout.addWidget(DialogHeader("作业管理器", "数据检查只校验模型；提交在后台求解，完成后可直接查看结果。"))
        self.table = QTableWidget(0, len(self.COLUMNS), self)
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.currentCellChanged.connect(lambda *_: self._show_messages())
        layout.addWidget(self.table, 1)
        self.messages = QListWidget(self)
        layout.addWidget(self.messages, 1)
        row = QHBoxLayout()
        self.buttons: dict[str, QPushButton] = {}
        for key, text in (("create", "创建…"), ("check", "数据检查"), ("inp", "写 inp…"),
                          ("submit", "提交"), ("results", "结果"), ("kill", "终止"),
                          ("delete", "删除")):
            button = QPushButton(text, self)
            button.clicked.connect(lambda _=False, k=key: self._run(k))
            row.addWidget(button)
            self.buttons[key] = button
        layout.addLayout(row)
        close = button_box(self, ok="关闭")
        close.accepted.connect(self.accept)
        close.rejected.connect(self.reject)
        layout.addWidget(close)
        style_dialog(self, 720, 460)
        self.refresh()

    def selected(self) -> Job | None:
        row = self.table.currentRow()
        jobs = self.window.jobs
        return jobs[row] if 0 <= row < len(jobs) else None

    def refresh(self) -> None:
        jobs = self.window.jobs
        current = self.table.currentRow()
        self.table.setRowCount(len(jobs))
        for row, job in enumerate(jobs):
            for col, text in enumerate((job.name, job.label, job.status, job.submitted,
                                        job.finished)):
                self.table.setItem(row, col, QTableWidgetItem(text))
        if jobs:
            self.table.selectRow(min(max(current, 0), len(jobs) - 1))
        self._show_messages()

    def _show_messages(self) -> None:
        job = self.selected()
        self.messages.clear()
        if job is not None:
            self.messages.addItems(job.messages or ["（还没有消息）"])
        busy = self.window.runner.busy
        for key, button in self.buttons.items():
            if key == "create":
                enabled = not busy
            elif key == "kill":
                enabled = busy
            elif key == "results":
                enabled = job is not None
            else:                       # 数据检查、写 inp、提交、删除：运行中不能动
                enabled = job is not None and not busy
            button.setEnabled(enabled)

    def _run(self, action: str) -> None:
        self.window.job_action(action, self.selected())
        self.refresh()

    def ask_inp_path(self, job: Job) -> str:
        path, _ = QFileDialog.getSaveFileName(self, "写 inp", f"{job.name}.inp",
                                              "Abaqus 输入文件 (*.inp)")
        return path
