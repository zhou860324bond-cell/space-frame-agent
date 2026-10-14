"""非模态模型连接窗口；后台检测与工程任务共用 Runner 的互斥与退出机制。"""
from __future__ import annotations

import os
import uuid
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFileDialog,
                               QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QSpinBox, QVBoxLayout)

from model_profiles import ModelProfile, ProfileStore, redact
from . import dialog_styles


class ModelSettings(QDialog):
    profiles_changed = Signal(object, str)

    def __init__(self, runner, profiles, selected, parent=None):
        super().__init__(parent)
        self.runner = runner
        self.profiles = [replace(p) for p in profiles]
        self.setWindowTitle("模型连接设置")
        self.resize(560, 470)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(dialog_styles.DialogHeader(
            "模型连接设置", "每个连接独立保存地址与密钥。检测只发送固定探针；正式对话会发送对话与工程上下文。"))
        box = QVBoxLayout()
        box.setContentsMargins(14, 10, 14, 12)
        box.setSpacing(8)
        outer.addLayout(box)
        top = QHBoxLayout()
        top.setSpacing(6)
        self.saved = QComboBox()
        self.saved.addItem("新建连接", "")
        for p in self.profiles:
            self.saved.addItem(p.label, p.id)
        top.addWidget(self.saved, 1)
        # 删除是破坏性操作，用红色描边的次要按钮，不能抢主操作的蓝色
        self.remove = QPushButton("删除")
        self.remove.setProperty("role", "danger")
        top.addWidget(self.remove)
        self.form = QFormLayout()
        self.form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.form.setHorizontalSpacing(10)
        self.form.setVerticalSpacing(7)
        self.form.addRow("已保存的连接", top)
        self.preset = QComboBox()
        self.preset.addItems(["DeepSeek", "OpenAI", "Anthropic", "自定义兼容接口"])
        self.label = QLineEdit()
        self.protocol = QComboBox()
        self.protocol.addItem("Chat Completions", "openai-chat")
        self.protocol.addItem("Anthropic Messages", "anthropic")
        self.url = QLineEdit()
        self.model = QComboBox()
        self.model.setEditable(True)
        self.model.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.key = QLineEdit()
        self.key.setEchoMode(QLineEdit.EchoMode.Password)
        key_row = QHBoxLayout()
        key_row.addWidget(self.key, 1)
        self.import_key = QPushButton("导入密钥文件")
        key_row.addWidget(self.import_key)
        self.persist = QCheckBox("使用 Windows 安全存储")
        self.persist.setEnabled(os.name == "nt")
        self.persist.setChecked(os.name == "nt")
        self.persist.setToolTip("未勾选时密钥只在本次运行有效，配置名称与地址仍保存")
        self.temperature = QComboBox()
        for label, value in [("提供商默认", None), ("严谨 · 0", 0.0), ("均衡 · 0.5", 0.5), ("探索 · 1", 1.0)]:
            self.temperature.addItem(label, value)
        self.tokens = QSpinBox()
        self.tokens.setRange(256, 32768)
        self.tokens.setValue(4096)
        for label, widget in [("提供商模板", self.preset), ("连接名称", self.label),
                              ("协议", self.protocol), ("API 基础地址", self.url),
                              ("模型标识", self.model), ("API 密钥", key_row),
                              ("密钥保存", self.persist), ("采样温度", self.temperature),
                              ("输出上限", self.tokens)]:
            self.form.addRow(label, widget)
        box.addLayout(self.form)
        self.status = QLabel("填写实际模型标识，或读取账户可用模型目录。目录不提供完整能力保证。")
        self.status.setWordWrap(True)
        self.status.setProperty("panel", "hint")
        box.addWidget(self.status)
        box.addStretch(1)
        actions = QHBoxLayout()
        self.directory = QPushButton("读取模型目录")
        self.probe = QPushButton("检测工具调用")
        self.probe.setToolTip("发送一次固定模型请求，可能计入提供商用量；不上传工程或调用求解器")
        self.save = QPushButton("保存并使用")
        actions.addWidget(self.directory)
        actions.addWidget(self.probe)
        actions.addStretch(1)
        actions.addWidget(self.save)
        box.addLayout(actions)
        dialog_styles.set_primary(self.save, self.remove, self.import_key, self.directory, self.probe)
        self.preset.currentIndexChanged.connect(self._preset)
        self.saved.currentIndexChanged.connect(self._load)
        self.import_key.clicked.connect(self._import)
        self.remove.clicked.connect(self._remove)
        self.save.clicked.connect(self._save)
        self.directory.clicked.connect(lambda: self._test(False))
        self.probe.clicked.connect(lambda: self._test(True))
        runner.busy_changed.connect(self._busy)
        self._preset(0)
        self.saved.setCurrentIndex(max(0, self.saved.findData(selected)))
        self._busy(runner.busy)

    def _preset(self, index):
        templates = [("DeepSeek", "https://api.deepseek.com", "deepseek-v4-flash", "openai-chat"),
                     ("OpenAI", "https://api.openai.com/v1", "", "openai-chat"),
                     ("Anthropic", "https://api.anthropic.com/v1", "", "anthropic"),
                     ("自定义连接", "", "", "openai-chat")]
        label, url, model, protocol = templates[index]
        self.label.setText(label)
        self.url.setText(url)
        self.model.clear()
        self.model.setEditText(model)
        self.protocol.setCurrentIndex(self.protocol.findData(protocol))
        # 提供商切换时不能携带旧密钥。
        self.key.clear()

    def _load(self, _index):
        profile = next((p for p in self.profiles if p.id == self.saved.currentData()), None)
        if profile is None:
            self._preset(self.preset.currentIndex())
            return
        self.label.setText(profile.label)
        self.url.setText(profile.base_url)
        self.model.clear()
        self.model.setEditText(profile.model)
        self.key.setText(profile.api_key)
        self.protocol.setCurrentIndex(self.protocol.findData(profile.protocol))
        self.temperature.setCurrentIndex(max(0, self.temperature.findData(profile.temperature)))
        self.tokens.setValue(profile.max_tokens)
        self.persist.setChecked(profile.persist_key)

    def _profile(self, directory=False):
        profile = ModelProfile(self.saved.currentData() or uuid.uuid4().hex,
                               self.label.text().strip(), self.model.currentText().strip() or
                               ("directory-probe" if directory else ""),
                               self.url.text().strip(), self.protocol.currentData(),
                               self.key.text().strip(), self.temperature.currentData(),
                               self.tokens.value(), self.persist.isChecked())
        profile.validate()
        return profile

    def _import(self):
        filename, _ = QFileDialog.getOpenFileName(self, "导入密钥文件", "", "密钥文本 (*.key *.txt);;全部文件 (*)")
        if not filename:
            return
        try:
            path = Path(filename)
            if path.stat().st_size > 8192:
                raise ValueError
            text = path.read_text(encoding="utf-8-sig").strip()
            if not text or any(c.isspace() for c in text):
                raise ValueError
            self.key.setText(text)
            self.status.setText("密钥已导入输入框；保存连接后生效。源文件由用户自行管理。")
        except (OSError, UnicodeError, ValueError):
            self.status.setText("文件无法读取或不包含单行密钥，请选择 UTF-8 文本密钥文件。")

    def _busy(self, busy):
        for widget in (self.saved, self.remove, self.preset, self.label, self.protocol,
                       self.url, self.model, self.key, self.import_key, self.temperature,
                       self.tokens, self.directory, self.probe, self.save):
            widget.setEnabled(not busy)
        self.persist.setEnabled(not busy and os.name == "nt")

    def _save(self):
        if self.runner.busy:
            return
        try:
            p = self._profile()
            profiles = [other for other in self.profiles if other.id != p.id] + [p]
            ProfileStore().save(profiles, p.id)
        except (ValueError, OSError) as exc:
            self.status.setText(redact(str(exc), [self.key.text()]))
            return
        self.profiles = profiles
        self.profiles_changed.emit(profiles, p.id)
        self.close()

    def _remove(self):
        if self.runner.busy or not self.saved.currentData():
            return
        profiles = [p for p in self.profiles if p.id != self.saved.currentData()]
        selected = profiles[0].id if profiles else ""
        try:
            ProfileStore().save(profiles, selected)
        except (ValueError, OSError) as exc:
            self.status.setText(redact(str(exc), [self.key.text()]))
            return
        self.profiles = profiles
        self.profiles_changed.emit(profiles, selected)
        self.saved.removeItem(self.saved.currentIndex())

    def _test(self, tools):
        if self.runner.busy:
            return
        try:
            profile = self._profile(directory=not tools)
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        from configured_provider import list_models, test_tools
        self.status.setText("正在检测工具调用…" if tools else "正在读取模型目录…")
        def done(result):
            if tools:
                self.status.setText("所选模型已返回有效工具调用；检测未调用求解器或修改工程模型。")
            else:
                current = self.model.currentText()
                self.model.clear()
                self.model.addItems(result)
                self.model.setEditText(current or (result[0] if result else ""))
                self.status.setText(f"目录连接成功，共 {len(result)} 个模型；请选择后检测工具调用。")
        self.runner.submit(lambda: test_tools(profile) if tools else list_models(profile),
                           on_done=done, on_failed=lambda kind, message: self.status.setText(
                               f"连接检测失败：{kind}。{redact(message, [profile.api_key])[:220]}"))
