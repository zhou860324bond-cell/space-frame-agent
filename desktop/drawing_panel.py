"""结构图纸识别 → 模型 JSON（桌面端面板）。

流程只有四步：上传图纸 → 识别文字 → 逐项核对 → 生成模型 JSON / 导入。
识别结果不是坐标而是图上的文字证据，组装由 ``drawing_assemble`` 确定性完成；
核对表里每行写明值、出处与状态，选中一行即在原图上高亮那段原文。缺失和冲突
必须由用户填写或改正，需确认的项点一次「确认原值」。全部处理完才能生成。
识别提示只列出来供对照原图，不需要逐条确认。
"""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QListWidget, QPushButton,
                               QSpinBox, QSplitter, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from . import glyphs, theme
from .sketch_panel import _provider_with_credentials

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
DXF_SUFFIX = ".dxf"
DXF_VIEW_NOTE = ("DXF 是矢量文件，这里不显示图片。\n核对表「来源」列写明每个数取自哪个实体"
                 "（#句柄）和图层，可在 CAD 中按句柄或图层找到原对象核对。")
STATUS_TEXT = {"ok": "已读取", "confirm": "需确认", "missing": "缺失", "conflict": "冲突",
               "info": "提示"}
STATUS_COLOR = {"ok": theme.SUCCESS, "confirm": theme.WARN,
                "missing": theme.ERROR, "conflict": theme.ERROR, "info": theme.INK_MUTED}
STATUS_ORDER = {"conflict": 0, "missing": 1, "confirm": 2, "info": 3, "ok": 4}
PDF_HINT = ("本版不能直接读取 PDF：请在看图软件中把需要的页导出为 PNG（建议 200 dpi 以上），"
            "再用「添加图纸」选入。")


def _layer_roles(evidence: dict | None) -> dict:
    """各图层用途；未判断与「不使用」都不读取，视为相同。"""
    return {item.get("name"): item.get("role") or "ignore"
            for item in (evidence or {}).get("layers") or []}


class DrawingView(QWidget):
    """显示一张图纸，叠画算出的轴线与当前核对项的原文位置。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(260, 220)
        self._pixmap: QPixmap | None = None
        self._image_id: str | None = None
        self._highlight: list[float] | None = None
        self._axes: list[tuple[str, str, float, float]] = []   # 族、轴号、位置、坐标
        self.note = ""

    def show_image(self, path: str | None, image_id: str | None) -> None:
        self._pixmap = QPixmap(path) if path else None
        self._image_id = image_id
        self.note = ""
        self.update()

    def show_note(self, text: str) -> None:
        """没有图片可显示（如 DXF）时写一段替代说明。"""
        self._pixmap = None
        self._image_id = None
        self.note = text
        self.update()

    @property
    def image_id(self) -> str | None:
        return self._image_id

    def set_overlay(self, axes, highlight: list[float] | None) -> None:
        self._axes = list(axes)
        self._highlight = highlight
        self.update()

    def _target(self) -> QRectF | None:
        if self._pixmap is None or self._pixmap.isNull():
            return None
        w, h = self._pixmap.width(), self._pixmap.height()
        scale = min(self.width() / w, self.height() / h)
        return QRectF((self.width() - w * scale) / 2, (self.height() - h * scale) / 2,
                      w * scale, h * scale)

    def paintEvent(self, event):  # noqa: N802 - Qt 命名
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(theme.PANEL_ALT))
        target = self._target()
        if target is None:
            painter.setPen(QColor(theme.INK_MUTED))
            painter.drawText(self.rect().adjusted(12, 12, -12, -12),
                             Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                             self.note or "添加图纸后在这里显示")
            return
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawPixmap(target, self._pixmap, QRectF(self._pixmap.rect()))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor(theme.ACCENT), 1.4, Qt.PenStyle.DashDotLine)
        for family, label, pos, coord in self._axes:
            painter.setPen(pen)
            if family == "x":
                x = target.left() + pos * target.width()
                painter.drawLine(int(x), int(target.top()), int(x), int(target.bottom()))
                text_at = (int(x) + 3, int(target.top()) + 14)
            else:
                y = target.top() + pos * target.height()
                painter.drawLine(int(target.left()), int(y), int(target.right()), int(y))
                text_at = (int(target.left()) + 3, int(y) - 3)
            painter.setPen(QColor(theme.ACCENT_DIM))
            painter.drawText(*text_at, f"{label}：{coord:g} m")
        if self._highlight:
            u1, v1, u2, v2 = self._highlight
            rect = QRectF(target.left() + u1 * target.width(),
                          target.top() + v1 * target.height(),
                          (u2 - u1) * target.width(), (v2 - v1) * target.height())
            painter.setPen(QPen(QColor("#8e44ad"), 2.4))
            painter.setBrush(QBrush(QColor(142, 68, 173, 40)))
            painter.drawRect(rect.adjusted(-3, -3, 3, 3))


class DrawingPanel(QWidget):
    """结构图纸识别面板。"""

    model_loaded = Signal(dict)

    def __init__(self, session, runner, parent=None):
        super().__init__(parent)
        self.session = session
        self.runner = runner
        self.images: list[str] = []
        self.evidence: dict | None = None
        self.assembly = None
        self._records: dict[str, str] = {}       # 证据图号 → 本地路径
        self._rows = []
        self._build_ui()
        self._refresh()

    # ---- 界面 ----
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(theme.SPACE_MD, theme.SPACE_MD, theme.SPACE_MD,
                                  theme.SPACE_MD)
        layout.setSpacing(theme.SPACE_SM + 2)
        title = QLabel("结构图纸识别 → 模型 JSON")
        title.setStyleSheet(f"font-weight:650; font-size:{theme.FONT_STRONG}; "
                            f"color:{theme.INK};")
        layout.addWidget(title)
        self.lbl_steps = QLabel()
        self.lbl_steps.setTextFormat(Qt.TextFormat.RichText)
        self.lbl_steps.setWordWrap(True)
        self.lbl_steps.setStyleSheet(f"font-size:{theme.FONT_CAPTION};")
        layout.addWidget(self.lbl_steps)

        self.btn_settings = QPushButton(f"识别设置 {glyphs.DISCLOSE_CLOSED}")
        self.btn_settings.setCheckable(True)
        layout.addWidget(self.btn_settings)
        self.settings_widget = QWidget(self)
        row = QHBoxLayout(self.settings_widget)
        row.setContentsMargins(0, 0, 0, 0)
        self.cmb_provider = QComboBox()
        self.cmb_provider.addItems(["openai", "anthropic", "deepseek"])
        self.txt_model = QLineEdit()
        self.txt_model.setPlaceholderText("模型名")
        self.txt_key = QLineEdit()
        self.txt_key.setPlaceholderText("API 密钥（不填读环境变量）")
        self.txt_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.spn_repairs = QSpinBox()
        self.spn_repairs.setRange(0, 3)
        self.spn_repairs.setValue(1)
        self.spn_repairs.setToolTip("尺寸链不闭合或引用错误时，让模型重读的次数")
        for widget in (QLabel("提供商"), self.cmb_provider, QLabel("模型"), self.txt_model,
                       QLabel("密钥"), self.txt_key, QLabel("重读"), self.spn_repairs):
            row.addWidget(widget)
        self.cmb_provider.currentTextChanged.connect(self._set_provider_defaults)
        self.cmb_provider.setCurrentText(_provider_with_credentials())
        self._set_provider_defaults(self.cmb_provider.currentText())
        self.settings_widget.setVisible(False)
        self.btn_settings.toggled.connect(self.settings_widget.setVisible)
        self.btn_settings.toggled.connect(lambda on: self.btn_settings.setText(
            f"识别设置 {glyphs.DISCLOSE_OPEN if on else glyphs.DISCLOSE_CLOSED}"))
        layout.addWidget(self.settings_widget)

        buttons = QHBoxLayout()
        self.btn_add = QPushButton("添加图纸…")
        self.btn_add.setToolTip("平面图、立面图、截面表的图片，可一次选多张；"
                                "或一个 DXF 文件（矢量图纸，数值直接取自实体）")
        self.btn_add.clicked.connect(self._pick_images)
        self.btn_clear = QPushButton("清空")
        self.btn_clear.clicked.connect(self.clear)
        self.btn_recognize = QPushButton("识别图纸")
        self.btn_recognize.setProperty("role", "primary")
        self.btn_recognize.clicked.connect(self.recognize)
        for button in (self.btn_add, self.btn_clear, self.btn_recognize):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.lst_images = QListWidget()
        self.lst_images.setMaximumHeight(64)
        self.lst_images.currentRowChanged.connect(self._show_image_at)
        layout.addWidget(self.lst_images)

        splitter = QSplitter(Qt.Orientation.Vertical, self)
        self.view = DrawingView(splitter)
        splitter.addWidget(self.view)
        table_box = QWidget(splitter)
        box = QVBoxLayout(table_box)
        box.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["状态", "类别", "项目", "值", "来源"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self.table.horizontalHeader()
        for column in (0, 1, 2):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        box.addWidget(self.table)
        self.lbl_row = QLabel("")
        self.lbl_row.setWordWrap(True)
        self.lbl_row.setStyleSheet(f"font-size:{theme.FONT_CAPTION}; color:{theme.INK_MUTED};")
        box.addWidget(self.lbl_row)
        edit_row = QHBoxLayout()
        self.txt_value = QLineEdit()
        self.txt_value.setPlaceholderText("照图填写或改正")
        self.cmb_value = QComboBox()
        self.btn_confirm = QPushButton("确认原值")
        self.btn_confirm.clicked.connect(self.confirm_selected)
        self.btn_apply = QPushButton("应用修改")
        self.btn_apply.clicked.connect(self.apply_selected)
        self.txt_value.returnPressed.connect(self.apply_selected)
        for widget in (self.txt_value, self.cmb_value, self.btn_confirm, self.btn_apply):
            edit_row.addWidget(widget)
        box.addLayout(edit_row)
        splitter.addWidget(table_box)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter, 1)

        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet(f"font-size:{theme.FONT_CAPTION}; color:{theme.INK_MUTED};")
        layout.addWidget(self.lbl_status)
        bottom = QHBoxLayout()
        self.btn_save = QPushButton("保存模型 JSON…")
        self.btn_save.clicked.connect(self._save_dialog)
        self.btn_import = QPushButton("导入为当前模型")
        self.btn_import.setProperty("role", "primary")
        self.btn_import.setToolTip("替换当前模型，占一个撤销步骤")
        self.btn_import.clicked.connect(self.import_model)
        bottom.addWidget(self.btn_save)
        bottom.addWidget(self.btn_import)
        layout.addLayout(bottom)

    def _set_provider_defaults(self, provider: str) -> None:
        from sketch_parser import SketchParser
        self.txt_model.setText(SketchParser._default_model(provider))

    def _set_steps(self, current: int) -> None:
        names = ("上传图纸", "识别文字", "逐项核对", "生成模型")
        parts = []
        for index, name in enumerate(names, 1):
            color = theme.ACCENT if index == current else theme.INK_DIM
            weight = 650 if index == current else 400
            parts.append(f'<span style="color:{color};font-weight:{weight}">'
                         f"{index} {name}</span>")
        self.lbl_steps.setText("　→　".join(parts))

    def _status(self, text: str, color: str = theme.INK_MUTED) -> None:
        self.lbl_status.setText(text)
        self.lbl_status.setStyleSheet(f"font-size:{theme.FONT_CAPTION}; color:{color};")

    # ---- 图纸 ----
    def _pick_images(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "添加结构图纸", "",
            "图纸图片或 DXF (*.png *.jpg *.jpeg *.webp *.dxf);;DXF (*.dxf);;PDF (*.pdf)")
        if paths:
            self.load_images(paths)

    @property
    def dxf_path(self) -> str | None:
        """当前图纸是一个 DXF 时返回它的路径。"""
        if len(self.images) == 1 and Path(self.images[0]).suffix.lower() == DXF_SUFFIX:
            return self.images[0]
        return None

    def load_images(self, paths) -> None:
        """加入图纸；对话面板拖入图纸也走这里。新图纸会让已有识别结果作废。

        DXF 一次读一个，且不与图片混用：图片靠大模型抄字，DXF 直接读实体，两份
        证据的出处不同。加入 DXF 后立即在本机读取，不需要识别模型。
        """
        skipped_pdf = False
        replaced = ""
        dxf = [str(p) for p in paths if Path(p).suffix.lower() == DXF_SUFFIX]
        if dxf:
            if self.images or len(dxf) > 1:
                replaced = (f"DXF 一次读取一个，且不与图片一起识别：已改为只读取"
                            f"「{Path(dxf[0]).name}」。")
            self.images = [dxf[0]]
        else:
            if self.dxf_path:
                replaced = "图片与 DXF 不一起识别：已移除 DXF，改为识别图片。"
                self.images = []
            for path in paths:
                suffix = Path(path).suffix.lower()
                if suffix == ".pdf":
                    skipped_pdf = True
                    continue
                if suffix in IMAGE_SUFFIXES and path not in self.images:
                    self.images.append(str(path))
        self.evidence = None
        self.assembly = None
        self._refresh()
        if self.dxf_path:
            self._read_dxf(replaced)
        elif skipped_pdf:
            self._status(PDF_HINT, theme.WARN)
        elif replaced:
            self._status(replaced + self.lbl_status.text(), theme.WARN)

    def _read_dxf(self, prefix: str = "") -> None:
        from drawing_dxf import DxfError, read_dxf
        try:
            evidence = read_dxf(self.dxf_path)
        except DxfError as exc:
            self._status(prefix + str(exc), theme.ERROR)
            return
        self.set_evidence(evidence)
        self._status(prefix + f"已读取 DXF：{len(evidence['layers'])} 个图层。图层用途要由你"
                     "或大模型判断：在核对表里为各图层选择用途，或点「识别图纸」请大模型判断。",
                     theme.WARN)

    def clear(self) -> None:
        self.images.clear()
        self.evidence = None
        self.assembly = None
        self._records.clear()
        self._refresh()

    def _show_image_at(self, index: int) -> None:
        if self.dxf_path:
            self.view.show_note(DXF_VIEW_NOTE)
        elif 0 <= index < len(self.images):
            self.view.show_image(self.images[index], f"P{index + 1}")
        else:
            self.view.show_image(None, None)
        self._update_overlay()

    # ---- 识别 ----
    def recognize(self) -> None:
        if not self.images or self.runner.busy:
            return
        provider = self.cmb_provider.currentText()
        model = self.txt_model.text().strip()
        key = self.txt_key.text().strip()
        repairs = self.spn_repairs.value()
        paths = list(self.images)
        work_dir = Path.cwd() / ".multimodal_tmp" / "drawing"
        dxf = self.dxf_path
        keep = [item for item in (self.evidence or {}).get("layers") or []
                if (item.get("source") or {}).get("user")]

        def parser():
            from sketch_parser import SketchParser
            return (SketchParser(provider=provider, api_key=key, model=model) if key
                    else SketchParser.from_env(provider, model=model))

        def job():
            from drawing_extract import DrawingExtractor
            return DrawingExtractor(parser(), work_dir).extract(paths, max_repairs=repairs)

        def dxf_job():
            from drawing_dxf import judge_layers, model_layer_items, read_dxf, scan_layers
            from drawing_extract import DrawingExtractor
            extractor = DrawingExtractor(parser(), work_dir)
            roles = judge_layers(lambda system, text: extractor._call(system, [("text", text)]),
                                 scan_layers(dxf))
            return read_dxf(dxf, model_layer_items(dxf, roles, f"{provider} {model}", keep))

        self.btn_recognize.setEnabled(False)
        self._set_steps(2)
        if dxf:
            self._status("正在请大模型按图层名与内容判断图层用途；数值仍直接取自 DXF 实体…")
            self.runner.submit(dxf_job, on_done=self._on_layers_judged,
                               on_failed=self._on_failed)
            return
        self._status("正在识别：先定位各区域，再按原分辨率逐块抄写文字…")
        self.runner.submit(job, on_done=self._on_recognized, on_failed=self._on_failed)

    def _on_layers_judged(self, evidence: dict) -> None:
        self.set_evidence(evidence)
        self._status("图层用途已由大模型判断，请在核对表的「图层」类中逐个确认或改选。"
                     + self.assembly.summary(), theme.WARN)

    def _on_recognized(self, result) -> None:
        if not result.success:
            self._refresh()
            self._status("识别没有得到结果：" + "；".join(result.errors[-2:])
                         + "。请检查识别设置中的提供商与密钥后重试。", theme.ERROR)
            return
        self.set_evidence(result.evidence)
        if result.errors:
            self._status(self.lbl_status.text() + " 识别过程提示：" + result.errors[-1],
                         theme.WARN)

    def _on_failed(self, exc_type: str, message: str) -> None:
        self._refresh()
        self._status(f"识别失败（{exc_type}）：{message}。请检查网络与密钥后重试。", theme.ERROR)

    def set_evidence(self, evidence: dict) -> None:
        """载入一份证据（识别结果或以后的 DXF 读取结果）并重新组装。"""
        self.evidence = evidence
        self._records = {f"P{index}": path for index, path in enumerate(self.images, 1)}
        self._reassemble()

    def _reassemble(self, keep_row: str | None = None) -> None:
        from drawing_assemble import assemble
        self.assembly = assemble(self.evidence) if self.evidence else None
        self._refresh(keep_row)

    # ---- 核对表 ----
    def _refresh(self, keep_row: str | None = None) -> None:
        self.lst_images.blockSignals(True)
        current = self.lst_images.currentRow()
        self.lst_images.clear()
        for index, path in enumerate(self.images, 1):
            self.lst_images.addItem(f"图 {index}  {Path(path).name}")
        self.lst_images.blockSignals(False)
        if self.images:
            self.lst_images.setCurrentRow(min(max(current, 0), len(self.images) - 1))
        else:
            self._show_image_at(-1)
        self.btn_recognize.setEnabled(bool(self.images) and not self.runner.busy)
        rows = sorted(self.assembly.rows, key=lambda r: STATUS_ORDER[r.status]) \
            if self.assembly else []
        self._rows = rows
        self.table.blockSignals(True)
        self.table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            cells = (STATUS_TEXT[row.status], row.category, row.item, row.value, row.source)
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setToolTip(row.message or text)
                if column == 0:
                    item.setForeground(QColor(STATUS_COLOR[row.status]))
                self.table.setItem(index, column, item)
        self.table.blockSignals(False)
        ready = bool(self.assembly and self.assembly.ready)
        self.btn_save.setEnabled(ready)
        self.btn_import.setEnabled(ready)
        if self.assembly is None:
            self._set_steps(1 if not self.images else 2)
            self._status("添加平面图、立面图与截面表后点「识别图纸」。" if not self.images
                         else f"已添加 {len(self.images)} 张图纸，点「识别图纸」开始。")
        else:
            self._set_steps(4 if ready else 3)
            self._status(self.assembly.summary(), theme.SUCCESS if ready else theme.WARN)
        target = next((i for i, r in enumerate(rows) if r.id == keep_row), None)
        if target is None:
            target = next((i for i, r in enumerate(rows) if r.blocking or r.confirmable),
                          0 if rows else None)
        if target is not None:
            self.table.selectRow(target)
        self._on_row_selected()

    def selected_row(self):
        indexes = self.table.selectionModel().selectedRows() if self.table.selectionModel() \
            else []
        if not indexes or indexes[0].row() >= len(self._rows):
            return None
        return self._rows[indexes[0].row()]

    def _on_row_selected(self) -> None:
        row = self.selected_row()
        self.txt_value.setVisible(row is not None and row.edit == "text")
        self.cmb_value.setVisible(row is not None and row.edit == "choice")
        self.btn_confirm.setEnabled(row is not None and row.confirmable)
        self.btn_apply.setEnabled(row is not None and row.edit != "none")
        if row is None:
            self.lbl_row.setText("")
            self._update_overlay()
            return
        self.lbl_row.setText(f"{row.item}：{row.message or '已读取，无需处理。'}")
        if row.edit == "text":
            self.txt_value.setText(row.current)
        elif row.edit == "choice":
            self.cmb_value.clear()
            self.cmb_value.addItems(list(row.options))
            if row.current in row.options:
                self.cmb_value.setCurrentText(row.current)
        if row.image and row.image in self._records:
            index = self.images.index(self._records[row.image])
            if self.lst_images.currentRow() != index:
                self.lst_images.setCurrentRow(index)
        self._update_overlay()

    def _update_overlay(self) -> None:
        image_id = self.view.image_id
        axes = []
        if self.evidence and self.assembly and image_id:
            for family in ("x", "y"):
                coords = self.assembly.axes.get(family, {})
                for axis in ((self.evidence.get("grids") or {}).get(family) or {}).get(
                        "axes") or []:
                    source = axis.get("source") or {}
                    label = str(axis.get("label"))
                    if source.get("image") == image_id and isinstance(
                            axis.get("pos"), (int, float)) and label in coords:
                        axes.append((family, label, float(axis["pos"]), coords[label]))
        row = self.selected_row()
        highlight = row.bbox if row is not None and row.image == image_id else None
        self.view.set_overlay(axes, highlight)

    # ---- 处理 ----
    def _resolve(self, value: str | None) -> None:
        from drawing_assemble import resolve
        row = self.selected_row()
        if row is None or self.evidence is None:
            return
        try:
            evidence = resolve(self.evidence, row.id, value)
            if self.dxf_path and row.id.startswith("layers.") \
                    and _layer_roles(evidence) != _layer_roles(self.evidence):
                from drawing_dxf import reread
                evidence = reread(evidence, self.dxf_path)
        except ValueError as exc:    # DxfError 也是 ValueError
            self._status(f"{row.item}未更新：{exc}", theme.ERROR)
            return
        self.evidence = evidence
        self._reassemble(keep_row=None if value is None else row.id)

    def confirm_selected(self) -> None:
        self._resolve(None)

    def apply_selected(self) -> None:
        row = self.selected_row()
        if row is None or row.edit == "none":
            return
        value = self.txt_value.text() if row.edit == "text" else self.cmb_value.currentText()
        self._resolve(value)

    # ---- 产物 ----
    def model_payload(self) -> dict | None:
        return self.assembly.model if self.assembly and self.assembly.ready else None

    def save_model(self, path: str | Path) -> Path:
        """写模型 JSON，并在旁边写一份证据与出处（文件名加 .drawing-evidence）。"""
        from model_io import atomic_write_text
        model = self.model_payload()
        if model is None:
            raise ValueError("核对表还有未处理的项，不能生成模型 JSON。")
        path = Path(path)
        atomic_write_text(path, json.dumps(model, ensure_ascii=False, indent=2))
        sidecar = path.with_name(path.stem + ".drawing-evidence.json")
        atomic_write_text(sidecar, json.dumps(
            {"evidence": self.evidence, "provenance": self.assembly.provenance},
            ensure_ascii=False, indent=2))
        return sidecar

    def _save_dialog(self) -> None:
        if self.model_payload() is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存模型 JSON", "图纸识别模型.json",
                                              "模型 JSON (*.json)")
        if not path:
            return
        try:
            sidecar = self.save_model(path)
        except OSError as exc:
            self._status(f"保存未完成：{exc}。请检查目录权限或另选位置。", theme.ERROR)
            return
        self._status(f"已保存 {Path(path).name}，识别证据与出处在 {sidecar.name}。",
                     theme.SUCCESS)

    def import_model(self) -> bool:
        model = self.model_payload()
        if model is None:
            return False
        if self.runner.busy:
            self._status("后台任务执行中；请等待完成后再导入。", theme.WARN)
            return False
        result = self.session.set_model(model)
        if not result.ok:
            errors = result.payload.get("errors") or [result.payload.get("error", "")]
            self._status("导入未完成：" + "；".join(map(str, errors[:3])), theme.ERROR)
            return False
        self.model_loaded.emit(self.session.model)
        self._status("已导入为当前模型（一个撤销步骤）。下一步：添加荷载后求解；"
                     "如需留存识别出处，请用「保存模型 JSON」。", theme.SUCCESS)
        return True
