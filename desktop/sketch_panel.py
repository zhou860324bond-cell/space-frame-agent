"""手绘草图 → 结构模型（桌面端面板）。

上传手绘草图，多模态 LLM 识别节点/杆件/支座/荷载，先转换成可编辑、
可确认的几何草稿。LLM 不产结构响应，数值仍由确定性求解器计算。

大模型调用走 Runner 后台线程，不阻塞界面。
"""

from __future__ import annotations

import math
from pathlib import Path

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
                               QListWidget, QListWidgetItem, QPushButton, QSpinBox,
                               QVBoxLayout, QWidget)

from . import theme
from .image_preprocess_widget import ImagePreprocessWidget
from .issue_panel import IssuePanel


def _provider_with_credentials(default: str = "openai") -> str:
    """默认选一个**真的有密钥**的提供商。

    下拉框原来固定停在第一项 openai，而多数机器上只配了 deepseek.key。
    于是打开面板点识别，必然是"缺少 API 密钥"——用户以为是识别功能坏了，
    实际只是选错了提供商。首次就注定失败的默认值不是中立，是坑。
    """
    import os
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    try:
        from credentials import load_api_key
        if load_api_key():
            return "deepseek"
    except Exception:                       # 读密钥失败不该拖垮面板构造
        pass
    return default


class SketchPanel(QWidget):
    """手绘草图识别面板。"""

    model_loaded = Signal(dict)   # 识别成功并加载为当前模型时发出

    def __init__(self, session, runner, parent=None):
        super().__init__(parent)
        self.session = session
        self.runner = runner
        self._image_path: str | None = None
        self._source_image_path: str | None = None
        self._preprocess_metadata: dict | None = None
        self._work_plane: dict | None = None
        self._result_model: dict | None = None
        self._result_draft = None
        self._v2_draft: dict | None = None
        self._v2_state = None
        self._elapsed_timer = None
        self._elapsed = 0
        self._v2_node_reuse: dict[int, int] = {}
        self._highlight_refs: set[str] = set()
        self._drag_node_id: int | None = None
        self._drag_changed = False

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # 标题
        title = QLabel("手绘草图 → 模型")
        title.setStyleSheet(f"font-weight:650; font-size:10pt; color:{theme.INK};")
        layout.addWidget(title)

        # 不常改的提供商、模型和密钥默认收起，主流程只留图片与确认。
        self.btn_settings = QPushButton("识别设置 ▸")
        self.btn_settings.setCheckable(True)
        layout.addWidget(self.btn_settings)
        self.settings_widget = QWidget(self)
        settings_layout = QVBoxLayout(self.settings_widget)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.setSpacing(3)
        cfg_row = QHBoxLayout()
        self.cmb_provider = QComboBox()
        self.cmb_provider.addItems(["openai", "anthropic", "deepseek"])
        self.cmb_provider.setToolTip("多模态 LLM 提供商")
        self.txt_model = QLineEdit("gpt-4o")
        self.txt_model.setPlaceholderText("模型名")
        self.txt_model.setToolTip(
            "多模态模型名称，如 gpt-4o / claude-3-5-sonnet-20241022 / "
            "deepseek-v4-flash-vision-exp")
        self.txt_key = QLineEdit()
        self.txt_key.setPlaceholderText("API 密钥（不填读环境变量）")
        self.txt_key.setEchoMode(QLineEdit.EchoMode.Password)
        cfg_row.addWidget(QLabel("提供商"))
        cfg_row.addWidget(self.cmb_provider)
        cfg_row.addWidget(QLabel("模型"))
        cfg_row.addWidget(self.txt_model)
        settings_layout.addLayout(cfg_row)
        self.cmb_provider.currentTextChanged.connect(self._set_provider_defaults)
        # **必须在信号接好之后再选**：setCurrentText 早于 connect 的话，
        # _set_provider_defaults 根本不会被触发，模型名会一直停在构造时写死的
        # "gpt-4o"。于是提供商是 deepseek、模型名是 gpt-4o，DeepSeek 直接
        # 400 invalid_request_error——实测就是这么failed的。
        # 再显式同步一次，防止选中项恰好等于当前项时信号不发。
        self.cmb_provider.setCurrentText(_provider_with_credentials())
        self._set_provider_defaults(self.cmb_provider.currentText())

        key_row = QHBoxLayout()
        key_row.addWidget(QLabel("密钥"))
        key_row.addWidget(self.txt_key)
        self.spn_retries = QSpinBox()
        self.spn_retries.setRange(1, 5)
        self.spn_retries.setValue(3)
        self.spn_retries.setToolTip("自修复重试次数")
        key_row.addWidget(QLabel("重试"))
        key_row.addWidget(self.spn_retries)
        settings_layout.addLayout(key_row)
        self.settings_widget.setVisible(False)
        self.btn_settings.toggled.connect(self.settings_widget.setVisible)
        self.btn_settings.toggled.connect(
            lambda checked: self.btn_settings.setText(
                "识别设置 ▾" if checked else "识别设置 ▸"))
        layout.addWidget(self.settings_widget)

        # 图片选择和预览
        btn_row = QHBoxLayout()
        self.btn_pick = QPushButton("选择草图图片…")
        self.btn_pick.clicked.connect(self._pick_image)
        self.btn_recognize = QPushButton("识别图片")
        self.btn_recognize.clicked.connect(self._recognize)
        self.btn_recognize.setEnabled(False)
        self.btn_recognize.setStyleSheet(
            f"background:{theme.ACCENT}; color:white; font-weight:600;")
        btn_row.addWidget(self.btn_pick)
        btn_row.addWidget(self.btn_recognize)
        layout.addLayout(btn_row)

        self.preprocess_widget = ImagePreprocessWidget(
            Path.cwd() / ".multimodal_tmp", self)
        self.preprocess_widget.image_changed.connect(self._on_preprocessed_image)
        self.preprocess_widget.work_plane_confirmed.connect(
            self._on_work_plane_confirmed)
        self.preprocess_widget.setVisible(False)
        layout.addWidget(self.preprocess_widget)

        self.lbl_image = QLabel("未选择图片")
        self.lbl_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_image.setMinimumHeight(120)
        self.lbl_image.setStyleSheet(
            f"border:1px dashed {theme.BORDER}; color:{theme.INK_MUTED}; "
            f"background:{theme.PANEL_ALT};")
        self.lbl_image.installEventFilter(self)
        layout.addWidget(self.lbl_image)

        self.issue_panel = IssuePanel(self)
        self.issue_panel.setVisible(False)
        self.issue_panel.issue_selected.connect(self._on_v2_issue_selected)
        self.issue_panel.resolution_requested.connect(self._resolve_v2_issue)
        layout.addWidget(self.issue_panel)

        self.edit_widget = QWidget(self)
        edit_layout = QVBoxLayout(self.edit_widget)
        edit_layout.setContentsMargins(0, 0, 0, 0)
        edit_layout.setSpacing(3)
        edit_hint = QLabel("拖动蓝/橙色节点可修正位置；编辑后需重新确认")
        edit_hint.setStyleSheet(f"color:{theme.INK_MUTED}; font-size:8pt;")
        edit_layout.addWidget(edit_hint)
        member_row = QHBoxLayout()
        self.cmb_member_i = QComboBox()
        self.cmb_member_j = QComboBox()
        self.btn_add_member = QPushButton("补画杆件")
        self.btn_add_member.clicked.connect(self._add_member)
        self.cmb_remove_member = QComboBox()
        self.btn_remove_member = QPushButton("删除杆件")
        self.btn_remove_member.clicked.connect(self._remove_member)
        member_row.addWidget(self.cmb_member_i)
        member_row.addWidget(QLabel("→"))
        member_row.addWidget(self.cmb_member_j)
        member_row.addWidget(self.btn_add_member)
        member_row.addWidget(self.cmb_remove_member)
        member_row.addWidget(self.btn_remove_member)
        edit_layout.addLayout(member_row)
        self.edit_widget.setVisible(False)
        layout.addWidget(self.edit_widget)

        self.btn_boundaries = QPushButton("支座与节点荷载 ▸")
        self.btn_boundaries.setCheckable(True)
        self.btn_boundaries.setVisible(False)
        layout.addWidget(self.btn_boundaries)
        self.boundary_widget = QWidget(self)
        boundary_layout = QGridLayout(self.boundary_widget)
        boundary_layout.setContentsMargins(0, 0, 0, 0)
        boundary_layout.setSpacing(3)

        self.cmb_support_node = QComboBox()
        self.cmb_support_type = QComboBox()
        for label, fix in (
                ("固定端", [1, 1, 1, 1, 1, 1]),
                ("铰支座", [1, 1, 1, 0, 0, 0]),
                ("滚动-X", [0, 1, 1, 0, 0, 0]),
                ("滚动-Z", [1, 1, 0, 0, 0, 0])):
            self.cmb_support_type.addItem(label, fix)
        self.txt_support_name = QLineEdit()
        self.txt_support_name.setPlaceholderText("支座名称")
        self.btn_apply_support = QPushButton("应用支座")
        self.btn_remove_support = QPushButton("删除支座")
        self.btn_apply_support.clicked.connect(self._apply_support_edit)
        self.btn_remove_support.clicked.connect(self._remove_support_edit)
        self.cmb_support_node.currentIndexChanged.connect(self._fill_support_values)
        boundary_layout.addWidget(QLabel("支座"), 0, 0)
        boundary_layout.addWidget(self.cmb_support_node, 0, 1)
        boundary_layout.addWidget(self.cmb_support_type, 0, 2)
        boundary_layout.addWidget(self.txt_support_name, 0, 3, 1, 3)
        boundary_layout.addWidget(self.btn_apply_support, 1, 0, 1, 3)
        boundary_layout.addWidget(self.btn_remove_support, 1, 3, 1, 3)

        self.cmb_load_case = QComboBox()
        self.txt_new_case = QLineEdit()
        self.txt_new_case.setPlaceholderText("新工况名称")
        self.btn_add_case = QPushButton("新建工况")
        self.btn_add_case.clicked.connect(self._add_load_case_edit)
        self.cmb_load_case.currentTextChanged.connect(self._refresh_load_names)
        boundary_layout.addWidget(QLabel("工况"), 2, 0)
        boundary_layout.addWidget(self.cmb_load_case, 2, 1, 1, 2)
        boundary_layout.addWidget(self.txt_new_case, 2, 3, 1, 2)
        boundary_layout.addWidget(self.btn_add_case, 2, 5)

        self.cmb_load_node = QComboBox()
        self.cmb_load_name = QComboBox()
        self.cmb_load_name.setEditable(True)
        self.cmb_load_name.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.cmb_load_name.currentTextChanged.connect(self._fill_load_values)
        boundary_layout.addWidget(QLabel("节点荷载"), 3, 0)
        boundary_layout.addWidget(self.cmb_load_node, 3, 1)
        boundary_layout.addWidget(self.cmb_load_name, 3, 2, 1, 4)

        self.load_spins = []
        for index, label in enumerate(("Fx", "Fy", "Fz", "Mx", "My", "Mz")):
            spin = QDoubleSpinBox()
            spin.setRange(-1e12, 1e12)
            spin.setDecimals(3)
            spin.setPrefix(label + " ")
            self.load_spins.append(spin)
            boundary_layout.addWidget(spin, 4 + index // 2, 3 * (index % 2), 1, 3)
        self.btn_apply_load = QPushButton("应用节点荷载")
        self.btn_remove_load = QPushButton("删除节点荷载")
        self.btn_apply_load.clicked.connect(self._apply_nodal_load_edit)
        self.btn_remove_load.clicked.connect(self._remove_nodal_load_edit)
        boundary_layout.addWidget(self.btn_apply_load, 7, 0, 1, 3)
        boundary_layout.addWidget(self.btn_remove_load, 7, 3, 1, 3)

        self.boundary_widget.setVisible(False)
        self.btn_boundaries.toggled.connect(self.boundary_widget.setVisible)
        self.btn_boundaries.toggled.connect(
            lambda checked: self.btn_boundaries.setText(
                "支座与节点荷载 ▾" if checked else "支座与节点荷载 ▸"))
        layout.addWidget(self.boundary_widget)

        # 状态标签
        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet(f"color:{theme.INK_MUTED}; font-size:8pt;")
        layout.addWidget(self.lbl_status)

        # 结果预览
        self.txt_result = QPlainTextEdit()
        self.txt_result.setReadOnly(True)
        self.txt_result.setPlaceholderText("识别结果会显示在这里…")
        self.txt_result.setMaximumHeight(120)
        self.btn_details = QPushButton("识别详情与 JSON ▸")
        self.btn_details.setCheckable(True)
        self.btn_details.setVisible(False)
        self.btn_details.toggled.connect(self.txt_result.setVisible)
        self.btn_details.toggled.connect(
            lambda checked: self.btn_details.setText(
                "识别详情与 JSON ▾" if checked else "识别详情与 JSON ▸"))
        layout.addWidget(self.btn_details)
        self.txt_result.setVisible(False)
        layout.addWidget(self.txt_result)

        self.lbl_questions = QLabel("请逐项确认识别疑问：")
        self.lbl_questions.setStyleSheet(f"color:{theme.WARN}; font-size:8pt;")
        self.lbl_questions.setVisible(False)
        layout.addWidget(self.lbl_questions)
        self.lst_questions = QListWidget()
        self.lst_questions.setMaximumHeight(90)
        self.lst_questions.itemChanged.connect(self._update_load_enabled)
        self.lst_questions.setVisible(False)
        layout.addWidget(self.lst_questions)

        self.scale_widget = QWidget(self)
        scale_row = QHBoxLayout(self.scale_widget)
        scale_row.setContentsMargins(0, 0, 0, 0)
        self.cmb_reference_member = QComboBox()
        self.spn_reference_length = QDoubleSpinBox()
        self.spn_reference_length.setRange(0.001, 1_000_000.0)
        self.spn_reference_length.setDecimals(4)
        self.spn_reference_length.setValue(1.0)
        self.spn_reference_length.setSuffix(" m")
        self.btn_scale = QPushButton("应用尺度")
        self.btn_scale.clicked.connect(self._apply_scale)
        scale_row.addWidget(QLabel("参考杆件"))
        scale_row.addWidget(self.cmb_reference_member)
        scale_row.addWidget(QLabel("实际长度"))
        scale_row.addWidget(self.spn_reference_length)
        scale_row.addWidget(self.btn_scale)
        self.scale_widget.setVisible(False)
        layout.addWidget(self.scale_widget)

        self.chk_confirm = QCheckBox("我已核对覆盖标注和模型拓扑")
        self.chk_confirm.setEnabled(False)
        self.chk_confirm.toggled.connect(self._update_load_enabled)
        layout.addWidget(self.chk_confirm)

        # 加载按钮
        self.btn_load = QPushButton("确认并加载识别草稿")
        self.btn_load.clicked.connect(self._load_model)
        self.btn_load.setEnabled(False)
        self.btn_load.setStyleSheet(
            f"background:{theme.ACCENT}; color:white; font-weight:600;")
        layout.addWidget(self.btn_load)

        layout.addStretch()

    def _pick_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择手绘草图", "",
            "图片文件 (*.png *.jpg *.jpeg *.webp)")
        if not path:
            return
        self._source_image_path = path
        try:
            self.preprocess_widget.set_source(path)
        except (OSError, ValueError) as exc:
            self.lbl_status.setText(f"图片预处理失败：{exc}")
            return
        self.preprocess_widget.setVisible(True)

    def _on_preprocessed_image(self, metadata: dict) -> None:
        """Use only the derived image and invalidate every stale recognition result."""
        self._preprocess_metadata = dict(metadata)
        self._work_plane = None
        self._image_path = str(metadata["derived_path"])
        pixmap = QPixmap(self._image_path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(
                self.lbl_image.width(), 160,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            self.lbl_image.setPixmap(scaled)
            self.lbl_image.setText("")
        self.btn_recognize.setEnabled(False)
        self.btn_load.setEnabled(False)
        self._result_model = None
        self._result_draft = None
        self._v2_draft = None
        self._v2_state = None
        self._v2_node_reuse.clear()
        self.chk_confirm.setChecked(False)
        self.chk_confirm.setEnabled(False)
        self.scale_widget.setVisible(False)
        self.edit_widget.setVisible(False)
        self.btn_boundaries.setChecked(False)
        self.btn_boundaries.setVisible(False)
        self.lst_questions.clear()
        self.lst_questions.setVisible(False)
        self.lbl_questions.setVisible(False)
        self.btn_details.setChecked(False)
        self.btn_details.setVisible(False)
        self.txt_result.clear()
        self.lbl_status.setText(
            f"已载入派生图：{Path(self._image_path).name}；请确认工作平面")

    def _on_work_plane_confirmed(self, payload: dict) -> None:
        self._work_plane = dict(payload)
        self.btn_recognize.setEnabled(True)
        self.lbl_status.setText(
            f"工作平面已确认：{payload['plane']}，可开始识别")

    def set_v2_draft(self, draft: dict) -> None:
        """Host the frozen v2 review APIs without changing the legacy v1 path."""
        from copy import deepcopy
        self._v2_draft = deepcopy(draft)
        self._v2_node_reuse.clear()
        self._result_draft = None
        self.chk_confirm.setChecked(False)
        self.chk_confirm.setEnabled(False)
        self.btn_load.setEnabled(False)
        self.issue_panel.setVisible(True)
        self.issue_panel.set_draft(self._v2_draft)
        self._show_overlay(self._v2_draft)
        self._refresh_v2_scale_controls()
        self._refresh_editor_controls()
        self.edit_widget.setVisible(True)
        self._refresh_boundary_controls()
        self.btn_boundaries.setVisible(True)
        self._refresh_v2_commit_state()

    def _on_v2_issue_selected(self, issue_id: str, refs: list) -> None:
        self._highlight_refs = set(str(item) for item in refs)
        if self._v2_draft is not None:
            self._show_overlay(self._v2_draft)

    def _resolve_v2_issue(self, issue_id: str, action: str) -> None:
        if self._v2_draft is None:
            return
        from copy import deepcopy
        from sketch_topology import resolve_intersection
        draft = deepcopy(self._v2_draft)
        issue = next((item for item in draft.get("issues") or []
                      if str(item.get("id")) == issue_id), None)
        if issue is None:
            return
        category = issue.get("category")
        try:
            if category == "intersection_unknown" and action in {"connect", "cross"}:
                ref = next(ref for ref in issue.get("entity_refs") or []
                           if str(ref).startswith("intersection:"))
                draft = resolve_intersection(draft, str(ref).split(":", 1)[1], action)
            elif category == "work_plane_unconfirmed" and action == "confirm":
                draft["work_plane"]["status"] = "confirmed"
                issue.update(status="resolved", resolution="工作平面已确认",
                             resolved_by="user")
            elif category == "perspective" and action == "confirm":
                draft["source"].setdefault("preprocessing", {})[
                    "perspective_status"] = "accepted"
                issue.update(status="resolved", resolution="用户确认近似正射/已矫正",
                             resolved_by="user")
            elif category == "low_confidence" and action == "confirm":
                refs = set(issue.get("entity_refs") or [])
                for entity in draft.get("entities") or []:
                    if self._v2_entity_ref(entity) in refs:
                        entity.update(source="user", confidence=None, verified=True)
                issue.update(status="resolved", resolution="用户已核对实体",
                             resolved_by="user")
            elif category == "merge_collision" and action == "reuse":
                draft_id = int(issue["candidate_node"])
                target_id = int(issue["existing_node"])
                self._v2_node_reuse[draft_id] = target_id
                issue.update(status="resolved", resolution=f"复用现有节点 {target_id}",
                             resolved_by="user")
            else:
                raise ValueError("此问题必须通过对应编辑器处理，不能直接确认")
        except (StopIteration, ValueError) as exc:
            self._show_edit_error("问题处理失败", exc)
            return
        if category != "intersection_unknown":
            draft["revision"] = int(draft.get("revision", 0)) + 1
            draft["confirmation"] = None
            draft["merge_plan"] = None
        if self._v2_state is not None and self._v2_state.draft is not None:
            # Controller owns the authoritative revision. Topology helpers already
            # increment their standalone result, so normalize before edit_draft.
            draft["revision"] = int(self._v2_state.draft.get("revision", 0))
            self._v2_state.edit_draft(draft)
            draft = self._v2_state.draft
        self._v2_draft = draft
        self._highlight_refs.clear()
        self.issue_panel.set_draft(draft)
        self._show_overlay(draft)
        self._refresh_v2_scale_controls()
        self._refresh_v2_commit_state()

    def _refresh_v2_scale_controls(self) -> None:
        if self._v2_draft is None:
            return
        current = self.cmb_reference_member.currentData()
        self.cmb_reference_member.clear()
        for member in sorted(self._v2_draft.get("image_model", {}).get("members") or [],
                             key=lambda item: int(item["id"])):
            self.cmb_reference_member.addItem(f"杆件 {member['id']}", int(member["id"]))
        index = self.cmb_reference_member.findData(current)
        if index >= 0:
            self.cmb_reference_member.setCurrentIndex(index)
        self.scale_widget.setVisible(
            self._v2_draft.get("scale", {}).get("status") != "confirmed")

    def _refresh_v2_commit_state(self) -> None:
        if self._v2_draft is None or self._v2_state is None:
            return
        from copy import deepcopy
        from draft_commit import prepare_commit
        from sketch_topology import materialize_geometry

        draft = deepcopy(self._v2_draft)
        blockers = [item for item in draft.get("issues") or []
                    if item.get("severity") == "blocking"
                    and item.get("status") == "open"]
        perspective = (draft.get("source", {}).get("preprocessing", {})
                       .get("perspective_status"))
        geometry_ready = (
            draft.get("work_plane", {}).get("status") == "confirmed"
            and draft.get("scale", {}).get("status") == "confirmed"
            and perspective == "accepted")
        self.chk_confirm.setEnabled(False)
        self.btn_load.setEnabled(False)
        if not geometry_ready or blockers:
            return
        try:
            draft["model"] = materialize_geometry(draft)
            collisions = []
            for node in draft["model"].get("nodes") or []:
                point = (float(node["x"]), float(node["y"]), float(node["z"]))
                for existing in self.session.model.get("nodes") or []:
                    old = (float(existing["x"]), float(existing["y"]),
                           float(existing["z"]))
                    if math.dist(point, old) <= 1.0e-6:
                        pair = int(node["id"]), int(existing["id"])
                        if self._v2_node_reuse.get(pair[0]) != pair[1]:
                            collisions.append(pair)
                        break
            if collisions:
                known = {str(item.get("id")) for item in draft.get("issues") or []}
                for draft_id, existing_id in collisions:
                    identifier = f"merge-node-{draft_id}-{existing_id}"
                    if identifier not in known:
                        draft.setdefault("issues", []).append({
                            "id": identifier, "category": "merge_collision",
                            "severity": "blocking", "entity_refs": [f"node:{draft_id}"],
                            "message": (f"识别节点 {draft_id} 与现有节点 {existing_id} 重合；"
                                        "请明确复用现有节点。"),
                            "status": "open", "resolution": None, "resolved_by": None,
                            "candidate_node": draft_id, "existing_node": existing_id,
                        })
                draft["revision"] = int(self._v2_state.draft.get("revision", 0))
                self._v2_state.edit_draft(draft)
                self._v2_draft = self._v2_state.draft
                self.issue_panel.set_draft(self._v2_draft)
                self.lbl_status.setText("检测到与现有模型重合的节点，请逐项确认复用。")
                return
            prepared, preview = prepare_commit(
                draft, self.session.model, node_reuse=self._v2_node_reuse)
            self._v2_state.install_commit_preview(prepared, preview)
        except (KeyError, TypeError, ValueError) as exc:
            self.lbl_status.setText(f"提交预演失败：{exc}")
            self.lbl_status.setStyleSheet(f"color:{theme.WARN}; font-size:8pt;")
            return
        self._v2_draft = self._v2_state.draft
        self.issue_panel.set_draft(self._v2_draft)
        self.chk_confirm.setEnabled(True)
        self.btn_details.setVisible(True)
        import json
        self.txt_result.setPlainText(
            "结构化变更预演：\n" + json.dumps(
                preview["operations"], ensure_ascii=False, indent=2))
        self.lbl_status.setText(
            f"预演完成：{len(preview['operations'])} 项结构化变更，请核对后确认加载。")
        self.lbl_status.setStyleSheet(f"color:{theme.ACCENT}; font-size:8pt;")
        self._update_load_enabled()

    @staticmethod
    def _v2_entity_ref(entity: dict) -> str:
        target = entity.get("target") or {}
        if "node" in target:
            return f"node:{target['node']}"
        if "member" in target:
            return f"member:{target['member']}"
        support = target.get("support") or {}
        if support:
            return f"support:{support.get('node')}:{support.get('name')}"
        load = target.get("load") or {}
        if load:
            return f"load:{load.get('case')}:{load.get('collection')}:{load.get('name')}"
        return ""

    def _set_provider_defaults(self, provider: str):
        """切换提供商时给出实际可用的视觉模型默认值。

        **必须问 SketchParser 要，不能在这里再抄一份。** 之前这里有一份独立的
        映射，和 sketch_parser._default_model 各写各的；DeepSeek 的模型名更新后
        只改了那一处，面板仍然填 deepseek-vl，于是每次识别都报 model not found，
        表现出来像"多模态识别一直失败"。
        """
        from sketch_parser import SketchParser
        self.txt_model.setText(SketchParser._default_model(provider))

    def _recognize(self):
        if not self._image_path or self.runner.busy:
            return
        self.btn_recognize.setEnabled(False)
        self.btn_pick.setEnabled(False)
        self.btn_load.setEnabled(False)
        self._result_model = None
        self._result_draft = None
        self._v2_draft = None
        self._v2_state = None
        self._v2_node_reuse.clear()
        self.chk_confirm.setChecked(False)
        self.chk_confirm.setEnabled(False)
        self.scale_widget.setVisible(False)
        self.edit_widget.setVisible(False)
        self.btn_boundaries.setChecked(False)
        self.btn_boundaries.setVisible(False)
        self.lst_questions.clear()
        self.lst_questions.setVisible(False)
        self.lbl_questions.setVisible(False)
        self.btn_details.setChecked(False)
        self.btn_details.setVisible(False)
        self.txt_result.clear()
        self._start_elapsed_ticker()

        provider = self.cmb_provider.currentText()
        model = self.txt_model.text().strip()
        key = self.txt_key.text().strip()
        retries = self.spn_retries.value()
        image_path = self._image_path

        def _job():
            from sketch_parser import SketchParser
            from multimodal_workflow import MultimodalControllerState
            if key:
                parser = SketchParser(provider=provider, api_key=key, model=model)
            else:
                parser = SketchParser.from_env(provider, model=model)
            if self._preprocess_metadata:
                state = MultimodalControllerState()
                state.load_image(str(self._preprocess_metadata["image_hash"]))
                job_id = state.start_recognition()
                self._v2_state = state
                return parser.parse_v2_with_retry(
                    image_path, state, job_id, max_repairs=min(2, max(0, retries - 1)))
            return parser.parse_with_retry(image_path, max_retries=retries)

        self.runner.submit(_job, on_done=self._on_recognized, on_failed=self._on_failed)

    def _start_elapsed_ticker(self) -> None:
        """识别期间每秒刷新已用时。

        原来只有一句静止的"正在识别"。视觉模型读一张大图几十秒是正常的，
        但没有任何变化的界面里，"还在跑"和"已经死了"看起来一模一样——
        用户只能猜，然后去点关闭。一个秒表就能把这两件事分开。
        """
        from sketch_parser import REQUEST_TIMEOUT_SECONDS
        self._elapsed = 0
        self._timeout_hint = int(REQUEST_TIMEOUT_SECONDS)

        def tick():
            self._elapsed += 1
            self.lbl_status.setText(
                f"多模态 LLM 正在识别…… 已用 {self._elapsed}s"
                f"（单次调用上限 {self._timeout_hint}s）")

        if self._elapsed_timer is None:
            from PySide6.QtCore import QTimer
            self._elapsed_timer = QTimer(self)
            self._elapsed_timer.timeout.connect(tick)
        else:
            self._elapsed_timer.timeout.disconnect()
            self._elapsed_timer.timeout.connect(tick)
        tick()
        self._elapsed_timer.start(1000)

    def _stop_elapsed_ticker(self) -> None:
        if self._elapsed_timer is not None:
            self._elapsed_timer.stop()

    def _on_recognized(self, result):
        self._stop_elapsed_ticker()
        self.btn_recognize.setEnabled(True)
        self.btn_pick.setEnabled(True)
        if result.success:
            if isinstance(result.draft, dict):
                from copy import deepcopy
                from dimension_constraints import apply_scale_to_draft
                from sketch_topology import detect_topology
                draft = deepcopy(result.draft)
                if self._preprocess_metadata:
                    draft["source"] = deepcopy(self._preprocess_metadata)
                if self._work_plane:
                    draft["work_plane"] = deepcopy(self._work_plane)
                draft = apply_scale_to_draft(draft)
                draft = detect_topology(draft)
                self._add_v2_review_issues(draft)
                if self._v2_state is not None:
                    self._v2_state.edit_draft(draft)
                    draft = self._v2_state.draft
                self.set_v2_draft(draft)
                blockers = sum(item.get("severity") == "blocking"
                               and item.get("status") == "open"
                               for item in draft.get("issues") or [])
                self.lbl_status.setText(
                    f"识别完成：请逐项处理 {blockers} 个阻断问题")
                return
            from recognition_draft import RecognitionDraft
            draft = result.draft or RecognitionDraft.from_payload(
                result.model, source_image=self._image_path or "")
            self._result_draft = draft
            self._result_model = draft.model
            n_nodes = len(draft.model.get("nodes", []))
            n_members = len(draft.model.get("members", []))
            n_supports = len(draft.model.get("supports", []))
            scale = ("尺度已确认" if draft.scale_status == "confirmed"
                     else "尺度待标定")
            self.lbl_status.setText(
                f"✅ 识别成功（{result.attempts} 次尝试）："
                f"{n_nodes} 节点 · {n_members} 杆件 · {n_supports} 支座 · {scale} · "
                f"{len(draft.warnings)} 个警告")
            self.lbl_status.setStyleSheet(f"color:{theme.ACCENT}; font-size:8pt;")
            self._refresh_result_text()
            self.btn_details.setVisible(True)
            self.chk_confirm.setEnabled(True)
            self._populate_questions()
            self._refresh_editor_controls()
            self._refresh_boundary_controls()
            self.edit_widget.setVisible(True)
            self.btn_boundaries.setVisible(True)
            self.scale_widget.setVisible(draft.scale_status == "unknown")
            self._show_overlay(draft)
            self._update_load_enabled()
        else:
            self.lbl_status.setText(f"❌ 识别失败（{result.attempts} 次尝试）")
            self.lbl_status.setStyleSheet(f"color:{theme.WARN}; font-size:8pt;")
            dump = self._dump_failure(result)
            text = "\n".join(result.errors)
            if dump:
                text += f"\n\n原始响应已写入：{dump}"
            self.txt_result.setPlainText(text)

    def _dump_failure(self, result) -> str:
        """把失败的错误清单与**模型原始响应**落盘。

        识别失败时界面上只有一句结论，模型到底吐了什么没人看得到，
        于是"格式不合法"和"根本没调通"分不开，只能靠猜。原始响应可能很长
        且含图片描述，所以写文件而不是塞进界面。

        写到 .multimodal_tmp/（已被 .gitignore 忽略）。**不含密钥**——
        密钥从不进入 SketchParser 的返回值。
        """
        import json
        from datetime import datetime
        try:
            root = Path(__file__).resolve().parent.parent
            out_dir = root / ".multimodal_tmp"
            out_dir.mkdir(exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = out_dir / f"recognition_failure_{stamp}.json"
            path.write_text(json.dumps({
                "provider": self.cmb_provider.currentText(),
                "model": self.txt_model.text(),
                "attempts": getattr(result, "attempts", None),
                "errors": list(getattr(result, "errors", []) or []),
                "raw_responses": list(getattr(result, "raw_responses", []) or []),
            }, ensure_ascii=False, indent=1), encoding="utf-8")
            return path.name
        except Exception:      # 诊断落盘失败绝不能盖住原本的识别错误
            return ""

    def _add_v2_review_issues(self, draft: dict) -> None:
        issues = draft.setdefault("issues", [])
        open_keys = {(item.get("category"), tuple(item.get("entity_refs") or []))
                     for item in issues if item.get("status") == "open"}
        perspective = (draft.get("source", {}).get("preprocessing", {})
                       .get("perspective_status"))
        additions = []
        if perspective != "accepted":
            additions.append(("perspective", [], "请确认图片近似正射或已经矫正"))
        for entity in draft.get("entities") or []:
            confidence = entity.get("confidence")
            if (not entity.get("verified") and
                    (entity.get("source") == "migration" or
                     entity.get("source") == "vision" and confidence is not None
                     and float(confidence) < 0.75)):
                ref = self._v2_entity_ref(entity)
                if ref:
                    additions.append(("low_confidence", [ref], f"低置信度实体：{ref}"))
        for index, (category, refs, message) in enumerate(additions, 1):
            key = category, tuple(refs)
            if key in open_keys:
                continue
            issues.append({"id": f"ui-{category}-{index}", "category": category,
                           "severity": "blocking", "entity_refs": refs,
                           "message": message, "status": "open",
                           "resolution": None, "resolved_by": None})
            open_keys.add(key)
    def _on_failed(self, exc_type, msg):
        self._stop_elapsed_ticker()
        self.btn_recognize.setEnabled(True)
        self.btn_pick.setEnabled(True)
        self.lbl_status.setText(f"❌ 调用失败：{exc_type}: {msg}")
        self.lbl_status.setStyleSheet(f"color:{theme.WARN}; font-size:8pt;")

    def _update_load_enabled(self, *_):
        if self._v2_draft is not None:
            ready = bool(self._v2_state and
                         self._v2_state.can_commit(self.session.model))
            self.btn_load.setEnabled(ready and self.chk_confirm.isChecked())
            return
        ready = bool(self._result_draft and self._result_draft.ready_to_load)
        questions_ok = all(
            self.lst_questions.item(row).checkState() == Qt.CheckState.Checked
            for row in range(self.lst_questions.count()))
        self.btn_load.setEnabled(
            ready and questions_ok and self.chk_confirm.isChecked())

    def _populate_questions(self):
        self.lst_questions.blockSignals(True)
        self.lst_questions.clear()
        for question in self._result_draft.questions:
            item = QListWidgetItem(question)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.lst_questions.addItem(item)
        self.lst_questions.blockSignals(False)
        self.lst_questions.setVisible(self.lst_questions.count() > 0)
        self.lbl_questions.setVisible(self.lst_questions.count() > 0)

    def _refresh_result_text(self):
        if self._result_draft is None:
            return
        import json
        review = []
        if self._result_draft.warnings:
            review.append("警告：\n- " + "\n- ".join(self._result_draft.warnings))
        review.append("当前识别草稿：\n" + json.dumps(
            self._result_draft.model, ensure_ascii=False, indent=2))
        self.txt_result.setPlainText("\n\n".join(review))

    def _refresh_editor_controls(self):
        if self._v2_draft is not None:
            model = self._v2_draft.get("image_model") or {}
        elif self._result_draft is not None:
            model = self._result_draft.model
        else:
            return
        nodes = model.get("nodes") or []
        members = model.get("members") or []
        for combo in (self.cmb_member_i, self.cmb_member_j):
            current = combo.currentData()
            combo.clear()
            for node in nodes:
                combo.addItem(f"节点 {node['id']}", int(node["id"]))
            if current is not None:
                combo.setCurrentIndex(max(0, combo.findData(current)))
        if self.cmb_member_j.count() > 1 and self.cmb_member_j.currentIndex() == 0:
            self.cmb_member_j.setCurrentIndex(1)
        for combo in (self.cmb_remove_member, self.cmb_reference_member):
            current = combo.currentData()
            combo.clear()
            for member in members:
                combo.addItem(f"杆件 {member['id']}", int(member["id"]))
            if current is not None:
                combo.setCurrentIndex(max(0, combo.findData(current)))
        self.btn_remove_member.setEnabled(len(members) > 1)

    def _reset_confirmation(self):
        self.chk_confirm.setChecked(False)
        self._update_load_enabled()

    def _add_member(self):
        if self._v2_draft is not None:
            from copy import deepcopy
            from sketch_topology import detect_topology
            draft = deepcopy(self._v2_draft)
            i, j = int(self.cmb_member_i.currentData()), int(self.cmb_member_j.currentData())
            if i == j:
                self._show_edit_error("补画失败", ValueError("两端节点不能相同"))
                return
            members = draft["image_model"].setdefault("members", [])
            edge = tuple(sorted((i, j)))
            if any(tuple(sorted((int(item["i"]), int(item["j"])))) == edge
                   for item in members):
                self._show_edit_error("补画失败", ValueError("该杆件已经存在"))
                return
            member_id = max((int(item["id"]) for item in members), default=0) + 1
            members.append({"id": member_id, "i": i, "j": j,
                            "material": None, "section": None})
            nodes = {int(item["id"]): item for item in draft["image_model"]["nodes"]}
            draft.setdefault("entities", []).append({
                "kind": "member", "id": f"member-{member_id}", "source": "user",
                "confidence": None, "recognition_confidence": None, "verified": True,
                "image_geometry": {"line": [
                    [float(nodes[i]["u"]), float(nodes[i]["v"])],
                    [float(nodes[j]["u"]), float(nodes[j]["v"])]],
                }, "target": {"member": member_id}, "payload": {},
            })
            draft = detect_topology(draft)
            self._apply_v2_edit(draft, f"已补画杆件 {member_id}")
            return
        if self._result_draft is None:
            return
        try:
            member_id = self._result_draft.add_member(
                int(self.cmb_member_i.currentData()),
                int(self.cmb_member_j.currentData()))
        except (TypeError, ValueError) as exc:
            self.lbl_status.setText(f"补画失败：{exc}")
            self.lbl_status.setStyleSheet(f"color:{theme.WARN}; font-size:8pt;")
            return
        self._after_geometry_edit(f"已补画杆件 {member_id}")

    def _remove_member(self):
        if self._v2_draft is not None:
            from sketch_topology import delete_member
            member_id = int(self.cmb_remove_member.currentData())
            try:
                draft = delete_member(self._v2_draft, member_id)
            except ValueError as exc:
                self._show_edit_error("删除失败", exc)
                return
            self._apply_v2_edit(draft, f"已删除杆件 {member_id}")
            return
        if self._result_draft is None:
            return
        member_id = self.cmb_remove_member.currentData()
        try:
            self._result_draft.remove_member(int(member_id))
        except (TypeError, ValueError) as exc:
            self.lbl_status.setText(f"删除失败：{exc}")
            self.lbl_status.setStyleSheet(f"color:{theme.WARN}; font-size:8pt;")
            return
        self._after_geometry_edit(f"已删除杆件 {member_id}")

    def _after_geometry_edit(self, message: str):
        self._result_model = self._result_draft.model
        self._reset_confirmation()
        self._refresh_editor_controls()
        self._refresh_boundary_controls()
        self._refresh_result_text()
        self._show_overlay(self._result_draft)
        self.lbl_status.setText(message + "，请重新核对后确认。")
        self.lbl_status.setStyleSheet(f"color:{theme.ACCENT}; font-size:8pt;")

    def _apply_v2_edit(self, draft: dict, message: str) -> None:
        self._v2_node_reuse.clear()
        draft["model"] = draft["merge_plan"] = draft["confirmation"] = None
        if self._v2_state is not None and self._v2_state.draft is not None:
            draft["revision"] = int(self._v2_state.draft.get("revision", 0))
            self._v2_state.edit_draft(draft)
            draft = self._v2_state.draft
        else:
            draft["revision"] = int(draft.get("revision", 0)) + 1
        self._v2_draft = draft
        self.chk_confirm.setChecked(False)
        self.issue_panel.set_draft(draft)
        self._show_overlay(draft)
        self._refresh_editor_controls()
        self._refresh_boundary_controls()
        self._refresh_v2_scale_controls()
        self._refresh_v2_commit_state()
        self.lbl_status.setText(message + "，请重新核对后确认。")
        self.lbl_status.setStyleSheet(f"color:{theme.ACCENT}; font-size:8pt;")

    def _refresh_boundary_controls(self):
        if self._v2_draft is not None:
            model = self._v2_draft.get("image_model") or {}
        elif self._result_draft is not None:
            model = self._result_draft.model
        else:
            return
        nodes = model.get("nodes") or []
        for combo in (self.cmb_support_node, self.cmb_load_node):
            current = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            for node in nodes:
                combo.addItem(f"节点 {node['id']}", int(node["id"]))
            if current is not None and combo.findData(current) >= 0:
                combo.setCurrentIndex(combo.findData(current))
            combo.blockSignals(False)
        current_case = self.cmb_load_case.currentText()
        self.cmb_load_case.blockSignals(True)
        self.cmb_load_case.clear()
        for case in model.get("load_cases") or []:
            self.cmb_load_case.addItem(str(case.get("name", "")))
        if current_case and self.cmb_load_case.findText(current_case) >= 0:
            self.cmb_load_case.setCurrentText(current_case)
        self.cmb_load_case.blockSignals(False)
        self._fill_support_values()
        self._refresh_load_names()
        has_case = self.cmb_load_case.count() > 0
        self.cmb_load_node.setEnabled(has_case)
        self.cmb_load_name.setEnabled(has_case)
        self.btn_apply_load.setEnabled(has_case)
        self.btn_remove_load.setEnabled(has_case)

    def _fill_support_values(self, *_):
        if ((self._result_draft is None and self._v2_draft is None)
                or self.cmb_support_node.currentData() is None):
            return
        node_id = int(self.cmb_support_node.currentData())
        model = (self._v2_draft.get("image_model") if self._v2_draft is not None
                 else self._result_draft.model)
        support = next((item for item in model.get("supports", [])
                        if int(item["node"]) == node_id), None)
        if support is None:
            self.txt_support_name.setText(f"BC-Node-{node_id}")
            self.cmb_support_type.setCurrentIndex(0)
            self.btn_remove_support.setEnabled(False)
            return
        self.txt_support_name.setText(str(support.get("name") or f"BC-Node-{node_id}"))
        index = self.cmb_support_type.findData(support["fix"])
        self.cmb_support_type.setCurrentIndex(max(0, index))
        self.btn_remove_support.setEnabled(True)

    def _refresh_load_names(self, *_):
        if self._v2_draft is not None:
            model = self._v2_draft.get("image_model") or {}
        elif self._result_draft is not None:
            model = self._result_draft.model
        else:
            return
        case_name = self.cmb_load_case.currentText()
        case = next((item for item in model.get("load_cases", [])
                     if str(item.get("name", "")) == case_name), None)
        current = self.cmb_load_name.currentText()
        self.cmb_load_name.blockSignals(True)
        self.cmb_load_name.clear()
        for entry in (case or {}).get("nodal_loads", []):
            self.cmb_load_name.addItem(str(entry.get("name", "")))
        if current and self.cmb_load_name.findText(current) >= 0:
            self.cmb_load_name.setCurrentText(current)
        elif self.cmb_load_name.count() == 0:
            node_id = self.cmb_load_node.currentData()
            self.cmb_load_name.setEditText(f"CF-Node-{node_id}" if node_id else "CF-1")
        self.cmb_load_name.blockSignals(False)
        self._fill_load_values()

    def _fill_load_values(self, *_):
        if self._v2_draft is not None:
            model = self._v2_draft.get("image_model") or {}
        elif self._result_draft is not None:
            model = self._result_draft.model
        else:
            return
        case_name = self.cmb_load_case.currentText()
        load_name = self.cmb_load_name.currentText().strip()
        case = next((item for item in model.get("load_cases", [])
                     if str(item.get("name", "")) == case_name), None)
        entry = next((item for item in (case or {}).get("nodal_loads", [])
                      if str(item.get("name", "")) == load_name), None)
        values = entry.get("load", [0.0] * 6) if entry else [0.0] * 6
        if entry and self.cmb_load_node.findData(int(entry["node"])) >= 0:
            self.cmb_load_node.setCurrentIndex(
                self.cmb_load_node.findData(int(entry["node"])))
        for spin, value in zip(self.load_spins, values):
            spin.setValue(float(value))
        self.btn_remove_load.setEnabled(entry is not None)

    def _resolve_questions(self, keyword: str):
        for row in range(self.lst_questions.count()):
            item = self.lst_questions.item(row)
            if keyword in item.text():
                item.setCheckState(Qt.CheckState.Checked)

    def _apply_support_edit(self):
        if self._v2_draft is not None:
            from copy import deepcopy
            draft = deepcopy(self._v2_draft)
            node_id = int(self.cmb_support_node.currentData())
            name = self.txt_support_name.text().strip() or f"BC-Node-{node_id}"
            fix = list(self.cmb_support_type.currentData())
            supports = [item for item in draft["image_model"].get("supports") or []
                        if int(item["node"]) != node_id]
            support = {"name": name, "node": node_id, "fix": fix}
            supports.append(support)
            draft["image_model"]["supports"] = supports
            draft["entities"] = [item for item in draft.get("entities") or []
                                 if not (item.get("kind") == "support" and
                                         int((item.get("target", {}).get("support") or {})
                                             .get("node", -1)) == node_id)]
            node = next(item for item in draft["image_model"]["nodes"]
                        if int(item["id"]) == node_id)
            draft["entities"].append({
                "kind": "support", "id": f"support-{node_id}-{name}",
                "source": "user", "confidence": None,
                "recognition_confidence": None, "verified": True,
                "image_geometry": {"point": [float(node["u"]), float(node["v"])]},
                "target": {"support": {"node": node_id, "name": name}},
                "payload": deepcopy(support),
            })
            self._apply_v2_edit(draft, f"已更新节点 {node_id} 的支座")
            return
        if self._result_draft is None:
            return
        node_id = self.cmb_support_node.currentData()
        try:
            self._result_draft.set_support(
                int(node_id), list(self.cmb_support_type.currentData()),
                self.txt_support_name.text())
        except (TypeError, ValueError) as exc:
            self._show_edit_error("支座设置失败", exc)
            return
        self._resolve_questions("支座")
        self._after_geometry_edit(f"已更新节点 {node_id} 的支座")

    def _remove_support_edit(self):
        if self._v2_draft is not None:
            from copy import deepcopy
            draft = deepcopy(self._v2_draft)
            node_id = int(self.cmb_support_node.currentData())
            before = len(draft["image_model"].get("supports") or [])
            draft["image_model"]["supports"] = [
                item for item in draft["image_model"].get("supports") or []
                if int(item["node"]) != node_id]
            if len(draft["image_model"]["supports"]) == before:
                self._show_edit_error("删除支座失败", ValueError("该节点没有支座"))
                return
            draft["entities"] = [item for item in draft.get("entities") or []
                                 if not (item.get("kind") == "support" and
                                         int((item.get("target", {}).get("support") or {})
                                             .get("node", -1)) == node_id)]
            self._apply_v2_edit(draft, f"已删除节点 {node_id} 的支座")
            return
        if self._result_draft is None:
            return
        node_id = self.cmb_support_node.currentData()
        try:
            self._result_draft.remove_support(int(node_id))
        except (TypeError, ValueError) as exc:
            self._show_edit_error("删除支座失败", exc)
            return
        self._resolve_questions("支座")
        self._after_geometry_edit(f"已删除节点 {node_id} 的支座")

    def _add_load_case_edit(self):
        if self._v2_draft is not None:
            from copy import deepcopy
            name = self.txt_new_case.text().strip()
            if not name or any(str(item.get("name")) == name
                               for item in self._v2_draft["image_model"].get("load_cases") or []):
                self._show_edit_error("新建工况失败", ValueError("工况名称为空或重复"))
                return
            draft = deepcopy(self._v2_draft)
            draft["image_model"].setdefault("load_cases", []).append(
                {"name": name, "nodal_loads": []})
            self.txt_new_case.clear()
            self._apply_v2_edit(draft, f"已新建工况 {name}")
            self.cmb_load_case.setCurrentText(name)
            return
        if self._result_draft is None:
            return
        try:
            self._result_draft.add_load_case(self.txt_new_case.text())
        except ValueError as exc:
            self._show_edit_error("新建工况失败", exc)
            return
        name = self.txt_new_case.text().strip()
        self.txt_new_case.clear()
        self._after_geometry_edit(f"已新建工况 {name}")
        self.cmb_load_case.setCurrentText(name)

    def _apply_nodal_load_edit(self):
        if self._v2_draft is not None:
            from copy import deepcopy
            draft = deepcopy(self._v2_draft)
            case_name = self.cmb_load_case.currentText()
            load_name = self.cmb_load_name.currentText().strip()
            node_id = int(self.cmb_load_node.currentData())
            if not load_name:
                self._show_edit_error("节点荷载设置失败", ValueError("荷载名称不能为空"))
                return
            case = next((item for item in draft["image_model"].get("load_cases") or []
                         if str(item.get("name")) == case_name), None)
            if case is None:
                self._show_edit_error("节点荷载设置失败", ValueError("请先创建工况"))
                return
            values = [spin.value() for spin in self.load_spins]
            loads = [item for item in case.get("nodal_loads") or []
                     if str(item.get("name")) != load_name]
            payload = {"name": load_name, "node": node_id, "load": values,
                       "units": ["N", "N", "N", "N*m", "N*m", "N*m"]}
            loads.append(payload)
            case["nodal_loads"] = loads
            draft["entities"] = [item for item in draft.get("entities") or []
                                 if not (item.get("kind") == "load" and
                                         (item.get("target", {}).get("load") or {}).get("case") == case_name
                                         and (item.get("target", {}).get("load") or {}).get("collection") == "nodal_loads"
                                         and (item.get("target", {}).get("load") or {}).get("name") == load_name)]
            node = next(item for item in draft["image_model"]["nodes"]
                        if int(item["id"]) == node_id)
            draft["entities"].append({
                "kind": "load", "id": f"load-{case_name}-{load_name}", "source": "user",
                "confidence": None, "recognition_confidence": None, "verified": True,
                "image_geometry": {"point": [float(node["u"]), float(node["v"])]},
                "target": {"load": {"case": case_name, "collection": "nodal_loads",
                                      "name": load_name}}, "payload": deepcopy(payload),
            })
            self._apply_v2_edit(draft, f"已更新节点荷载 {load_name}")
            return
        if self._result_draft is None:
            return
        try:
            self._result_draft.set_nodal_load(
                self.cmb_load_case.currentText(), int(self.cmb_load_node.currentData()),
                [spin.value() for spin in self.load_spins],
                self.cmb_load_name.currentText())
        except (TypeError, ValueError) as exc:
            self._show_edit_error("节点荷载设置失败", exc)
            return
        self._resolve_questions("荷载")
        self._after_geometry_edit(
            f"已更新节点荷载 {self.cmb_load_name.currentText().strip()}")

    def _remove_nodal_load_edit(self):
        if self._v2_draft is not None:
            from copy import deepcopy
            draft = deepcopy(self._v2_draft)
            case_name = self.cmb_load_case.currentText()
            load_name = self.cmb_load_name.currentText().strip()
            case = next((item for item in draft["image_model"].get("load_cases") or []
                         if str(item.get("name")) == case_name), None)
            before = len((case or {}).get("nodal_loads") or [])
            if case is None:
                return
            case["nodal_loads"] = [item for item in case.get("nodal_loads") or []
                                   if str(item.get("name")) != load_name]
            if len(case["nodal_loads"]) == before:
                self._show_edit_error("删除节点荷载失败", ValueError("荷载不存在"))
                return
            draft["entities"] = [item for item in draft.get("entities") or []
                                 if not (item.get("kind") == "load" and
                                         (item.get("target", {}).get("load") or {}).get("case") == case_name
                                         and (item.get("target", {}).get("load") or {}).get("collection") == "nodal_loads"
                                         and (item.get("target", {}).get("load") or {}).get("name") == load_name)]
            self._apply_v2_edit(draft, f"已删除节点荷载 {load_name}")
            return
        if self._result_draft is None:
            return
        case_name = self.cmb_load_case.currentText()
        load_name = self.cmb_load_name.currentText().strip()
        try:
            self._result_draft.remove_nodal_load(case_name, load_name)
        except ValueError as exc:
            self._show_edit_error("删除节点荷载失败", exc)
            return
        self._resolve_questions("荷载")
        self._after_geometry_edit(f"已删除节点荷载 {load_name}")

    def _show_edit_error(self, prefix: str, exc: Exception):
        self.lbl_status.setText(f"{prefix}：{exc}")
        self.lbl_status.setStyleSheet(f"color:{theme.WARN}; font-size:8pt;")

    def _show_overlay(self, draft):
        """把模型给出的归一化实体位置叠到原图上，供人工核对。"""
        if not self._image_path:
            return
        pixmap = QPixmap(self._image_path)
        if pixmap.isNull():
            return
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        def point(values):
            if not isinstance(values, (list, tuple)) or len(values) != 2:
                return None
            try:
                x, y = float(values[0]), float(values[1])
            except (TypeError, ValueError):
                return None
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                return None
            return x * pixmap.width(), y * pixmap.height()

        entities = draft.get("entities", []) if isinstance(draft, dict) else draft.entities
        for entity in entities:
            geometry = entity.get("image_geometry") or {}
            raw_confidence = entity.get("confidence")
            confidence = 1.0 if raw_confidence is None else float(raw_confidence)
            if self._v2_entity_ref(entity) in self._highlight_refs:
                colour = QColor("#d946ef")
            elif confidence < 0.75:
                colour = QColor(theme.WARN)
            elif entity.get("kind") == "support":
                colour = QColor(theme.DRAFT_SUPPORT)
            elif entity.get("kind") == "load":
                colour = QColor(theme.LOAD)
            else:
                colour = QColor(theme.ACCENT)
            painter.setPen(QPen(colour, max(2, pixmap.width() // 300)))
            line = geometry.get("line")
            if isinstance(line, list) and len(line) == 2:
                start, end = point(line[0]), point(line[1])
                if start and end:
                    painter.drawLine(int(start[0]), int(start[1]),
                                     int(end[0]), int(end[1]))
            where = point(geometry.get("point"))
            if where:
                radius = max(4, pixmap.width() // 150)
                painter.setBrush(colour)
                painter.drawEllipse(int(where[0] - radius), int(where[1] - radius),
                                    radius * 2, radius * 2)
                painter.drawText(int(where[0] + radius + 2), int(where[1] - radius),
                                 str(entity.get("id", "")))
            bounds = geometry.get("bbox")
            if isinstance(bounds, list) and len(bounds) == 4:
                top_left = point(bounds[:2])
                bottom_right = point(bounds[2:])
                if top_left and bottom_right:
                    left, right = sorted((int(top_left[0]), int(bottom_right[0])))
                    top, bottom = sorted((int(top_left[1]), int(bottom_right[1])))
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawRect(left, top, right - left, bottom - top)
                    label = f"{entity.get('kind', '')} {entity.get('id', '')}".strip()
                    painter.drawText(left + 2, max(12, top - 3), label)
        painter.end()
        scaled = pixmap.scaled(
            self.lbl_image.width(), 200,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        self.lbl_image.setPixmap(scaled)
        self.lbl_image.setText("")
        self.lbl_image.setCursor(Qt.CursorShape.OpenHandCursor)

    def _image_position(self, event) -> tuple[float, float] | None:
        pixmap = self.lbl_image.pixmap()
        if pixmap is None or pixmap.isNull():
            return None
        left = (self.lbl_image.width() - pixmap.width()) / 2.0
        top = (self.lbl_image.height() - pixmap.height()) / 2.0
        pos = event.position()
        u = (pos.x() - left) / pixmap.width()
        v = (pos.y() - top) / pixmap.height()
        if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
            return None
        return u, v

    def _move_v2_node_preview(self, node_id: int, u: float, v: float) -> None:
        """Update all image evidence during drag; controller changes only on release."""
        if self._v2_draft is None or not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
            raise ValueError("节点图片坐标必须在 [0,1] 内")
        nodes = {int(item["id"]): item
                 for item in self._v2_draft.get("image_model", {}).get("nodes") or []}
        if node_id not in nodes:
            raise ValueError(f"节点 {node_id} 不存在")
        nodes[node_id]["u"], nodes[node_id]["v"] = float(u), float(v)
        for entity in self._v2_draft.get("entities") or []:
            target = entity.get("target") or {}
            if entity.get("kind") == "node" and int(target.get("node", -1)) == node_id:
                entity["image_geometry"] = {"point": [float(u), float(v)]}
                entity.update(source="user", confidence=None, verified=True)
            if entity.get("kind") != "member" or "member" not in target:
                continue
            member_id = int(target["member"])
            member = next((item for item in self._v2_draft["image_model"].get("members") or []
                           if int(item["id"]) == member_id), None)
            if member is None or node_id not in (int(member["i"]), int(member["j"])):
                continue
            a, b = nodes[int(member["i"])], nodes[int(member["j"])]
            entity["image_geometry"] = {
                "line": [[float(a["u"]), float(a["v"])],
                         [float(b["u"]), float(b["v"])]]}
        self._v2_draft["model"] = self._v2_draft["merge_plan"] = None
        self._v2_draft["confirmation"] = None

    def _finish_v2_node_move(self, node_id: int) -> None:
        from dimension_constraints import apply_scale_to_draft
        from sketch_topology import detect_topology

        draft = apply_scale_to_draft(detect_topology(self._v2_draft))
        self._v2_node_reuse.clear()
        if self._v2_state is not None and self._v2_state.draft is not None:
            draft["revision"] = int(self._v2_state.draft.get("revision", 0))
            self._v2_state.edit_draft(draft)
            draft = self._v2_state.draft
        else:
            draft["revision"] = int(draft.get("revision", 0)) + 1
        self._v2_draft = draft
        self.issue_panel.set_draft(draft)
        self._show_overlay(draft)
        self._refresh_v2_scale_controls()
        self._refresh_v2_commit_state()
        self.lbl_status.setText(f"已移动节点 {node_id}，尺寸与交点已重新计算。")

    def eventFilter(self, watched, event):
        active = self._v2_draft is not None or self._result_draft is not None
        if watched is self.lbl_image and active:
            if (event.type() == QEvent.Type.MouseButtonPress
                    and event.button() == Qt.MouseButton.LeftButton):
                point = self._image_position(event)
                if point is not None:
                    nearest = None
                    entities = (self._v2_draft.get("entities") or []
                                if self._v2_draft is not None
                                else self._result_draft.entities)
                    for entity in entities:
                        if entity.get("kind") != "node":
                            continue
                        if self._v2_draft is not None:
                            where = (entity.get("image_geometry") or {}).get("point")
                            node_id = (entity.get("target") or {}).get("node")
                        else:
                            where = self._result_draft._image_point(entity)
                            node_id = entity.get("id")
                        if where is None:
                            continue
                        distance = ((where[0] - point[0]) ** 2
                                    + (where[1] - point[1]) ** 2) ** 0.5
                        if nearest is None or distance < nearest[0]:
                            nearest = distance, int(node_id)
                    pixmap = self.lbl_image.pixmap()
                    threshold = 14.0 / max(1, min(pixmap.width(), pixmap.height()))
                    if nearest is not None and nearest[0] <= threshold:
                        self._drag_node_id = nearest[1]
                        self._drag_changed = False
                        self.lbl_image.setCursor(Qt.CursorShape.ClosedHandCursor)
                        return True
            elif (event.type() == QEvent.Type.MouseMove
                  and self._drag_node_id is not None):
                point = self._image_position(event)
                if point is not None:
                    try:
                        if self._v2_draft is not None:
                            self._move_v2_node_preview(
                                self._drag_node_id, point[0], point[1])
                        else:
                            self._result_draft.move_node_image(
                                self._drag_node_id, point[0], point[1])
                    except ValueError as exc:
                        self.lbl_status.setText(f"拖动失败：{exc}")
                    else:
                        self._drag_changed = True
                        self._result_model = (None if self._v2_draft is not None
                                              else self._result_draft.model)
                        self._reset_confirmation()
                        self._show_overlay(self._v2_draft or self._result_draft)
                        self.lbl_status.setText(f"正在移动节点 {self._drag_node_id}…")
                return True
            elif event.type() == QEvent.Type.MouseButtonRelease:
                moved_node = self._drag_node_id
                self._drag_node_id = None
                self.lbl_image.setCursor(Qt.CursorShape.OpenHandCursor)
                if self._drag_changed and moved_node is not None:
                    if self._v2_draft is not None:
                        try:
                            self._finish_v2_node_move(moved_node)
                        except ValueError as exc:
                            self._show_edit_error("节点移动失败", exc)
                    else:
                        self._after_geometry_edit(f"已移动节点 {moved_node}")
                self._drag_changed = False
                return True
        return super().eventFilter(watched, event)

    def _apply_scale(self):
        if self._v2_draft is not None:
            from copy import deepcopy
            from dimension_constraints import (add_member_length_dimension,
                                               apply_scale_to_draft)
            member_id = self.cmb_reference_member.currentData()
            try:
                draft = add_member_length_dimension(
                    self._v2_draft, int(member_id),
                    float(self.spn_reference_length.value()), "m")
                draft = apply_scale_to_draft(draft)
                if self._v2_state is not None and self._v2_state.draft is not None:
                    draft["revision"] = int(self._v2_state.draft.get("revision", 0))
                    self._v2_state.edit_draft(draft)
                    draft = self._v2_state.draft
                else:
                    draft = deepcopy(draft)
                    draft["revision"] = int(draft.get("revision", 0)) + 1
                    draft["confirmation"] = draft["merge_plan"] = None
            except (TypeError, ValueError) as exc:
                self.lbl_status.setText(f"尺度标定失败：{exc}")
                self.lbl_status.setStyleSheet(f"color:{theme.WARN}; font-size:8pt;")
                return
            self._v2_draft = draft
            self.issue_panel.set_draft(draft)
            self._show_overlay(draft)
            self._refresh_v2_scale_controls()
            self._refresh_v2_commit_state()
            return
        if self._result_draft is None:
            return
        member_id = self.cmb_reference_member.currentData()
        try:
            self._result_draft.calibrate(
                int(member_id), float(self.spn_reference_length.value()))
        except (TypeError, ValueError) as exc:
            self.lbl_status.setText(f"尺度标定失败：{exc}")
            self.lbl_status.setStyleSheet(f"color:{theme.WARN}; font-size:8pt;")
            return
        self.scale_widget.setVisible(False)
        for row in range(self.lst_questions.count()):
            item = self.lst_questions.item(row)
            if "尺度" in item.text():
                item.setCheckState(Qt.CheckState.Checked)
        self._reset_confirmation()
        self._refresh_result_text()
        self.lbl_status.setText(
            f"✅ 尺度已确认：{self._result_draft.scale_evidence}。请重新核对覆盖标注。")
        self.lbl_status.setStyleSheet(f"color:{theme.ACCENT}; font-size:8pt;")
        self._update_load_enabled()

    def _load_model(self):
        if self._v2_draft is not None:
            if (self._v2_state is None or not self.chk_confirm.isChecked()
                    or not self._v2_state.can_commit(self.session.model)):
                return
            from datetime import datetime, timezone
            from draft_commit import DraftCommitError, commit_prepared
            try:
                result = commit_prepared(
                    self.session, self._v2_state,
                    confirmed_at=datetime.now(timezone.utc).isoformat())
            except DraftCommitError as exc:
                self.lbl_status.setText(f"无法加载模型：{exc}")
                self.lbl_status.setStyleSheet(f"color:{theme.WARN}; font-size:8pt;")
                self._refresh_v2_commit_state()
                return
            self._result_model = self.session.model
            self.model_loaded.emit(self.session.model)
            self.chk_confirm.setEnabled(False)
            self.btn_load.setEnabled(False)
            warnings = result.payload.get("warnings") or []
            self.lbl_status.setText(
                f"V2 草稿已原子加载（1 个撤销步骤）"
                + (f"；仍有 {len(warnings)} 项工程属性待补全。" if warnings else "。"))
            self.lbl_status.setStyleSheet(f"color:{theme.ACCENT}; font-size:8pt;")
            return
        if (self._result_draft is None or not self.chk_confirm.isChecked()
                or not self._result_draft.ready_to_load):
            return
        result = self.session.apply_draft(
            self._result_draft.to_frame_draft(),
            note="多模态识别草稿经人工确认后加载；请继续指派属性并完成边界条件。")
        if not result.ok:
            errors = result.payload.get("errors") or [result.payload.get("error", "模型校验失败")]
            self.lbl_status.setText("无法加载模型：" + "；".join(map(str, errors[:3])))
            self.lbl_status.setStyleSheet(f"color:{theme.WARN}; font-size:8pt;")
            return
        self._result_model = self.session.model
        self.model_loaded.emit(self.session.model)
        warnings = result.payload.get("warnings") or []
        if warnings:
            self.lbl_status.setText(
                f"已加载为几何草稿，仍有 {len(warnings)} 项待补全；"
                "请设置材料、截面和边界条件后再求解。")
        else:
            self.lbl_status.setText("识别草稿已加载，请执行模型检查后求解。")
        self.lbl_status.setStyleSheet(f"color:{theme.ACCENT}; font-size:8pt;")
