"""桌面端多模态草图面板的轻量接线测试。"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6", reason="未安装 PySide6，跳过桌面端测试")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402
from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtGui import QColor, QPixmap  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from agent import Session  # noqa: E402
from desktop.sketch_panel import SketchPanel  # noqa: E402
from recognition_draft import DRAFT_FORMAT, RecognitionDraft  # noqa: E402
from sketch_parser import ParseResult  # noqa: E402


class IdleRunner:
    busy = False


@pytest.fixture(scope="module")
def qt_app():
    yield QApplication.instance() or QApplication([])


@pytest.mark.parametrize(("provider", "model"), [
    ("openai", "gpt-4o"),
    ("anthropic", "claude-3-5-sonnet-20241022"),
    # DeepSeek 平台上没有 deepseek-vl，那是开源权重名；能接收图片的是这个。
    ("deepseek", "deepseek-flash"),
])
def test_provider_switch_sets_a_matching_vision_model(qt_app, provider, model):
    panel = SketchPanel(Session(), IdleRunner())
    panel.cmb_provider.setCurrentText(provider)
    assert panel.txt_model.text() == model


def test_advanced_recognition_settings_are_collapsed_by_default(qt_app):
    panel = SketchPanel(Session(), IdleRunner())

    assert panel.settings_widget.isHidden()
    panel.btn_settings.setChecked(True)
    assert not panel.settings_widget.isHidden()


@pytest.mark.parametrize("action", ["edit", "delete"])
def test_support_manual_change_retains_symbol_evidence_and_only_resolves_its_review(qt_app, action):
    """人工支座编辑/删除不能丢复核原观察，或同时解除全局问题；改名后的其他问题仍能定位。"""
    from sketch_action_review import apply_action_review
    from test_sketch_action_review import support_draft, support_review
    value = apply_action_review(support_draft(), support_review())
    value["issues"].extend([
        {"id": "global", "category": "load_incomplete", "severity": "blocking", "status": "open", "entity_refs": []},
        {"id": "old-low", "category": "low_confidence", "severity": "blocking", "status": "open",
         "entity_refs": ["support:1:S1", "node:1"]},
    ])
    panel = SketchPanel(Session(), IdleRunner())
    panel.set_v2_draft(value)
    panel.cmb_support_node.setCurrentIndex(panel.cmb_support_node.findData(1))
    if action == "edit":
        panel.txt_support_name.setText("Reviewed")
        panel._apply_support_edit()
    else:
        panel._remove_support_edit()
    result = panel._v2_draft
    by_id = {i["id"]: i for i in result["issues"]}
    assert by_id["support-symbol-review-1"]["status"] == "resolved"
    assert by_id["global"]["status"] == by_id["old-low"]["status"] == "open"
    history = result["edit_history"][-1]
    previous = next(e for e in history["previous"]["entities"] if e["kind"] == "support")
    assert previous["symbol_review"]["observation"]["symbol"] == "triangle"
    if action == "edit":
        assert by_id["old-low"]["entity_refs"] == ["support:1:Reviewed", "node:1"]
        entity = next(e for e in result["entities"] if e["kind"] == "support")
        assert entity["original_observation"]["entities"][0]["symbol_review"]
        assert entity["verified"] and result["confirmation"] is None
    else:
        assert result["image_model"]["supports"] == []
        assert history["replacement"] is None


def recognition_payload(scale="confirmed"):
    return {
        "format": DRAFT_FORMAT,
        "scale": {"status": scale, "evidence": ""},
        "model": {
            "units": "N-m-Pa", "materials": [], "sections": [],
            "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                      {"id": 2, "x": 2, "y": 0, "z": 0}],
            "members": [{"id": 7, "i": 1, "j": 2}],
            "supports": [], "load_cases": [],
        },
        "entities": [], "questions": ["请确认是否没有支座"], "warnings": [],
    }


def confirm_questions(panel):
    for row in range(panel.lst_questions.count()):
        panel.lst_questions.item(row).setCheckState(Qt.CheckState.Checked)


def partial_panel(qt_app):
    from test_sketch_load_edit import beam_draft
    panel = SketchPanel(Session(), IdleRunner())
    panel.set_v2_draft(beam_draft())
    return panel


def fill_partial_inputs(panel):
    panel.span_a.setValue(1)
    panel.span_b.setValue(3)
    for spin, value in zip(panel.span_w1, [0, -1000, 0], strict=True):
        spin.lineEdit().selectAll()
        QTest.keyClicks(spin.lineEdit(), str(value))


def missing_nodal_panel(qt_app):
    from copy import deepcopy
    from sketch_parser import _convert_case_loads
    panel = partial_panel(qt_app)
    draft = deepcopy(panel._v2_draft)
    case = draft["image_model"]["load_cases"][0]
    case["nodal_loads"] = [{"node": 2, "name": "F"}, {"node": 2, "name": "G"}]
    problems = _convert_case_loads(case)
    for index, issue in enumerate(problems):
        issue.update(id=f"missing-{index}", status="open")
    draft["issues"] = problems + [{"id": "global", "category": "load_incomplete", "severity": "blocking",
        "entity_refs": [], "status": "open", "message": "其他荷载需要核对"}]
    draft["entities"].append({"id": "observed-F", "kind": "load", "source": "vision", "verified": False,
        "confidence": None, "target": {"load": {"case": case["name"], "collection": "nodal_loads", "name": "F"}},
        "image_geometry": {"point": [1, .5]}})
    panel.set_v2_draft(draft)
    panel.cmb_load_name.setCurrentText("F")
    return panel


def test_nodal_placeholder_stays_blank_and_cannot_apply_after_qt_fixup(qt_app):
    """真实无数值占位不能显示六个零，Qt 失焦自动显示零也不能被应用为工程荷载。"""
    from copy import deepcopy
    panel = missing_nodal_panel(qt_app)
    original = deepcopy(panel._v2_draft)
    assert all(not s.cleanText().strip() for s in panel.load_spins)
    for spin in panel.load_spins:
        spin.interpretText()
    panel.btn_apply_load.click()
    assert panel._v2_draft == original and "明确输入 0" in panel.lbl_status.text()
    assert not panel.btn_load.isEnabled()


@pytest.mark.parametrize("action", ["apply", "delete"])
def test_nodal_placeholder_manual_edit_preserves_evidence_and_other_blockers(qt_app, action):
    """填写或删除某个占位只处理其精确缺值问题，原观察及同节点另一条荷载仍保留。"""
    panel = missing_nodal_panel(qt_app)
    if action == "apply":
        for spin, value in zip(panel.load_spins, [0, 0, -500, 0, 0, 0], strict=True):
            spin.lineEdit().selectAll()
            QTest.keyClicks(spin.lineEdit(), str(value))
        panel.btn_apply_load.click()
        entry = next(i for i in panel._v2_draft["image_model"]["load_cases"][0]["nodal_loads"] if i["name"] == "F")
        assert entry["load"] == [0, 0, -500, 0, 0, 0]
        entity = next(e for e in panel._v2_draft["entities"] if e.get("kind") == "load")
        assert entity["verified"] and entity["original_observation"]["loads"] == [{"node": 2, "name": "F"}]
    else:
        panel.btn_remove_load.click()
    draft = panel._v2_draft
    issues = {i["id"]: i for i in draft["issues"]}
    assert issues["missing-0"]["status"] == "resolved" and issues["missing-0"]["observations"] == [{"node": 2, "name": "F"}]
    assert issues["missing-1"]["status"] == issues["global"]["status"] == "open"
    assert draft["edit_history"][-1]["previous"]["entities"][0]["id"] == "observed-F"
    assert not panel.btn_load.isEnabled()


def test_nodal_case_switch_does_not_reuse_filled_values_for_another_placeholder(qt_app):
    """填写到一半后切换另一无数值荷载，应重新显示缺值，不能借用上一条的分量。"""
    panel = missing_nodal_panel(qt_app)
    panel.load_spins[2].setValue(-200)
    panel.cmb_load_name.setCurrentText("G")
    assert all(not s.cleanText().strip() for s in panel.load_spins)
    panel.btn_apply_load.click()
    assert "分量尚未" in panel.lbl_status.text()


@pytest.mark.parametrize("values", [None, [0, 0, 0], [0, False, 0, 0, 0, 0],
                                   [0, float("nan"), 0, 0, 0, 0], [0, None, 0, 0, 0, 0]])
def test_invalid_or_partial_nodal_vector_cannot_be_silently_zeroed(qt_app, values):
    """旧草稿中的短向量、空值、布尔及非有限分量应显示缺失，不能截断或补零后应用。"""
    panel = missing_nodal_panel(qt_app)
    panel._v2_draft["image_model"]["load_cases"][0]["nodal_loads"][0]["load"] = values
    panel._fill_load_values()
    assert panel._missing_load_values
    panel.btn_apply_load.click()
    assert "分量尚未" in panel.lbl_status.text() and not panel.btn_load.isEnabled()


def test_partial_editor_stays_collapsed_and_requires_explicit_missing_values(qt_app):
    """新增局部编辑不能挤占原布局，缺失强度也不能由默认零值自动补齐。"""
    panel = partial_panel(qt_app)
    assert panel.partial_widget.isHidden() and not panel.btn_partial_loads.isHidden()
    panel.btn_partial_loads.setChecked(True)
    assert not panel.partial_widget.isHidden()
    panel.span_a.setValue(1)
    panel.span_b.setValue(3)
    panel.btn_apply_span.click()
    assert "必填" in panel.lbl_status.text()
    assert panel._v2_draft["image_model"]["load_cases"][0]["member_spans"] == []
    fill_partial_inputs(panel)
    panel.btn_apply_span.click()
    load = panel._v2_draft["image_model"]["load_cases"][0]["member_spans"][0]
    assert load["a"] == 1 and load["b"] == 3 and load["w1"] == [0, -1000, 0]
    assert not panel.chk_confirm.isChecked()
    panel.btn_remove_span.click()
    assert panel._v2_draft["image_model"]["load_cases"][0]["member_spans"] == []


def test_partial_editor_does_not_clamp_end_to_member_length(qt_app):
    """越界输入必须报原因，不得自动把终点截成杆长后提交另一种荷载。"""
    panel = partial_panel(qt_app)
    fill_partial_inputs(panel)
    panel.span_b.setValue(9)
    revision = panel._v2_draft["revision"]
    panel.btn_apply_span.click()
    assert "杆长" in panel.lbl_status.text() and "8" in panel.lbl_status.text()
    assert panel.span_b.value() == 9 and panel._v2_draft["revision"] == revision


def test_partial_editor_case_switch_clears_other_case_values(qt_app):
    """切换工况不能把前一工况的荷载强度和范围继续作为新工况默认值。"""
    panel = partial_panel(qt_app)
    fill_partial_inputs(panel)
    panel.btn_apply_span.click()
    panel.cmb_span_case.setCurrentText("W")
    assert not panel.span_a.cleanText() and not panel.span_w1[1].cleanText()
    assert not panel.btn_remove_span.isEnabled()
    assert panel._v2_draft["image_model"]["load_cases"][1]["member_spans"] == []


def test_partial_editor_waits_for_confirmed_scale(qt_app):
    """尺度未知时只允许查看、删除已识别荷载，不能把图像距离当米应用。"""
    panel = partial_panel(qt_app)
    panel._v2_draft["scale"].update(status="unknown", length_per_pixel=None)
    panel._refresh_partial_controls()
    assert not panel.btn_apply_span.isEnabled() and "尺度" in panel.lbl_span_length.text()


def test_partial_editor_trapezoid_requires_end_intensity(qt_app):
    """梯形末端强度缺失时不能默认成三角形，均布模式也不应启用末端编辑。"""
    panel = partial_panel(qt_app)
    assert not any(s.isEnabled() for s in panel.span_w2)
    panel.cmb_span_kind.setCurrentIndex(1)
    fill_partial_inputs(panel)
    panel.btn_apply_span.click()
    assert "必填" in panel.lbl_status.text()
    for spin, value in zip(panel.span_w2, [0, -2000, 0], strict=True):
        spin.lineEdit().selectAll()
        QTest.keyClicks(spin.lineEdit(), str(value))
    panel.btn_apply_span.click()
    load = panel._v2_draft["image_model"]["load_cases"][0]["member_spans"][0]
    assert load["kind"] == "partial_trapezoid" and load["w2"] == [0, -2000, 0]


def test_partial_empty_field_remains_missing_after_qt_focus_fixup(qt_app):
    """Qt 离开空数值框会显示零，不能把这种自动修复当作用户已确认的零强度。"""
    panel = partial_panel(qt_app)
    panel.span_a.setValue(1)
    panel.span_b.setValue(3)
    for spin in panel.span_w1:
        spin.interpretText()
    panel._apply_partial_load_edit()
    assert "必填" in panel.lbl_status.text()
    assert panel._v2_draft["image_model"]["load_cases"][0]["member_spans"] == []


def test_partial_range_overlay_is_visible_and_does_not_mark_full_member(qt_app, tmp_path):
    """局部范围与梁轴重叠时应有可辨括线，且不能把无荷载的后半段标成受载。"""
    from sketch_load_edit import set_partial_load
    panel = partial_panel(qt_app)
    path = tmp_path / "range.png"
    source = QPixmap(101, 201)
    source.fill(Qt.GlobalColor.white)
    source.save(str(path))
    panel._image_path = str(path)
    draft = set_partial_load(panel._v2_draft, "D", "q", 7, 1, 3, [0, -1, 0])
    panel.set_v2_draft(draft)
    image = panel._preview_pixmap.toImage()
    assert image.pixelColor(30, 110) != QColor("white")
    assert image.pixelColor(75, 110) == QColor("white")


def test_manual_v2_nodal_load_materializes_without_schema_extra_units(qt_app):
    """防止人工集中力矩在 image_model 中带入 units 后物化被模型 schema 拒收。"""
    from model_io import validate_payload
    from sketch_topology import materialize_geometry
    panel = partial_panel(qt_app)
    panel.load_spins[4].setValue(-5000)
    panel.btn_apply_load.click()
    load = panel._v2_draft["image_model"]["load_cases"][0]["nodal_loads"][0]
    assert load["load"][4] == -5000 and "units" not in load
    errors = validate_payload(materialize_geometry(panel._v2_draft))
    assert not [e for e in errors if "Additional properties" in e], errors


def test_missing_action_node_can_be_inserted_and_selected_for_nodal_loads(qt_app):
    """自动识别漏掉集中力矩作用点时，按钮应补节点、拆杆并选择新节点，不能直接提交。"""
    from multimodal_workflow import migrate_v1_payload
    panel = SketchPanel(Session(), IdleRunner())
    draft = migrate_v1_payload(recognition_payload()["model"], image_hash="a" * 64)
    draft["source"].update(width_px=501, height_px=301)
    panel.set_v2_draft(draft)
    panel.chk_confirm.setChecked(True)
    old_revision = panel._v2_draft["revision"]
    panel.spin_member_position.setValue(40)
    panel.btn_insert_member_node.click()
    model = panel._v2_draft["image_model"]
    assert len(model["nodes"]) == 3 and len(model["members"]) == 2, panel.lbl_status.text()
    assert model["nodes"][-1]["id"] == 3
    assert panel.cmb_load_node.currentData() == 3
    assert panel._v2_draft["revision"] > old_revision
    assert not panel.chk_confirm.isChecked() and panel._v2_draft["confirmation"] is None
    assert not panel.btn_load.isEnabled()


def test_manual_axis_correction_moves_support_marker_without_confirming_type(qt_app):
    """人工拖梁轴端点修正混合图时，支座覆盖不能留在旧位置或顺便确认支座类型。"""
    from multimodal_workflow import migrate_v1_payload
    panel = SketchPanel(Session(), IdleRunner())
    model = recognition_payload()["model"]
    model["supports"] = [{"node": 1, "fix": [1, 1, 1, 0, 0, 0], "name": "S"}]
    draft = migrate_v1_payload(model, image_hash="a" * 64)
    draft["source"].update(width_px=501, height_px=301)
    panel.set_v2_draft(draft)
    panel._move_v2_node_preview(1, 0.2, 0.6)
    support = next(e for e in panel._v2_draft["entities"] if e["kind"] == "support")
    assert support["image_geometry"] == {"point": [0.2, 0.6]}
    assert support["original_observation"] and not support["verified"]
    assert panel._v2_draft["image_model"]["supports"] == model["supports"]


def test_recognized_model_requires_explicit_confirmation_before_loading(qt_app):
    session = Session()
    panel = SketchPanel(session, IdleRunner())
    draft = RecognitionDraft.from_payload(recognition_payload())

    panel._on_recognized(ParseResult(
        model=draft.model, draft=draft, success=True, attempts=1))

    assert panel._stage_index == 3
    assert "校核与装配" in panel.lbl_stage.text()
    assert "尚未写入" in panel.lbl_stage.text() or not session.model
    assert panel.chk_confirm.isEnabled()
    assert panel.txt_result.isHidden()
    assert not panel.btn_details.isHidden()
    assert not panel.btn_load.isEnabled()
    assert not session.model

    panel.chk_confirm.setChecked(True)
    assert not panel.btn_load.isEnabled()
    confirm_questions(panel)
    assert panel.btn_load.isEnabled()
    panel._load_model()

    assert panel._stage_index == 4
    assert "计算模型" in panel.lbl_stage.text()
    assert len(session.model["nodes"]) == 2
    assert session.model["members"][0]["section"] == ""
    assert len(session.history) == 1
    assert "几何草稿" in panel.lbl_status.text()


def test_unknown_scale_must_be_calibrated_before_loading(qt_app):
    panel = SketchPanel(Session(), IdleRunner())
    draft = RecognitionDraft.from_payload(recognition_payload("unknown"))
    panel._on_recognized(ParseResult(
        model=draft.model, draft=draft, success=True, attempts=1))
    panel.chk_confirm.setChecked(True)
    confirm_questions(panel)

    assert not panel.btn_load.isEnabled()
    assert not panel.scale_widget.isHidden()

    panel.spn_reference_length.setValue(6.0)
    panel._apply_scale()

    assert not panel.btn_load.isEnabled()
    panel.chk_confirm.setChecked(True)
    assert panel.btn_load.isEnabled()
    assert panel._result_draft.model["nodes"][1]["x"] == pytest.approx(6.0)


def test_recognized_entities_are_drawn_over_the_source_image(qt_app, tmp_path):
    path = tmp_path / "drawing.png"
    source = QPixmap(120, 80)
    source.fill(Qt.GlobalColor.white)
    assert source.save(str(path))
    data = recognition_payload()
    data["entities"] = [
        {"kind": "member", "id": 7, "confidence": 0.9,
         "image_geometry": {"line": [[0.1, 0.5], [0.9, 0.5]]}},
        {"kind": "node", "id": 1, "confidence": 0.9,
         "image_geometry": {"point": [0.1, 0.5]}},
        {"kind": "support", "id": "S1", "confidence": None,
         "image_geometry": {"bbox": [0.05, 0.55, 0.2, 0.85]}},
    ]
    draft = RecognitionDraft.from_payload(data, source_image=path)
    panel = SketchPanel(Session(), IdleRunner())
    panel._image_path = str(path)

    panel._show_overlay(draft)

    rendered = panel.lbl_image.pixmap().toImage()
    coloured = sum(
        1 for y in range(rendered.height()) for x in range(rendered.width())
        if rendered.pixelColor(x, y) != QColor(Qt.GlobalColor.white))
    assert coloured > 20


def test_member_editor_adds_and_removes_members(qt_app):
    data = recognition_payload()
    data["model"]["nodes"].append({"id": 3, "x": 2, "y": 0, "z": 2})
    data["model"]["members"].append({"id": 8, "i": 2, "j": 3})
    draft = RecognitionDraft.from_payload(data)
    panel = SketchPanel(Session(), IdleRunner())
    panel._on_recognized(ParseResult(
        model=draft.model, draft=draft, success=True, attempts=1))

    panel.cmb_member_i.setCurrentIndex(panel.cmb_member_i.findData(1))
    panel.cmb_member_j.setCurrentIndex(panel.cmb_member_j.findData(3))
    panel._add_member()

    added = max(member["id"] for member in draft.model["members"])
    assert any({member["i"], member["j"]} == {1, 3}
               for member in draft.model["members"])
    panel.cmb_remove_member.setCurrentIndex(panel.cmb_remove_member.findData(added))
    panel._remove_member()
    assert all(member["id"] != added for member in draft.model["members"])


@pytest.mark.parametrize("zoom", [100, 300])
def test_node_can_be_dragged_on_the_image_overlay(qt_app, tmp_path, zoom):
    """防止放大并滚动预览后，标签坐标与图片坐标错位，拖动修改错误位置。"""
    path = tmp_path / "drag.png"
    source = QPixmap(200, 120)
    source.fill(Qt.GlobalColor.white)
    assert source.save(str(path))
    data = recognition_payload()
    data["entities"] = [
        {"kind": "node", "id": 1, "confidence": 0.9,
         "image_geometry": {"point": [0.1, 0.5]}},
        {"kind": "node", "id": 2, "confidence": 0.9,
         "image_geometry": {"point": [0.9, 0.5]}},
        {"kind": "member", "id": 7, "confidence": 0.9,
         "image_geometry": {"line": [[0.1, 0.5], [0.9, 0.5]]}},
    ]
    draft = RecognitionDraft.from_payload(data, source_image=path)
    panel = SketchPanel(Session(), IdleRunner())
    panel.resize(500, 700)
    panel._image_path = str(path)
    panel._on_recognized(ParseResult(
        model=draft.model, draft=draft, success=True, attempts=1))
    panel.show()
    qt_app.processEvents()
    panel.preview_zoom.setValue(zoom)
    qt_app.processEvents()
    if zoom > 100:
        panel.preview_scroll.horizontalScrollBar().setValue(50)
        panel.preview_scroll.verticalScrollBar().setValue(20)

    pixmap = panel.lbl_image.pixmap()
    left = (panel.lbl_image.width() - pixmap.width()) / 2
    top = (panel.lbl_image.height() - pixmap.height()) / 2
    start = QPoint(int(left + 0.9 * pixmap.width()), int(top + 0.5 * pixmap.height()))
    end = QPoint(int(left + 0.7 * pixmap.width()), int(top + 0.3 * pixmap.height()))
    QTest.mousePress(panel.lbl_image, Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(panel.lbl_image, pos=end)
    QTest.mouseRelease(panel.lbl_image, Qt.MouseButton.LeftButton, pos=end)

    assert draft._image_point(draft._node_entity(2)) == pytest.approx((0.7, 0.3), abs=0.02)
    assert not panel.chk_confirm.isChecked()


def test_work_plane_change_invalidates_and_reconfirmation_updates_draft(qt_app):
    """防止用户改平面和偏移后，导入仍使用先前已确认的工作平面。"""
    from image_preprocess import work_plane_payload
    from multimodal_workflow import migrate_v1_payload
    panel = SketchPanel(Session(), IdleRunner())
    draft = migrate_v1_payload(recognition_payload()["model"], image_hash="a" * 64)
    panel.set_v2_draft(draft)
    panel._on_work_plane_confirmed(work_plane_payload("XZ", confirmed=True))
    panel.preprocess_widget.plane.setCurrentText("YZ")
    panel.preprocess_widget.offset.setValue(2.5)
    assert panel._work_plane is None
    assert panel._v2_draft["work_plane"]["status"] == "proposed"
    assert not panel.btn_load.isEnabled() and not panel.chk_confirm.isChecked()
    panel.preprocess_widget.confirm_plane.click()
    assert panel._v2_draft["work_plane"] == work_plane_payload("YZ", offset=2.5, confirmed=True)
    assert panel._v2_draft["confirmation"] is None


def test_new_image_clears_previous_review_queue(qt_app, tmp_path):
    """防止第二张图仍显示上一张图的问题及紫色定位引用。"""
    from PIL import Image
    panel = SketchPanel(Session(), IdleRunner())
    panel.issue_panel.set_draft({"issues": [{"id": "old", "status": "open",
                                           "severity": "blocking", "category": "perspective"}]})
    panel._highlight_refs = {"node:77"}
    path = tmp_path / "new.png"
    Image.new("RGB", (200, 120), "white").save(path)
    panel.preprocess_widget.set_source(str(path))
    assert panel._highlight_refs == set()
    assert panel.issue_panel.list.count() == 0 and panel.issue_panel.isHidden()
    assert not panel.btn_recognize.isEnabled()


def test_plane_change_requires_new_recognition_of_load_directions(qt_app):
    """防止确认新平面直接放行旧平面的荷载方向；偏移改变仍仅需重新确认。"""
    from multimodal_workflow import migrate_v1_payload
    panel = SketchPanel(Session(), IdleRunner())
    model = recognition_payload()["model"]
    model["load_cases"] = [{"name": "D", "nodal_loads": [
        {"node": 1, "name": "P", "load": [0, 0, -100, 0, 0, 0]}]}]
    panel.set_v2_draft(migrate_v1_payload(model, image_hash="a" * 64))
    panel.preprocess_widget.offset.setValue(2.5)
    assert not any(i["id"] == "ui-plane-loads" for i in panel._v2_draft["issues"])
    panel._v2_draft["issues"].append({"id": "ui-plane-loads", "category": "low_confidence",
                                      "severity": "warning", "status": "resolved", "message": "old"})
    panel.preprocess_widget.plane.setCurrentText("XY")
    panel.preprocess_widget.confirm_plane.click()
    blocker = next(i for i in panel._v2_draft["issues"] if i["id"] == "ui-plane-loads-edit")
    assert blocker["status"] == "open" and "重新识别" in blocker["message"]
    assert not panel.btn_load.isEnabled()


def test_overlay_redraw_does_not_accumulate_removed_markers(qt_app, tmp_path):
    """防止复用已解码图片后，删除或移动对象仍留下上一轮的覆盖标记。"""
    path = tmp_path / "image.png"
    source = QPixmap(200, 120)
    source.fill(Qt.GlobalColor.white)
    source.save(str(path))
    panel = SketchPanel(Session(), IdleRunner())
    panel._image_path = str(path)
    panel._show_overlay({"entities": [{"kind": "node", "id": 1, "confidence": 0.9,
                                     "image_geometry": {"point": [0.5, 0.5]}}]})
    assert panel._preview_pixmap.toImage().pixelColor(100, 60) != QColor("white")
    panel._show_overlay({"entities": []})
    assert panel._preview_pixmap.toImage().pixelColor(100, 60) == QColor("white")


def test_stale_recognition_uses_snapshot_and_cannot_replace_new_image(qt_app, tmp_path, monkeypatch):
    """防止后台读取变化中的面板状态，旧响应覆盖后来选择的图片和审核草稿。"""
    from PIL import Image
    from sketch_parser import SketchParser, V2ParseResult
    from image_preprocess import work_plane_payload

    class QueuedRunner:
        busy = False

        def submit(self, job, **callbacks):
            self.job, self.callbacks = job, callbacks
            return True

    runner = QueuedRunner()
    panel = SketchPanel(Session(), runner)
    panel.txt_key.setText("test-key")
    first, second = tmp_path / "first.png", tmp_path / "second.png"
    Image.new("RGB", (200, 120), "white").save(first)
    Image.new("RGB", (200, 120), "black").save(second)
    panel.preprocess_widget.set_source(str(first))
    panel._on_work_plane_confirmed(work_plane_payload("XZ", confirmed=True))
    expected_hash = panel._preprocess_metadata["image_hash"]
    expected_path = panel._image_path
    captured = {}

    def parse(_parser, path, state, job_id, **kwargs):
        captured.update(path=path, image_hash=state.image_hash, review_actions=kwargs["review_actions"])
        return V2ParseResult(success=False)

    monkeypatch.setattr(SketchParser, "parse_v2_with_retry", parse)
    assert not panel.chk_review_actions.isChecked()
    panel.chk_review_actions.setChecked(True)
    panel._recognize()
    panel.chk_review_actions.setChecked(False)
    assert not panel.preprocess_widget.isEnabled()
    panel.preprocess_widget.set_source(str(second))
    runner.callbacks["on_done"](runner.job())
    assert captured == {"path": expected_path, "image_hash": expected_hash, "review_actions": True}
    assert panel._v2_draft is None and panel._v2_state is None
    assert panel._source_image_path == str(second)
    assert panel.preprocess_widget.isEnabled() and "舍弃" in panel.lbl_status.text()


def test_desktop_cancellation_stops_before_format_repair(qt_app, tmp_path, monkeypatch):
    """防止桌面取消只舍弃最终结果，期间仍继续发起收费的格式修复请求。"""
    from threading import Event
    from PIL import Image
    from image_preprocess import work_plane_payload
    from sketch_parser import SketchParser
    from task_control import task_scope

    class QueuedRunner:
        busy = False

        def submit(self, job, **callbacks):
            self.job = job
            return True

    event, calls = Event(), []
    runner = QueuedRunner()
    panel = SketchPanel(Session(), runner)
    panel.txt_key.setText("test-key")
    path = tmp_path / "image.png"
    Image.new("RGB", (8, 6), "white").save(path)
    panel.preprocess_widget.set_source(str(path))
    panel._on_work_plane_confirmed(work_plane_payload("XZ", confirmed=True))

    def call(*args, **kwargs):
        calls.append(1)
        event.set()
        return "invalid JSON"

    monkeypatch.setattr(SketchParser, "_call_llm", call)
    panel._recognize()
    with task_scope(event):
        result = runner.job()
    assert result.cancelled and calls == [1]
    panel._stop_elapsed_ticker()


def test_support_and_nodal_load_editor_updates_the_recognition_draft(qt_app):
    data = recognition_payload()
    data["questions"] = ["请确认支座类型", "请确认荷载数值"]
    data["entities"] = [
        {"kind": "node", "id": 1, "confidence": 0.9,
         "image_geometry": {"point": [0.1, 0.5]}},
        {"kind": "node", "id": 2, "confidence": 0.9,
         "image_geometry": {"point": [0.9, 0.5]}},
    ]
    draft = RecognitionDraft.from_payload(data)
    panel = SketchPanel(Session(), IdleRunner())
    panel._on_recognized(ParseResult(
        model=draft.model, draft=draft, success=True, attempts=1))

    assert panel.boundary_widget.isHidden()
    panel.btn_boundaries.setChecked(True)
    panel.cmb_support_node.setCurrentIndex(panel.cmb_support_node.findData(1))
    panel.cmb_support_type.setCurrentIndex(1)
    panel.txt_support_name.setText("BC-Pin")
    panel._apply_support_edit()

    assert draft.model["supports"] == [{
        "node": 1, "fix": [1, 1, 1, 0, 0, 0], "name": "BC-Pin"}]
    assert panel.lst_questions.item(0).checkState() == Qt.CheckState.Checked

    panel.txt_new_case.setText("Wind")
    panel._add_load_case_edit()
    panel.cmb_load_node.setCurrentIndex(panel.cmb_load_node.findData(2))
    panel.cmb_load_name.setEditText("P1")
    panel.load_spins[2].setValue(-1000.0)
    panel._apply_nodal_load_edit()

    entry = draft.model["load_cases"][0]["nodal_loads"][0]
    assert entry == {"name": "P1", "node": 2,
                     "load": [0.0, 0.0, -1000.0, 0.0, 0.0, 0.0]}
    assert panel.lst_questions.item(1).checkState() == Qt.CheckState.Checked

    panel._remove_nodal_load_edit()
    assert draft.model["load_cases"][0]["nodal_loads"] == []
    panel.cmb_support_node.setCurrentIndex(panel.cmb_support_node.findData(1))
    panel._remove_support_edit()
    assert draft.model["supports"] == []


def test_panel_model_defaults_come_from_the_parser(qt_app):
    """面板不得自带一份模型名映射。

    守的是一次真实事故：面板抄了一份 defaults，DeepSeek 模型名更新后只改了
    sketch_parser 那一处，面板继续填旧名字，每次识别都报 model not found，
    看起来像"识别能力有问题"，实际是两份常量漂移。
    """
    from sketch_parser import SketchParser
    panel = SketchPanel(Session(), IdleRunner())
    for provider in ("openai", "anthropic", "deepseek"):
        panel.cmb_provider.setCurrentText(provider)
        assert panel.txt_model.text() == SketchParser._default_model(provider), provider


def test_recognition_shows_elapsed_time(qt_app):
    """识别期间界面必须动起来。

    原来只有一句静止的"正在识别"。视觉模型读大图几十秒是正常的，但界面毫无
    变化时，"还在跑"和"已经死了"看起来一模一样——用户只能猜，然后去点关闭。
    """
    from sketch_parser import REQUEST_TIMEOUT_SECONDS
    panel = SketchPanel(Session(), IdleRunner())
    panel._start_elapsed_ticker()
    text = panel.lbl_status.text()
    assert "已用" in text and "1s" in text
    assert str(int(REQUEST_TIMEOUT_SECONDS)) in text, "要让用户知道上限是多少"
    panel._stop_elapsed_ticker()
    assert not panel._elapsed_timer.isActive()


def test_vision_calls_have_a_bounded_timeout():
    """必须显式限定单次调用时长。

    openai SDK 默认 600 秒超时并自动重试 2 次，最坏情况一次识别静默阻塞
    半小时——用户看到的就是"卡住了"。项目自己有修复重试循环，SDK 层不该再重试。
    """
    from sketch_parser import REQUEST_TIMEOUT_SECONDS
    assert 10.0 <= REQUEST_TIMEOUT_SECONDS <= 300.0
    import inspect
    from sketch_parser import SketchParser
    source = inspect.getsource(SketchParser._get_client)
    assert "timeout=REQUEST_TIMEOUT_SECONDS" in source
    assert "max_retries=0" in source


def test_default_provider_follows_available_credentials(monkeypatch):
    """默认提供商必须选一个真的有密钥的。

    下拉框原来固定停在第一项 openai，而多数机器上只配了 deepseek.key。
    打开面板点识别必然报"缺少 API 密钥"，用户以为识别功能坏了——
    首次就注定失败的默认值不是中立，是坑。
    """
    from desktop.sketch_panel import _provider_with_credentials
    import credentials

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(credentials, "load_api_key", lambda *a, **k: "k")
    assert _provider_with_credentials() == "deepseek"

    monkeypatch.setenv("OPENAI_API_KEY", "x")
    assert _provider_with_credentials() == "openai"

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(credentials, "load_api_key", lambda *a, **k: None)
    assert _provider_with_credentials() == "openai", "都没有时保持原默认"


def test_prompt_pins_a_node_numbering_order():
    """编号顺序必须写死在提示词里。

    视觉模型给的 id 本来是任意的，人没法把"识别出的 3 号"和图上某个节点对上。
    定死"先按 v 从大到小分层、同层按 u 从小到大"之后，编号才可核对。
    """
    from sketch_parser import V2_SKETCH_SYSTEM_PROMPT as prompt
    assert "v 从大到小" in prompt and "u 从小到大" in prompt
    assert "fixed / pinned / roller" in prompt, "支座类型要给模型枚举清楚"


def test_panel_opens_with_a_model_name_matching_the_preselected_provider(qt_app, monkeypatch):
    """开面板时提供商和模型名必须是一致的一对。

    实测事故：预选写在 currentTextChanged.connect 之前，信号没接上，
    模型名一直停在构造时写死的 "gpt-4o"。于是提供商是 deepseek、模型名是
    gpt-4o，DeepSeek 返回 400 invalid_request_error——用户看到的又是
    "识别失败"，而真正的原因是两个控件不同步。
    """
    import credentials
    from sketch_parser import SketchParser
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(credentials, "load_api_key", lambda *a, **k: "k")

    panel = SketchPanel(Session(), IdleRunner())
    provider = panel.cmb_provider.currentText()
    assert provider == "deepseek"
    assert panel.txt_model.text() == SketchParser._default_model(provider)


@pytest.mark.parametrize("automatic_interpret", [False, True])
def test_reference_length_blank_cannot_apply_default_scale(qt_app, automatic_interpret):
    """未填参考长度及 Qt 失焦补零不能把默认 1 m 写成已确认尺度。"""
    from copy import deepcopy
    panel = SketchPanel(Session(), IdleRunner())
    draft = RecognitionDraft.from_payload(recognition_payload("unknown"))
    panel._on_recognized(ParseResult(model=draft.model, draft=draft, success=True, attempts=1))
    before = deepcopy(draft.model)
    assert not panel.spn_reference_length.cleanText().strip()
    if automatic_interpret:
        panel.spn_reference_length.interpretText()
    panel._apply_scale()
    assert panel._result_draft.scale_status == "unknown"
    assert panel._result_draft.model == before
    assert "实际长度尚未填写" in panel.lbl_status.text()
    assert not panel.btn_load.isEnabled()


@pytest.mark.parametrize("transition", ["new_draft", "new_member", "erase"])
def test_reference_length_cannot_leak_to_another_draft_or_member(qt_app, transition):
    """上一图/上一参考杆件的实际长度，以及清空后的自动补零，都不能被确认使用。"""
    from multimodal_workflow import migrate_v1_payload
    model = recognition_payload()["model"]
    model["members"].append({"id": 8, "i": 2, "j": 1})
    draft = migrate_v1_payload(model, image_hash="a" * 64)
    panel = SketchPanel(Session(), IdleRunner())
    panel.set_v2_draft(draft)
    panel.spn_reference_length.setValue(10.0)
    if transition == "new_draft":
        panel.set_v2_draft(draft)
    elif transition == "new_member":
        panel.cmb_reference_member.setCurrentIndex(1)
    else:
        panel.spn_reference_length.lineEdit().selectAll()
        QTest.keyClick(panel.spn_reference_length.lineEdit(), Qt.Key.Key_Backspace)
        panel.spn_reference_length.interpretText()
    previous = panel._v2_draft
    panel._apply_scale()
    assert panel._v2_draft is previous
    assert "实际长度尚未填写" in panel.lbl_status.text()


def test_dimension_observations_visible_literal_and_not_automatic_scale(qt_app):
    """真实新图有 10 m 标注但无已确认尺寸绑定，原观察需可见且不能自动套到参考杆件。"""
    from copy import deepcopy
    from multimodal_workflow import migrate_v1_payload
    draft = migrate_v1_payload(recognition_payload()["model"], image_hash="a" * 64)
    draft["dimensions"] = [{"id": "D1", "text": "10 m <span>", "unit": "m"},
                           {"id": "D2", "text": "20°", "unit": "deg"}]
    before = deepcopy(draft["dimensions"])
    panel = SketchPanel(Session(), IdleRunner())
    panel.set_v2_draft(draft)
    assert not panel.txt_dimension_observations.isHidden()
    assert panel.txt_dimension_observations.isReadOnly()
    assert "10 m <span>" in panel.txt_dimension_observations.toPlainText()
    assert "20°" in panel.txt_dimension_observations.toPlainText()
    assert "待核对" in panel.txt_dimension_observations.toPlainText()
    assert not panel.spn_reference_length.cleanText().strip()
    assert panel._v2_draft["dimensions"] == before
    panel._v2_draft["dimensions"] = []
    panel._refresh_v2_scale_controls()
    assert panel.txt_dimension_observations.isHidden()



def test_explicit_reference_length_creates_only_user_scale_evidence(qt_app):
    """实际长度明确填写后应正常标定，并保留原尺寸观察；不能把模型文字改成自动确认。"""
    from multimodal_workflow import migrate_v1_payload
    draft = migrate_v1_payload(recognition_payload()["model"], image_hash="a" * 64)
    draft["source"].update(width_px=201, height_px=101)
    draft["image_model"]["nodes"] = [{"id": 1, "u": .1, "v": .5}, {"id": 2, "u": .9, "v": .5}]
    original = {"id": "observed", "text": "10 m", "unit": "m", "status": "proposed"}
    draft["dimensions"] = [original]
    panel = SketchPanel(Session(), IdleRunner())
    panel.set_v2_draft(draft)
    panel.spn_reference_length.setValue(10.0)
    panel._refresh_v2_scale_controls()
    assert panel.spn_reference_length.value() == 10.0
    panel._apply_scale()
    assert panel._v2_draft["scale"]["status"] == "confirmed"
    assert panel._v2_draft["scale"]["length_per_pixel"] == pytest.approx(10 / 160)
    assert panel._v2_draft["dimensions"][0] == original
    assert panel._v2_draft["dimensions"][-1]["source"] == "user"
    assert not panel.btn_load.isEnabled()
