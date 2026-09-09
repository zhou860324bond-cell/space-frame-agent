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
    ("deepseek", "deepseek-v4-flash-vision-exp"),
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


def test_recognized_model_requires_explicit_confirmation_before_loading(qt_app):
    session = Session()
    panel = SketchPanel(session, IdleRunner())
    draft = RecognitionDraft.from_payload(recognition_payload())

    panel._on_recognized(ParseResult(
        model=draft.model, draft=draft, success=True, attempts=1))

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


def test_node_can_be_dragged_on_the_image_overlay(qt_app, tmp_path):
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
