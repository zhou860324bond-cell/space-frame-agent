"""结构图纸识别面板：核对表处理、叠图高亮、生成与导入、主窗口入口。

防的问题：旧草图识别要走约十步、导入后截面为空不能计算；新面板只有
「添加 → 识别 → 核对 → 生成」四步，未处理完核对表时不能生成或导入。
"""

from __future__ import annotations

import json
import os

import pytest

pytest.importorskip("PySide6", reason="未安装 PySide6，跳过桌面端测试")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from agent import Session  # noqa: E402
from desktop.drawing_panel import DrawingPanel  # noqa: E402
from drawing_assemble import resolve  # noqa: E402
from drawing_extract import ExtractResult  # noqa: E402
from model_io import validate_payload  # noqa: E402
from test_drawing_assemble import _evidence  # noqa: E402


class Runner:
    busy = False

    def __init__(self):
        self.jobs = []

    def submit(self, fn, on_done=None, on_failed=None, on_progress=None):
        self.jobs.append((fn, on_done, on_failed))


@pytest.fixture(scope="module")
def qt_app():
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def drawing(tmp_path):
    from PIL import Image
    path = tmp_path / "平面图.png"
    Image.new("RGB", (2000, 1000), "white").save(path)
    return str(path)


@pytest.fixture
def panel(qt_app, drawing):
    panel = DrawingPanel(Session(), Runner())
    panel.load_images([drawing])
    return panel


def _select(panel, row_id):
    index = next(i for i, row in enumerate(panel._rows) if row.id == row_id)
    panel.table.selectRow(index)
    return panel.selected_row()


def test_generate_and_import_are_locked_until_every_row_is_handled(panel):
    """只剩一条需确认（规范表材料）时不能导入；确认后导入，模型带截面可直接用。"""
    panel.set_evidence(_evidence())
    assert not panel.btn_import.isEnabled() and not panel.btn_save.isEnabled()
    row = panel.selected_row()
    assert row.id == "material" and panel.btn_confirm.isEnabled()
    loaded = []
    panel.model_loaded.connect(loaded.append)
    panel.confirm_selected()
    assert panel.btn_import.isEnabled()
    assert "核对完毕" in panel.lbl_status.text()
    assert panel.import_model()
    assert len(panel.session.model["members"]) == 26
    assert all(m["section"] for m in panel.session.model["members"])
    assert loaded and panel.session.undo()


def test_missing_spacing_is_filled_through_the_text_editor(panel):
    """缺失的间距用文本框照图填写，来源记为用户。"""
    evidence = _evidence()
    evidence["grids"]["x"]["spacings"].pop(1)
    evidence["grids"]["x"].pop("total")
    panel.set_evidence(evidence)
    row = _select(panel, "grids.x.spacings+2:3")
    assert row.status == "missing"
    assert panel.txt_value.isVisibleTo(panel) and not panel.btn_confirm.isEnabled()
    panel.txt_value.setText("6000")
    panel.apply_selected()
    assert panel.evidence["grids"]["x"]["spacings"][-1]["source"] == {"user": True}
    assert not any(r.status == "missing" for r in panel.assembly.rows)


def test_support_choice_uses_the_combo_box(panel):
    """没有柱脚说明时从下拉框选，而不是默认固接。"""
    evidence = _evidence()
    evidence["support"] = None
    panel.set_evidence(evidence)
    _select(panel, "support")
    assert panel.cmb_value.isVisibleTo(panel)
    panel.cmb_value.setCurrentText("铰接")
    panel.apply_selected()
    panel.evidence = resolve(panel.evidence, "material")
    panel._reassemble()
    assert {tuple(s["fix"]) for s in panel.model_payload()["supports"]} == {(1, 1, 1, 0, 0, 0)}


def test_invalid_input_explains_and_keeps_evidence(panel):
    """单位只能选 mm/m，填错时说明原因，证据不变。"""
    evidence = _evidence()
    del evidence["units"]["length"]
    panel.set_evidence(evidence)
    before = json.dumps(panel.evidence, ensure_ascii=False, sort_keys=True)
    _select(panel, "units.length")
    panel._resolve("英寸")
    assert "单位只能是" in panel.lbl_status.text()
    assert json.dumps(panel.evidence, ensure_ascii=False, sort_keys=True) == before


def test_selected_row_highlights_its_source_and_axes_are_drawn_back(panel):
    """选中一行就在原图上框出原文位置；算出的轴线坐标画回图上核对。"""
    evidence = _evidence()
    evidence["levels"][1]["source"]["confidence"] = 0.5
    panel.set_evidence(evidence)
    row = _select(panel, "levels.1")
    assert panel.view._highlight == row.bbox
    assert ("x", "2", 0.4, 6.0) in panel.view._axes
    assert ("y", "B", 0.08, 7.2) in panel.view._axes


def test_pdf_is_refused_with_a_next_step(qt_app, tmp_path):
    """本版不读 PDF：说明原因并给出导出 PNG 的做法，不静默忽略。"""
    panel = DrawingPanel(Session(), Runner())
    panel.load_images([str(tmp_path / "结构图.pdf")])
    assert panel.images == []
    assert "导出为 PNG" in panel.lbl_status.text()


def test_save_writes_model_and_evidence_sidecar(panel, tmp_path):
    """保存的模型 JSON 通过校验，旁边的证据文件记下每个数的出处。"""
    panel.set_evidence(_evidence())
    with pytest.raises(ValueError, match="未处理"):
        panel.save_model(tmp_path / "模型.json")
    _select(panel, "material")
    panel.confirm_selected()
    sidecar = panel.save_model(tmp_path / "模型.json")
    model = json.loads((tmp_path / "模型.json").read_text(encoding="utf-8"))
    assert validate_payload(model) == []
    trace = json.loads(sidecar.read_text(encoding="utf-8"))
    assert sidecar.name == "模型.drawing-evidence.json"
    assert trace["evidence"]["format"] == "space-frame-drawing-evidence/v1"
    assert trace["provenance"]["nodes"]["1"]["z"].startswith("标高")


def test_recognize_runs_in_background_and_reports_failure(panel):
    """识别走后台；没有结果时写明原因和下一步，不留空表。"""
    panel.recognize()
    assert len(panel.runner.jobs) == 1 and not panel.btn_recognize.isEnabled()
    _, on_done, _ = panel.runner.jobs[0]
    on_done(ExtractResult(errors=["调用识别模型失败：缺少 API 密钥"]))
    assert "识别设置" in panel.lbl_status.text()
    assert panel.btn_recognize.isEnabled()


def test_main_window_entry_and_dropped_drawing(qt_app, drawing):
    """主入口是图纸识别；对话面板拖入的图纸进入图纸识别面板。"""
    from desktop.main_window import MainWindow
    w = MainWindow()
    assert w.actions_by_name["drawing_ai"].shortcut().toString() == "Ctrl+I"
    w.open_drawing_ai()
    assert w.drawing_dock.isVisible()
    w.drawing_dock.close()
    w._recognize_dropped_image(drawing)
    assert w.drawing.images == [drawing] and w.drawing_dock.isVisible()
    w.drawing_dock.close()
