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
from desktop.drawing_panel import DrawingPanel, Recognition  # noqa: E402
from drawing_assemble import resolve  # noqa: E402
from drawing_extract import ExtractResult  # noqa: E402
from model_io import validate_payload  # noqa: E402
from test_drawing_assemble import _evidence, _screenshot_evidence  # noqa: E402


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


def test_screenshot_frame_can_be_reviewed_and_imported(panel):
    """用户截图的八杆框架应完成核对并导入，不把同一标高和统一规格列成冲突。"""
    panel.set_evidence(_screenshot_evidence())
    assert not panel.assembly.blocking
    assert len([row for row in panel._rows if row.category == "杆件"]) == 8
    _select(panel, "material")
    panel.confirm_selected()
    assert panel.btn_import.isEnabled()
    assert panel.import_model()
    assert len(panel.session.model["members"]) == 8
    assert len(panel.session.model["sections"]) == 1


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


def test_dxf_is_read_locally_and_view_explains_there_is_no_picture(qt_app, drawing, tmp_path):
    """「添加图纸」接受 DXF：本机直接读出图层待判断，视图写明没有图片时怎样核对出处。

    2026-10-17 起 DXF 可以和图片一起使用（各自产出证据再合并），加入 DXF 不再移除图片。"""
    pytest.importorskip("ezdxf", reason="未安装 ezdxf")
    from test_drawing_dxf import _frame_dxf
    dxf = _frame_dxf(tmp_path / "框架立面.dxf")
    panel = DrawingPanel(Session(), Runner())
    panel.load_images([drawing])
    panel.load_images([dxf])
    assert panel.images == [drawing, dxf] and panel.dxf_path == dxf
    assert "已读取 DXF" in panel.lbl_status.text() and "1 张图片" in panel.lbl_status.text()
    assert "句柄" in panel.view.note and panel.view._pixmap is None
    layer_rows = [row for row in panel._rows if row.category == "图层"]
    assert layer_rows and all(row.status == "confirm" for row in layer_rows)
    assert not panel.btn_import.isEnabled()
    # 其余图层已指定、只差「尺寸」：在核对表里选为尺寸标注，按新用途重读，间距来自标注实体
    from drawing_dxf import read_dxf
    from test_drawing_dxf import ROLES
    panel.set_evidence(read_dxf(dxf, {k: v for k, v in ROLES.items() if k != "尺寸"}))
    assert not panel.evidence["grids"]["x"]["spacings"]
    row = next(row for row in panel._rows if "「尺寸」" in row.item)
    _select(panel, row.id)
    panel.cmb_value.setCurrentText("尺寸标注")
    panel.apply_selected()
    spacing = panel.evidence["grids"]["x"]["spacings"][0]
    assert spacing["value"] == 6000.0 and spacing["source"]["handles"]


def test_dxf_layer_judgement_runs_in_background(qt_app, tmp_path):
    """「识别图纸」对 DXF 只请大模型判断图层用途；结果待确认，用户已定的图层不被覆盖。"""
    pytest.importorskip("ezdxf", reason="未安装 ezdxf")
    from drawing_dxf import model_layer_items, read_dxf
    from test_drawing_dxf import ROLES, _frame_dxf
    dxf = _frame_dxf(tmp_path / "框架立面.dxf")
    panel = DrawingPanel(Session(), Runner())
    panel.load_images([dxf])
    panel.recognize()
    assert len(panel.runner.jobs) == 1 and "图层用途" in panel.lbl_status.text()
    _, on_done, _ = panel.runner.jobs[0]
    on_done(Recognition(dxf={dxf: read_dxf(dxf, model_layer_items(dxf, ROLES, "测试模型"))}))
    assert "逐个确认" in panel.lbl_status.text()
    assert any("大模型" in row.message for row in panel._rows if row.category == "图层")


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
    on_done(Recognition(images=ExtractResult(errors=["调用识别模型失败：缺少 API 密钥"])))
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


def test_recognition_notes_are_shown_but_never_need_clicking(panel):
    """FEMA 451 真实验收（2026-10-16）：识别提示逐条点确认太繁琐。提示行显示为「提示」，
    不抢选中、不能也不必确认；只剩它们时即可导入。"""
    evidence = _evidence()
    evidence["notes"] = [{"text": f"第 {k} 处字迹不清", "source": {"image": "P1",
                                                          "confidence": 0.4}}
                         for k in range(5)]
    panel.set_evidence(evidence)
    assert panel.selected_row().id == "material"
    panel.confirm_selected()
    assert panel.btn_import.isEnabled()
    assert "识别提示 5 条" in panel.lbl_status.text()
    note = _select(panel, "notes.0")
    assert panel.table.item(panel._rows.index(note), 0).text() == "提示"
    assert not panel.btn_confirm.isEnabled() and not panel.btn_apply.isEnabled()


def test_notes_box_is_transcribed_as_its_own_evidence(panel):
    """「补充说明」与图片一起提交，但作为一份平级证据：来源列写第几句，和图上读数不一致时
    报冲突让用户选；改了说明，旧的抄写结果立即撤出，等重新识别。"""
    from test_drawing_merge import _statement
    panel.txt_notes.setPlainText("1~2 轴间距应为 6600")
    panel.recognize()
    assert "逐句原样抄写补充说明" in panel.lbl_status.text()
    _, on_done, _ = panel.runner.jobs[0]
    statement = _statement({"grids": {"x": {"axes": [], "spacings": [
        {"from": "1", "to": "2", "text": "6600", "source": {"sentence": 1}}]}}},
        ["1~2 轴间距应为 6600"])
    on_done(Recognition(images=ExtractResult(evidence=_evidence()),
                        statements=ExtractResult(evidence=statement)))
    row = _select(panel, "grids.x.spacings.0")
    assert row.status == "conflict" and "用户说明第 1 句" in row.source
    assert not panel.cmb_value.isHidden() and panel.cmb_value.count() == 2
    panel.cmb_value.setCurrentText("6600（用户说明第 1 句）")
    panel.apply_selected()
    assert panel.evidence["grids"]["x"]["spacings"][0]["text"] == "6600"
    panel.txt_notes.setPlainText("1~2 轴间距应为 6000")
    assert "statements" not in panel.parts
    assert not any("用户说明" in r.source for r in panel._rows)
    assert "补充说明没有识别" in panel.lbl_status.text()


def test_notes_alone_can_be_recognized(qt_app):
    """没有图纸、只写说明也能提交识别（简单框架可以全靠文字描述）。"""
    panel = DrawingPanel(Session(), Runner())
    assert not panel.btn_recognize.isEnabled()
    panel.txt_notes.setPlainText("x 向轴号 1、2，间距 6000 mm")
    assert panel.btn_recognize.isEnabled()
    panel.recognize()
    assert len(panel.runner.jobs) == 1


def test_several_dxf_files_and_images_are_kept_together(qt_app, drawing, tmp_path):
    """几个 DXF 与图片可以同时加入：DXF 各自编号 D1、D2 并在本机读取，同名图层写明属于哪个文件；
    识别时图片抄字、两个 DXF 的图层判断一起排进后台任务。"""
    pytest.importorskip("ezdxf", reason="未安装 ezdxf")
    from test_drawing_dxf import _frame_dxf
    first = _frame_dxf(tmp_path / "A 轴立面.dxf")
    second = _frame_dxf(tmp_path / "B 轴立面.dxf")
    panel = DrawingPanel(Session(), Runner())
    panel.load_images([drawing, first])
    panel.load_images([second, first])
    assert panel.images == [drawing, first, second]
    labels = [panel.lst_images.item(i).text() for i in range(panel.lst_images.count())]
    assert labels[0].startswith("图 1") and labels[1].startswith("DXF") and "B 轴立面" in labels[2]
    assert [record["id"] for record in panel.evidence["files"]] == ["D1", "D2"]
    layer_items = [row.item for row in panel._rows if row.category == "图层"]
    assert any("A 轴立面.dxf" in item for item in layer_items)
    assert any("B 轴立面.dxf" in item for item in layer_items)
    assert "1 张图片" in panel.lbl_status.text()
    panel.recognize()
    text = panel.lbl_status.text()
    assert "抄写图片上的文字" in text and "图层用途" in text


def test_resolutions_survive_adding_another_image_and_recognizing_again(panel, drawing, tmp_path):
    """真实操作顺序：识别 → 确认材料、补填支座 → 又加一张图（图片证据暂时撤出）→ 重新识别。
    用户处理过的行要找回来，不用从头再点一遍。"""
    from PIL import Image
    panel.recognize()
    _, on_done, _ = panel.runner.jobs[-1]
    evidence = _evidence()
    evidence["support"] = None
    on_done(Recognition(images=ExtractResult(evidence=evidence)))
    _select(panel, "material")
    panel.confirm_selected()
    _select(panel, "support")
    panel.cmb_value.setCurrentText("铰接")
    panel.apply_selected()
    assert panel.assembly.ready
    extra = tmp_path / "立面图.png"
    Image.new("RGB", (800, 600), "white").save(extra)
    panel.load_images([str(extra)])
    assert "images" not in panel.parts and panel.assembly is None
    panel.recognize()
    _, on_done, _ = panel.runner.jobs[-1]
    on_done(Recognition(images=ExtractResult(evidence=evidence)))
    assert panel.assembly.ready, [(r.id, r.status) for r in panel.assembly.pending]
    assert panel.evidence["support"]["kind"] == "pinned"
