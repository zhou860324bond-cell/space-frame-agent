"""参数化建模、七张表、模型 JSON 三个对话框。

这三条入口是**从网页版搬回来的**——迁移到桌面端时我只搬了自然语言一条，
桌面端因此比它要替代的东西功能更少，而我用"预填到输入框"的按钮
把这个洞盖住了。

**这里一律不调 `exec()`。** 模态对话框在 offscreen 环境下不会返回，
表现为整个测试套件卡死——比失败难查得多（我已经踩过一次，套件挂了十分钟）。
所以只构造对话框、直接操作控件、调它的取值与校验函数。
"""

from __future__ import annotations

import json
import os

import pytest

pytest.importorskip("PySide6", reason="未安装 PySide6，跳过桌面端测试")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication                  # noqa: E402

from agent import Session                                   # noqa: E402
from desktop.dialogs import (JsonDialog, ParametricDialog,   # noqa: E402
                             TableDialog)
from desktop.material_dialog import MaterialDialog           # noqa: E402
from desktop.load_dialog import LoadDialog                   # noqa: E402
from desktop.bc_panel import BCPanel                         # noqa: E402
from desktop.bc_dialog import BCDialog                       # noqa: E402
from desktop.assignment_dialog import AssignmentDialog       # noqa: E402
from desktop.section_dialog import SectionDialog, calc_section  # noqa: E402

MATERIALS = [{"name": "Q355", "E": 2.06e11, "nu": 0.3, "density": 7850.0}]
SECTIONS = [{"name": "COLUMN", "A": 0.0147, "Iy": 4.2e-5, "Iz": 1.18e-3,
             "J": 9e-7},
            {"name": "RAFTER", "A": 0.0186, "Iy": 5.2e-5, "Iz": 2.47e-3,
             "J": 1.4e-6}]


@pytest.fixture(scope="module")
def qt_app():
    yield QApplication.instance() or QApplication([])


def defined() -> Session:
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    return s


def built() -> Session:
    s = defined()
    g = s.generate_frame(spans=[6.0, 6.0], storeys=[3.6], column_section="COLUMN",
                         beam_section="RAFTER", material="Q355")
    s.set_load_cases(cases=[{"name": "D", "member_loads":
                             [{"member": m, "w": [0, 0, -20e3]}
                              for m in g.payload["beam_member_ids"]]}])
    return s


# ------------------------------------------------- 参数化建模

def test_unequal_spans_are_supported(qt_app):
    """**不等跨是常态。** 只给一个跨度和一个跨数，就永远做不出
    边跨小、中跨大的常见布置——所以输入是"一串数"而不是"一个数×几跨"。"""
    d = ParametricDialog(defined())
    d.spans.setText("4.5, 9, 4.5")
    d.storeys.setText("4.2, 3.6, 3.6")
    d._accept()
    assert d.result_kind == "frame"
    assert d.params["spans"] == [4.5, 9.0, 4.5]
    assert d.params["storeys"] == [4.2, 3.6, 3.6]


def test_the_parameters_actually_build_a_model(qt_app):
    """对话框取出来的参数要能直接喂给生成器。**这一条最要紧**——
    参数名对不上的话，对话框看着好好的，一点确定就报错。"""
    s = defined()
    d = ParametricDialog(s)
    d.spans.setText("6, 6"); d.storeys.setText("3.6"); d.bays.setText("6")
    d._accept()
    got = s.generate_frame(**d.params)
    assert got.ok, got.payload
    assert len(s.model["nodes"]) > 0


def test_portal_parameters_build_a_portal_frame(qt_app):
    s = defined()
    d = ParametricDialog(s)
    d.kind.setCurrentIndex(1)
    d.p_spans.setText("24"); d.eave.setValue(7.5); d.ridge.setValue(1.2)
    d._accept()
    assert d.result_kind == "portal"
    assert "rafter_section" in d.params and "beam_section" not in d.params
    got = s.generate_portal_frame(**d.params)
    assert got.ok, got.payload


def test_geometry_can_be_parameterised_before_materials_are_defined(qt_app):
    s = Session()
    d = ParametricDialog(s)
    d.spans.setText("6, 9")
    d.storeys.setText("3.6")
    d._accept()

    assert d.params["material"] == ""
    assert d.params["column_section"] == ""
    result = s.generate_frame(**d.params)
    assert result.ok
    assert not result.payload["analysis_ready"]
    assert s.model["nodes"] and s.model["members"]


def test_irregular_parameter_page_builds_stepped_multi_bent_frame(qt_app):
    s = defined()
    d = ParametricDialog(s)
    d.kind.setCurrentIndex(2)
    d.b_profile.setText("-2:7.2, 6:8.4, 15:7.8")
    d.b_columns.setText("0, 6, 10, 15")
    d.b_levels.setText("3.6")
    d.b_base_levels.setText("0, 0, 1.2, 1.2")
    d.b_bays.setText("4.5, 6")
    d._accept()

    assert d.result_kind == "bent"
    bays = d.params.pop("bays")
    generated = s.generate_bent(**d.params)
    assert generated.ok, generated.payload
    extruded = s.extrude_bents(bays, tie_section=d.params["beam_section"])
    assert extruded.ok, extruded.payload
    assert extruded.payload["bent_count"] == 3
    assert len({node["y"] for node in s.model["nodes"]}) == 3


def test_irregular_dialog_treats_levels_as_absolute_coordinates(qt_app):
    d = ParametricDialog(defined())
    d.kind.setCurrentIndex(2)
    d.b_profile.setText("0:3, 6:3")
    d.b_columns.setText("0, 6")
    d.b_levels.setText("-1, 0, 1.5")
    d.b_base_levels.setText("-3, -3")
    d.b_bays.clear()
    d._accept()
    assert d.params["levels"] == [-1.0, 0.0, 1.5]


def test_a_bad_number_names_the_offending_value(qt_app, monkeypatch):
    """只说"格式错误"，用户得自己逐个找。**要指出是哪一项。**"""
    from PySide6.QtWidgets import QMessageBox
    shown = []
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *a, **k: shown.append(a[2]))
    d = ParametricDialog(defined())
    d.spans.setText("6, 六, 6")
    d._accept()
    assert shown and "六" in shown[0], shown


def test_a_negative_span_is_refused(qt_app, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    shown = []
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *a, **k: shown.append(a[2]))
    d = ParametricDialog(defined())
    d.spans.setText("6, -3")
    d._accept()
    assert shown and "大于 0" in shown[0]


def test_a_full_width_comma_works_too(qt_app):
    """中文输入法下打出来的是全角逗号。**不认它就是在为难用户。**"""
    d = ParametricDialog(defined())
    d.spans.setText("6，6，6")
    d._accept()
    assert d.params["spans"] == [6.0, 6.0, 6.0]


def test_the_load_is_converted_to_newtons(qt_app):
    """界面填 kN/m，工具收 N/m。**换算漏了就是一千倍的错**，
    而且模型照样算得出来，只是结果全错。"""
    d = ParametricDialog(defined())
    d.quick_setup.setChecked(True)
    d.beam_load.setValue(20.0)
    d._accept()
    assert d.params["beam_load"] == pytest.approx(20000.0)


def test_the_two_generators_use_different_parameter_names(qt_app):
    """框架用 beam_section / beam_load，门式用 rafter_section / rafter_load。

    **这类映射最容易漏。** 漏了之后界面看着好好的，一点确定就报
    "unexpected keyword argument"——用户完全不知道发生了什么。
    这条测试是被这个 bug 逼出来的。
    """
    for index, expect_section, expect_load in ((0, "beam_section", "beam_load"),
                                               (1, "rafter_section",
                                                "rafter_load")):
        d = ParametricDialog(defined())
        d.quick_setup.setChecked(True)
        d.kind.setCurrentIndex(index)
        d.beam_load.setValue(20.0)
        d._accept()
        assert expect_section in d.params, d.params
        assert expect_load in d.params, d.params


def test_zero_load_is_omitted_not_sent_as_zero(qt_app):
    d = ParametricDialog(defined())
    d.beam_load.setValue(0.0)
    d._accept()
    assert "beam_load" not in d.params


def test_parametric_dialog_does_not_invent_properties_or_loads(qt_app):
    d = ParametricDialog(defined())
    assert not d.material.isEditable()
    assert not d.col_sec.isEditable()
    assert not d.beam_sec.isEditable()
    assert d.beam_load.value() == 0.0
    d._accept()
    assert d.params["material"] == ""
    assert d.params["column_section"] == ""
    assert d.params["beam_section"] == ""
    assert d.params["base"] == "free"


def test_geometry_first_model_has_no_implicit_column_base_condition(qt_app):
    s = defined()
    d = ParametricDialog(s)
    d._accept()
    assert s.generate_frame(**d.params).ok
    # 单榀二维建模所需的面外运动约束可以存在，但不得偷偷固定柱脚平面内自由度。
    assert all(fix[0] == 0 and fix[2] == 0 and fix[4] == 0
               for fix in (entry["fix"] for entry in s.model["supports"]))


def test_property_dialogs_follow_the_mm_unit_system(qt_app):
    material = MaterialDialog(units="N-mm-MPa")
    assert material.spn_E.suffix().strip() == "MPa"
    assert material.spn_E.value() == pytest.approx(206000.0)
    assert material.spn_density.value() == pytest.approx(7.85e-9)

    section = SectionDialog(units="N-mm-MPa")
    assert section.param_spins["I 型钢"]["h"].suffix().strip() == "mm"
    assert section.param_spins["I 型钢"]["h"].value() == pytest.approx(300.0)
    assert section.get_section()["A"] > 1000.0


def test_small_sections_are_not_rounded_to_zero():
    props = calc_section("圆钢", {"d": 0.001})
    assert props["Iy"] > 0.0 and props["J"] > 0.0


def test_load_dialog_makes_name_and_analysis_step_explicit(qt_app):
    d = LoadDialog("member", 7, units="N-mm-MPa",
                   cases=["Dead", "Wind"], current_case="Wind")
    assert d.get_name() == "Line-Member-7"
    assert d.get_case() == "Wind"
    assert d.ll_widget.spn_p3.suffix().strip() == "N/mm"


def test_member_load_dialog_supports_real_span_load_shapes(qt_app):
    d = LoadDialog("member", 7, units="N-m-Pa", cases=["Dead"])
    assert [d.cmb_type.itemData(i) for i in range(d.cmb_type.count())] == [
        "line", "trapezoid", "member_point"]

    d.cmb_type.setCurrentIndex(d.cmb_type.findData("trapezoid"))
    d.trap_widget.start_spins[2].setValue(-10.0)
    d.trap_widget.end_spins[2].setValue(-20.0)
    assert d.get_load() == [0.0, 0.0, -10.0]
    assert d.get_end_load() == [0.0, 0.0, -20.0]

    d.cmb_type.setCurrentIndex(d.cmb_type.findData("member_point"))
    d.point_widget.force_spins[2].setValue(-5000.0)
    d.point_widget.spn_a.setValue(2.5)
    assert d.get_load() == [0.0, 0.0, -5000.0]
    assert d.get_position() == pytest.approx(2.5)


def test_bc_panel_separates_initial_support_and_step_displacement(qt_app):
    s = built()
    panel = BCPanel(s)
    panel.set_selection("node", 1)
    panel.cmb_case.setCurrentText("D")
    panel.txt_settlement_name.setText("Settlement-Base-1")
    panel.settlement_spins[2].setValue(-0.005)
    panel._apply_settlement()
    entry = s.model["load_cases"][0]["settlements"][0]
    assert entry["name"] == "Settlement-Base-1"
    assert entry["node"] == 1
    assert entry["d"][2] == pytest.approx(-0.005)


def test_free_support_preset_really_removes_the_support(qt_app):
    s = built()
    panel = BCPanel(s)
    panel.set_selection("node", 1)
    panel.cmb_support.setCurrentIndex(panel.cmb_support.count() - 1)
    panel._apply_support()
    assert all(int(item["node"]) != 1 for item in s.model["supports"])


def test_initial_bc_dialog_has_an_explicit_editable_name(qt_app):
    dialog = BCDialog(8, [1, 1, 1, 0, 0, 0], current_name="Base-Pin-8")
    assert dialog.get_name() == "Base-Pin-8"
    assert dialog.get_fix() == [1, 1, 1, 0, 0, 0]


def test_property_assignment_selects_material_and_section_together(qt_app):
    dialog = AssignmentDialog(
        [3], ["COLUMN", "RAFTER"], ["Q235", "Q355"],
        current_section="RAFTER", current_material="Q355")
    assert dialog.get_assignment() == ("RAFTER", "Q355")


# ------------------------------------------------- 七张表

def test_all_seven_tables_are_present(qt_app):
    from model_tables import LABELS, TABLE_NAMES

    d = TableDialog(built())
    assert d.tabs.count() == len(TABLE_NAMES)
    titles = [d.tabs.tabText(i) for i in range(d.tabs.count())]
    for name in TABLE_NAMES:
        assert any(LABELS[name] in t for t in titles), name


def test_empty_tables_still_show_their_headers(qt_app):
    """`to_tables` 对空表返回空列表，拿不到列名。**表头必须照样画出来**，
    否则用户不知道该往哪一列填什么——这正是 model_tables 里
    补 COLUMNS 映射的原因。"""
    from model_tables import COLUMNS

    d = TableDialog(defined())              # 只有材料截面，几何表全空
    for name, table in d.tables.items():
        headers = [table.horizontalHeaderItem(c).text()
                   for c in range(table.columnCount())]
        assert headers == list(COLUMNS[name]), name


def test_editing_a_coordinate_writes_back(qt_app):
    s = built()
    d = TableDialog(s)
    nodes = d.tables["nodes"]
    nodes.item(0, 3).setText("1.5")         # 把 1 号节点抬高
    d._accept()
    assert d.model is not None
    moved = [n for n in d.model["nodes"] if n["id"] == 1][0]
    assert moved["z"] == pytest.approx(1.5)


def test_a_broken_table_does_not_overwrite_the_model(qt_app, monkeypatch):
    """**先校验再落地。** `set_model` 是故意先存后验的（Agent 要靠它做
    增量修补），但人工编辑不能这样：一份改坏的表把好模型顶掉，
    用户就只能靠撤销找回来。"""
    from PySide6.QtWidgets import QMessageBox
    shown = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: shown.append(a))

    s = built()
    before = json.dumps(s.model, sort_keys=True)
    d = TableDialog(s)
    d.tables["nodes"].item(0, 1).setText("不是数字")
    d._accept()
    assert shown, "该提示表格有误"
    assert d.model is None, "校验没过就不该产出模型"
    assert json.dumps(s.model, sort_keys=True) == before, "原模型被改坏了"


def test_a_blank_row_means_deletion(qt_app):
    """整行清空即删除——比另设一个删除列直观。"""
    s = built()
    d = TableDialog(s)
    n = len(s.model["members"])
    members = d.tables["members"]
    for c in range(members.columnCount()):
        item = members.item(members.rowCount() - 4, c)   # 最后一行真实数据
        if item:
            item.setText("")
    rows = d._rows_of("members")
    assert len(rows) == n - 1


# ------------------------------------------------- 模型 JSON

def test_json_round_trips(qt_app):
    s = built()
    d = JsonDialog(s)
    d._accept()
    assert d.model is not None
    assert d.model["nodes"] == s.model["nodes"]


def test_bad_json_reports_line_and_column(qt_app, monkeypatch):
    """只说"格式错误"没用——JSON 手改最常见的就是漏个逗号，
    **得告诉他在第几行**。"""
    from PySide6.QtWidgets import QMessageBox
    shown = []
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *a, **k: shown.append(a[2]))
    d = JsonDialog(built())
    d.edit.setPlainText('{\n  "nodes": [1,2\n}')
    d._accept()
    assert shown and "行" in shown[0] and d.model is None


def test_json_that_fails_validation_is_refused(qt_app, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    shown = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: shown.append(a))
    d = JsonDialog(built())
    d.edit.setPlainText('{"nodes": [], "members": [], "supports": []}')
    d._accept()
    assert shown and d.model is None
