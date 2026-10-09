"""属性面板：点中 → 改 → 模型变 → 结果失效 → 能撤销。

**这里盯的是「界面会不会说谎」。** 面板的失败模式不是崩溃，是显示着
一个模型里并不存在的值：截面名打错了，提交被拒，控件却还停在新值上，
用户以为改成功了，直到重新求解才发现结果和预期对不上。

所以每个测试都对着 `session.model` 断言，而不是对着控件断言——
控件显示什么不重要，模型里是什么才重要。
"""

from __future__ import annotations

import os

import pytest

from agent import Session
from sections import i_section

HAS_QT = True
try:                                    # 没装 Qt 时只跳过要开窗的测试
    from PySide6.QtWidgets import QApplication
except Exception:  # noqa: BLE001  import 探测：没装 Qt 就跳过要开窗的测试  pragma: no cover
    HAS_QT = False

needs_qt = pytest.mark.skipif(not HAS_QT, reason="未安装 PySide6")

MAT = [{"name": "Q355", "E": 2.06e11, "nu": 0.3, "density": 7850.0}]
SEC = [i_section("COLUMN", 0.4, 0.2, 0.008, 0.012),
       i_section("BEAM", 0.5, 0.2, 0.008, 0.014),
       i_section("BRACE", 0.15, 0.15, 0.006, 0.008)]


def build() -> Session:
    s = Session()
    s.define_materials_and_sections(MAT, SEC)
    g = s.generate_frame(spans=[6.0], storeys=[3.6], column_section="COLUMN",
                         beam_section="BEAM", material="Q355")
    s.set_load_cases(cases=[{"name": "D", "member_loads":
                             [{"member": m, "w": [0, 0, -20e3]}
                              for m in g.payload["beam_member_ids"]]}])
    s.solve_model()
    return s


@pytest.fixture(scope="module")
def app():
    if not HAS_QT:
        pytest.skip("未安装 PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


def panel(session):
    from desktop.properties import PropertiesPanel
    return PropertiesPanel(session)


@needs_qt
def test_selecting_a_member_shows_editable_properties(app):
    s = build()
    p = panel(s)
    mid = s.model["members"][0]["id"]
    p.show_object("member", mid)
    assert p.title.text() == f"杆件 {mid}"
    # 截面和材料可改；端点和长度是几何，改这两个是移动节点，不在这里
    assert set(p._widgets) == {"section", "material", "ref_vector",
                               "offset_i", "offset_j", "mu_y", "mu_z",
                               "release_i", "release_j"}
    assert p._widgets["section"].currentText() == s.model["members"][0]["section"]


@needs_qt
def test_editing_a_section_changes_the_model_and_kills_the_result(app):
    s = build()
    assert s.solution is not None
    p = panel(s)
    mid = s.model["members"][0]["id"]
    p.show_object("member", mid)

    p._widgets["section"].setCurrentText("BRACE")

    entry = next(m for m in s.model["members"] if m["id"] == mid)
    assert entry["section"] == "BRACE"
    assert s.solution is None, "刚度变了，旧位移必须作废，否则变形图画的是别的结构"
    assert "已修改" in p.status.text()


@needs_qt
def test_a_manual_edit_is_undoable(app):
    """手工改的和 Agent 改的必须进同一条撤销链。"""
    s = build()
    p = panel(s)
    mid = s.model["members"][0]["id"]
    was = next(m for m in s.model["members"] if m["id"] == mid)["section"]
    p.show_object("member", mid)
    p._widgets["section"].setCurrentText("BRACE")

    assert "BRACE" in s.history.steps[-1].summary
    assert s.undo()
    assert next(m for m in s.model["members"] if m["id"] == mid)["section"] == was


@needs_qt
def test_a_rejected_edit_reverts_the_widget_instead_of_lying(app):
    """**提交失败时控件必须退回去。**

    面板显示 BRACE 而模型里还是 COLUMN，是最坏的一种失败：
    没有任何报错，用户以为改了。
    """
    s = build()
    p = panel(s)
    mid = s.model["members"][0]["id"]
    p.show_object("member", mid)

    # 往下拉框里塞一个模型里不存在的截面名，模拟一次被拒的提交
    p._widgets["section"].addItem("未定义截面")
    p._widgets["section"].setCurrentText("未定义截面")

    assert next(m for m in s.model["members"] if m["id"] == mid)["section"] == "COLUMN"
    assert p.status.text().startswith("未生效")
    assert "COLUMN" in p.status.text() or "BEAM" in p.status.text(), "要列出可选值"
    assert p._widgets["section"].currentText() == "COLUMN", "控件没退回去就是在说谎"


@needs_qt
def test_support_and_release_presets_write_the_right_arrays(app):
    s = build()
    p = panel(s)
    nid = s.model["nodes"][0]["id"]
    p.show_object("node", nid)
    assert p._widgets["support"].currentText() == "固接"

    p._widgets["support"].setCurrentText("铰接")
    fix = next(x["fix"] for x in s.model["supports"] if x["node"] == nid)
    assert fix == [1, 1, 1, 0, 0, 0]

    mid = s.model["members"][0]["id"]
    p.show_object("member", mid)
    p._widgets["release_j"].setCurrentText("平面内铰接")
    rel = next(m for m in s.model["members"] if m["id"] == mid)["releases"]
    assert rel == {"j": ["rz"]}
    # 再改回刚接 = 取消释放，不是留一个空列表
    p._widgets["release_j"].setCurrentText("刚接")
    assert "releases" not in next(m for m in s.model["members"] if m["id"] == mid)


@needs_qt
def test_member_reference_vector_is_validated_and_saved(app):
    """非对称截面要能明确梁方向，且平行向量不能悄悄写进模型。"""
    s = build()
    p = panel(s)
    mid = s.model["members"][0]["id"]
    p.show_object("member", mid)

    box = p._widgets["ref_vector"]
    box.setText("0, 1, 0")
    box.editingFinished.emit()
    assert next(m for m in s.model["members"] if m["id"] == mid)["ref_vector"] == [0.0, 1.0, 0.0]

    box.setText("1, 2")
    box.editingFinished.emit()
    assert "三个分量" in p.status.text()
    assert p._widgets["ref_vector"].text() == "0, 1, 0"


@needs_qt
def test_moving_a_node_commits_once_not_per_keystroke(app):
    s = build()
    p = panel(s)
    nid = s.model["nodes"][-1]["id"]
    p.show_object("node", nid)
    before = len(s.history.steps)

    box = p._widgets["x"]
    assert not box.keyboardTracking(), "边打边提交会在建模过程里堆一串垃圾步骤"
    box.setValue(7.5)
    box.editingFinished.emit()

    assert next(n for n in s.model["nodes"] if n["id"] == nid)["x"] == 7.5
    assert len(s.history.steps) == before + 1


@needs_qt
def test_mm_model_coordinate_editor_is_labelled_in_mm(app):
    s = build()
    assert s.set_units("N-mm-MPa").ok
    p = panel(s)
    p.show_object("node", s.model["nodes"][0]["id"])
    assert p.form.labelForField(p._widgets["x"]).text().endswith("(mm)")


@needs_qt
def test_refilling_the_form_does_not_pollute_the_history(app):
    """重新显示同一个对象不该记账——面板每次刷新都会重填控件。"""
    s = build()
    p = panel(s)
    mid = s.model["members"][0]["id"]
    p.show_object("member", mid)
    p._widgets["section"].setCurrentText("BRACE")
    n = len(s.history.steps)

    for _ in range(3):
        p.show_object("member", mid)
        p._widgets["section"].setCurrentText("BRACE")   # 同值再提交
    assert len(s.history.steps) == n


@needs_qt
def test_the_panel_clears_when_the_selected_object_is_deleted(app):
    s = build()
    p = panel(s)
    mid = s.model["members"][0]["id"]
    p.show_object("member", mid)
    s.remove_members(ids=[mid])
    p.refresh()
    assert p.kind is None and p.title.text() == "未选中对象"


@needs_qt
def test_the_effective_length_factor_can_be_set_and_cleared(app):
    """有侧移的框架柱 μ>1，推定值偏小、判定偏不安全。这一栏必须能填。"""
    s = build()
    p = panel(s)
    mid = s.model["members"][0]["id"]
    p.show_object("member", mid)

    p._widgets["mu_y"].setText("2")
    p._commit("mu_y", "2")
    entry = next(m for m in s.model["members"] if m["id"] == mid)
    assert entry["mu_y"] == 2.0

    p._commit("mu_y", "")                 # 清空 = 回到按杆端约束推定
    entry = next(m for m in s.model["members"] if m["id"] == mid)
    assert "mu_y" not in entry


@needs_qt
def test_a_nonsense_effective_length_factor_is_refused_and_rolled_back(app):
    """提交失败要把界面退回去。模型没变而面板显示着新值，就是界面在说谎。"""
    s = build()
    p = panel(s)
    mid = s.model["members"][0]["id"]
    p.show_object("member", mid)
    p._commit("mu_z", "负一")
    assert "计算长度系数" in p.status.text()
    entry = next(m for m in s.model["members"] if m["id"] == mid)
    assert "mu_z" not in entry


@needs_qt
def test_advanced_fields_remain_accessible_and_refresh_preserves_expansion(app):
    """常用截面编辑不应被高级参数淹没，同一对象刷新也不能关掉正在编辑的分组。"""
    s = build()
    p = panel(s)
    mid = s.model["members"][0]["id"]
    p.show_object("member", mid)
    assert p.advanced_host.isHidden()
    assert p.form.indexOf(p._widgets["section"]) >= 0
    assert p.advanced_host.isAncestorOf(p._widgets["release_j"])
    p.advanced_button.click()
    p.refresh()
    assert p.advanced_button.isChecked()
    assert not p.advanced_host.isHidden()
    p.show_object("node", s.model["nodes"][0]["id"])
    assert p.advanced_button.isHidden()
    assert p.advanced_host.isHidden()


@needs_qt
@pytest.mark.parametrize("key,value", [
    ("mu_y", "nan"), ("mu_z", "inf"), ("mu_y", "0"),
    ("ref_vector", "0, nan, 1"), ("offset_i", "inf, 0, 0"),
    ("offset_j", "1, 2")])
def test_invalid_advanced_input_keeps_solution_and_reports_beside_field(app, key, value):
    """非法高级输入不能污染模型或使有效结果失效，原因应显示在恢复原值的输入旁。"""
    s = build()
    p = panel(s)
    p.show_object("member", s.model["members"][0]["id"])
    before = len(s.history.steps)
    solution = s.solution
    original = p._widgets[key].text()
    p._commit(key, value)
    assert s.solution is solution
    assert len(s.history.steps) == before
    assert p._widgets[key].text() == original
    assert p._widgets[key].property("invalid")
    assert not p._errors[key].isHidden()
    assert "已恢复原值" in p._errors[key].text()
    assert p.advanced_button.isChecked()
    p._commit(key, original)
    assert not p._widgets[key].property("invalid")
    assert p._errors[key].isHidden()
    assert not p.status.text()


@needs_qt
def test_failed_edit_rebuilds_without_stale_rows_and_scrolls_to_error(app):
    """窄屏拒绝提交后旧标签不能叠在新字段上，折叠区域的错误必须滚入可见范围。"""
    from PySide6.QtWidgets import QScrollArea
    from PySide6.QtTest import QTest
    from PySide6.QtCore import Qt
    from desktop import theme
    s = build()
    p = panel(s)
    p.setStyleSheet(theme.STYLESHEET)
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(p)
    area.resize(300, 360)
    p.show_object("member", s.model["members"][0]["id"])
    rows = (p.form.rowCount(), p.advanced_form.rowCount())
    area.show()
    for _ in range(3):
        field = p._widgets["mu_y"]
        field.setFocus()
        field.selectAll()
        QTest.keyClicks(field, "nan")
        QTest.keyClick(field, Qt.Key.Key_Return)
        for _ in range(5):
            app.processEvents()
        QTest.qWait(100)
        assert (p.form.rowCount(), p.advanced_form.rowCount()) == rows
        error = p._errors["mu_y"]
        position = error.mapTo(area.viewport(), error.rect().center())
        assert area.viewport().rect().contains(position), (position, p.size(), area.verticalScrollBar().value(), p._error_key, p._error_timer.isActive(), p.isVisible(), error.isHidden())
    fields = [p._widgets[k].parentWidget() for k in ("ref_vector", "offset_i", "offset_j", "mu_y")]
    assert all(a.geometry().bottom() < b.y() for a, b in zip(fields, fields[1:], strict=False))
    QTest.qWait(100)
    assert not p._error_timer.isActive(), "错误已可见后，定位不能继续反复改变表单高度"
    area.close()
