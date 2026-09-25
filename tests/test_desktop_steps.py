"""分析步与幅值曲线对话框。

这两个对话框接的是一条**已经存在但界面够不到**的通路：内核早就支持跨步
传播与失活，而「分析步」按钮做的却是选求解过程。这类"能力只交付了一半"
的缺口自己不会报错——按钮在、能点、弹出来也正常，只是点不到真正的功能。

所以这里验的不是"能不能弹出来"，是**弹出来的东西是不是接在真东西上**：
表格里的数必须来自 Session 的结算结果，按钮按下去必须真的改到模型。
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6", reason="未安装 PySide6，跳过桌面端测试")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication              # noqa: E402

from agent import Session                                # noqa: E402
from desktop.step_dialog import (AmplitudeDialog,        # noqa: E402
                                 StepManagerDialog)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def loaded_session():
    s = Session()
    s.add_nodes([[0, 0, 0], [6, 0, 0], [12, 0, 0]])
    s.define_materials_and_sections(
        materials=[{"name": "M", "E": 2.1e11, "nu": 0.3}],
        sections=[{"name": "S", "A": 0.02, "Iy": 2e-4, "Iz": 4e-4, "J": 1e-5}])
    s.add_members([[1, 2], [2, 3]], "S", "M")
    s.set_supports([1], fix=[1, 1, 1, 1, 1, 1])
    s.set_supports([2], fix=[0, 0, 1, 0, 0, 0])
    s.set_supports([3], fix=[0, 0, 1, 0, 0, 0])
    s.add_load_case("DL")
    s.add_load_case("LL")
    for member in (1, 2):
        s.set_member_load(member, [0, 0, -10e3], case_name="DL")
        s.set_member_load(member, [0, 0, -15e3], case_name="LL")
    return s


# --- 幅值曲线 -------------------------------------------------------------

def test_the_amplitude_dialog_lists_the_builtin_curves(app):
    """内置两条必须列出来并说明含义——RAMP 和 STEP 的区别最容易选错。"""
    session = loaded_session()
    dialog = AmplitudeDialog(session)
    names = [dialog.table.item(r, 0).text()
             for r in range(dialog.table.rowCount())]
    assert {"RAMP", "STEP"} <= set(names)
    row = names.index("STEP")
    assert dialog.table.item(row, 2).text(), "内置曲线必须带一句说明"


def test_creating_a_curve_in_the_dialog_reaches_the_model(app):
    """按下"新建"要真的改到模型，不能只往表格里塞一行。"""
    session = loaded_session()
    dialog = AmplitudeDialog(session)
    dialog.name_edit.setText("前段加满")
    dialog.points_edit.setText("0,0; 0.3,1; 1,1")
    dialog._add()
    assert session.model["amplitudes"]["前段加满"] == [[0.0, 0.0], [0.3, 1.0],
                                                       [1.0, 1.0]]
    names = [dialog.table.item(r, 0).text()
             for r in range(dialog.table.rowCount())]
    assert "前段加满" in names


def test_a_malformed_table_does_not_reach_the_model(app, monkeypatch):
    """输错格式要挡在对话框里，不能把半截数据写进模型。"""
    import desktop.step_dialog as module
    monkeypatch.setattr(module.QMessageBox, "warning",
                        staticmethod(lambda *a, **k: None))
    session = loaded_session()
    dialog = AmplitudeDialog(session)
    dialog.name_edit.setText("坏的")
    dialog.points_edit.setText("0,0; 这不是数")
    dialog._add()
    assert "坏的" not in (session.model.get("amplitudes") or {})


# --- 分析步 ---------------------------------------------------------------

def test_the_step_table_shows_what_actually_takes_effect(app):
    """**这是整个对话框的要点。**

    第二步只声明了一行"失活 LL"，而实际在算的是第一步传下来的 DL。
    表格里只给声明的话，这件事就只能靠人自己推——推错了没有任何提示。
    """
    session = loaded_session()
    session.add_step("满载", loads={"DL": "RAMP", "LL": "RAMP"})
    session.add_step("撤活载", deactivate_loads=["LL"])
    dialog = StepManagerDialog(session)

    assert dialog.table.rowCount() == 2
    declared_second = dialog.table.item(1, 2).text()
    effective_second = dialog.table.item(1, 3).text()
    assert "失活" in declared_second
    assert "DL" in effective_second, "传播过来的 DL 必须出现在生效栏里"
    assert "LL" not in effective_second


def test_selecting_a_step_explains_what_changed(app):
    """选中一行要说清楚这一步相对上一步差在哪。"""
    session = loaded_session()
    session.add_step("满载", loads={"DL": "RAMP"})
    session.add_step("拆中支座", deactivate_supports=[2])
    dialog = StepManagerDialog(session)
    dialog.table.selectRow(1)
    assert "失活支座" in dialog.changes.text()


def test_creating_a_step_in_the_dialog_reaches_the_model(app):
    session = loaded_session()
    dialog = StepManagerDialog(session)
    dialog.name_edit.setText("满载")
    dialog.case_box.setCurrentText("DL")
    dialog.amp_box.setCurrentText("RAMP")
    dialog._add()
    assert [s["name"] for s in session.model["steps"]] == ["满载"]
    assert dialog.table.rowCount() == 1


def test_a_step_that_cannot_be_resolved_is_reported_not_written(app, monkeypatch):
    """失活一个没生效的工况要弹错误并保持模型原样。"""
    import desktop.step_dialog as module
    seen = []
    monkeypatch.setattr(module.QMessageBox, "warning",
                        staticmethod(lambda *a, **k: seen.append(a)))
    session = loaded_session()
    session.add_step("满载", loads={"DL": "RAMP"})
    dialog = StepManagerDialog(session)
    dialog.name_edit.setText("撤个没有的")
    dialog.deactivate_cases.setText("LL")
    dialog._add()
    assert seen, "结算不出来时必须弹错误，不能静静地什么都不发生"
    assert [s["name"] for s in session.model["steps"]] == ["满载"]


def test_the_dialog_does_not_offer_material_nonlinearity(app):
    """材料非线性进不了分析步，下拉框里就不该出现它。

    列出来再在求解时报错，等于让用户白填一遍。
    """
    from desktop.step_dialog import ANALYSES
    labels = [dialog_label for dialog_label, _ in ANALYSES]
    assert "材料" not in "".join(labels)
    assert [kernel for _, kernel in ANALYSES] == ["linear", "pdelta"]


def test_solving_from_the_dialog_reports_every_step(app, monkeypatch):
    """求解按钮要把每一步的结果都报出来，并且带上那条能力边界。"""
    import desktop.step_dialog as module
    shown = []
    monkeypatch.setattr(module.QMessageBox, "information",
                        staticmethod(lambda *a, **k: shown.append(a[2])))
    session = loaded_session()
    session.add_step("满载", loads={"DL": "RAMP", "LL": "RAMP"})
    session.add_step("撤活载", deactivate_loads=["LL"])
    dialog = StepManagerDialog(session)
    dialog._solve()

    assert shown, "求解完必须给用户一个结果摘要"
    text = shown[0]
    assert "满载" in text and "撤活载" in text
    assert "未变形" in text, "「每步从零重解」这条边界必须一起交付，不能只在文档里"


# --- 通用闸门：Markdown 不许漏到界面上 -------------------------------------

def test_no_dialog_shows_literal_markdown_asterisks(app):
    """`**重点**` 必须渲染成加粗，不能把星号原样摆给用户看。

    这个项目的中文说明一律按 Markdown 风格写——docstring、工具描述、返回值
    里的 note 全是 `**这样**`。同一句话复制到界面上时星号会原样显示，不崩、
    不报错，只是难看，所以没人会专门去测。`result_rows.py` 早就为此写过一行
    `replace("**", "")`，说明这个坑踩过一次了。

    这里把它变成闸门：任何对话框里只要还有带字面星号的标签，就红。
    """
    from PySide6.QtWidgets import QLabel

    session = loaded_session()
    session.add_step("满载", loads={"DL": "RAMP"})
    leaked = []
    for cls in (StepManagerDialog, AmplitudeDialog):
        dialog = cls(session)
        leaked += [f"{cls.__name__}: {label.text()[:50]}"
                   for label in dialog.findChildren(QLabel)
                   if "**" in label.text()]
    assert not leaked, f"这些标签把 Markdown 星号漏给用户了：{leaked}"


def test_emphasis_turns_double_stars_into_bold():
    from desktop.dialog_styles import emphasis

    assert emphasis("漏写**不等于**撤销") == "漏写<b>不等于</b>撤销"
    # 星号不成对时原样退回：猜"大概想加粗哪里"只会更奇怪
    assert emphasis("三个**星号**在**这") == "三个**星号**在**这"
    # HTML 必须转义，否则说明文字里一个尖括号就能改掉整段排版
    assert emphasis("a < b & c") == "a &lt; b &amp; c"


def test_the_table_does_not_show_python_reprs(app):
    """表格里不许出现 {'DL': 'RAMP'} 这种东西。

    直接 str() 一个 dict，引号和花括号会占掉本来就不宽的列，真正要看的
    工况名反而被省略号吃掉——而且它把实现细节漏给了用户。
    """
    session = loaded_session()
    session.add_step("满载", loads={"DL": "RAMP", "LL": "RAMP"})
    dialog = StepManagerDialog(session)
    texts = [dialog.table.item(0, c).text()
             for c in range(dialog.table.columnCount())]
    joined = "".join(texts)
    for forbidden in ("{", "}", "'", "[", "]"):
        assert forbidden not in joined, (
            f"表格里出现了 Python repr 的 {forbidden!r}：{texts}")
    assert "DL=RAMP" in joined


def test_the_table_shows_the_display_name_of_the_analysis(app):
    """分析类型那栏要显示"线性静力"，不是内核里的 "linear"。"""
    session = loaded_session()
    session.add_step("满载", loads={"DL": "RAMP"}, analysis="pdelta")
    dialog = StepManagerDialog(session)
    assert dialog.table.item(0, 1).text() == "P-Delta 二阶弹性"


def test_every_cell_carries_its_full_text_as_a_tooltip(app):
    """列宽有限，生效的荷载一多就会被省略号吃掉——悬停必须给全文。"""
    session = loaded_session()
    session.add_step("满载", loads={"DL": "RAMP", "LL": "RAMP"})
    dialog = StepManagerDialog(session)
    for c in range(dialog.table.columnCount()):
        item = dialog.table.item(0, c)
        assert item.toolTip() == item.text()
