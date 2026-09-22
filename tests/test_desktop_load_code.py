"""规范组合、活载布置、面荷载、边界条件管理器四个对话框。

这四样求解侧早就有了，界面一直够不到。验的不是"弹不弹得出来"，是
**弹出来的东西是不是接在真东西上**：按钮按下去必须真的改到模型，
表格里的数必须来自会话返回值。

另外盯住一件容易搞错的事：这些工具的返回值字段名并不像看上去那么直觉
（``entries`` 是段数不是列表、活载的 ``cases`` 是列表不是字典、
边界条件是 ``nodes`` 不是 ``node``）。对话框照着猜的字段名取值时不会报错，
只会显示空白——所以这里逐个对着真实返回值断言。
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6", reason="未安装 PySide6，跳过桌面端测试")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication                    # noqa: E402

from agent import Session                                      # noqa: E402
from desktop.load_code_dialog import (AreaLoadDialog,          # noqa: E402
                                      BCManagerDialog,
                                      CombinationDialog,
                                      LivePatternDialog)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def quiet(monkeypatch):
    """把弹窗静音，并把弹出来的正文收集起来供断言。"""
    import desktop.load_code_dialog as module
    shown: dict[str, list] = {"info": [], "warn": []}
    monkeypatch.setattr(module.QMessageBox, "information",
                        staticmethod(lambda *a, **k: shown["info"].append(a[2])))
    monkeypatch.setattr(module.QMessageBox, "warning",
                        staticmethod(lambda *a, **k: shown["warn"].append(a[2])))
    return shown


def beam_session():
    """三跨连续梁，跨的归类推得出来。"""
    s = Session()
    s.add_nodes([[0, 0, 0], [6, 0, 0], [12, 0, 0], [18, 0, 0]])
    s.define_materials_and_sections(
        materials=[{"name": "M", "E": 2.1e11, "nu": 0.3}],
        sections=[{"name": "S", "A": 0.02, "Iy": 2e-4, "Iz": 4e-4, "J": 1e-5}])
    s.add_members([[1, 2], [2, 3], [3, 4]], "S", "M")
    s.set_supports([1], fix=[1, 1, 1, 0, 0, 0])
    for node in (2, 3, 4):
        s.set_supports([node], fix=[0, 0, 1, 0, 0, 0])
    s.add_load_case("DL")
    s.add_load_case("WX")
    for member in (1, 2, 3):
        s.set_member_load(member, [0, 0, -10e3], case_name="DL")
    s.set_nodal_load(4, [20e3, 0, 0, 0, 0, 0], case_name="WX")
    return s


# --- 规范组合 -------------------------------------------------------------

def test_combinations_reach_the_model_and_are_shown(app, quiet):
    session = beam_session()
    dialog = CombinationDialog(session)
    dialog.boxes["dead"].setCurrentText("DL")
    dialog.boxes["wind"].setCurrentText("WX")
    dialog._generate()

    assert session.model["combos"], "组合必须真的写进模型"
    assert dialog.table.rowCount() == len(session.model["combos"])
    shown = "".join(quiet["info"])
    assert "包络只在这些组合里取极值" in shown, (
        "「漏一个组合看不出来」这句提醒必须一起交付")


def test_the_combination_table_shows_the_factors_not_a_dict(app, quiet):
    """系数那栏要写成 1.3×DL + 1.5×WX，不是 Python 字典。"""
    session = beam_session()
    dialog = CombinationDialog(session)
    dialog.boxes["dead"].setCurrentText("DL")
    dialog.boxes["wind"].setCurrentText("WX")
    dialog._generate()
    factors = dialog.table.item(0, 1).text()
    assert "×" in factors and "{" not in factors and "'" not in factors
    assert dialog.table.item(0, 2).text(), "依据那一栏不能是空的"


def test_generating_with_nothing_picked_is_refused(app, quiet):
    session = beam_session()
    dialog = CombinationDialog(session)
    dialog._generate()
    assert not session.model.get("combos")
    assert quiet["info"], "什么都没选时要提示，不能静静地什么都不做"


# --- 活载布置 -------------------------------------------------------------

def test_live_patterns_reach_the_model_and_list_the_cases(app, quiet):
    session = beam_session()
    before = len(session.model["load_cases"])
    dialog = LivePatternDialog(session)
    dialog.members.setText("1, 2, 3")
    dialog.intensity.setValue(12.0)
    dialog._generate()

    assert len(session.model["load_cases"]) > before
    assert dialog.table.rowCount() >= 3, "满布/奇数跨/偶数跨至少三条"
    assert "跨" in dialog.table.item(0, 1).text()


def test_the_span_grouping_is_shown_because_it_has_to_be_checked(app, quiet):
    """跨的归类是几何推断出来的，必须摆给用户核对。

    混进了柱或另一方向的梁时，归出来的跨会很怪——而那不会报错。
    """
    session = beam_session()
    dialog = LivePatternDialog(session)
    dialog.members.setText("1, 2, 3")
    dialog._generate()
    shown = "".join(quiet["info"])
    assert "归组结果" in shown and "要核对" in shown


def test_the_live_load_is_applied_downwards(app, quiet):
    """界面里填的是正数，加到模型上必须是向下（全局 −Z）。"""
    session = beam_session()
    dialog = LivePatternDialog(session)
    dialog.members.setText("1, 2, 3")
    dialog.intensity.setValue(12.0)
    dialog._generate()
    added = [c for c in session.model["load_cases"]
             if str(c["name"]).startswith("LL-")]
    assert added
    loads = added[0]["member_loads"]
    assert loads, "生成的工况里得真有荷载"
    assert loads[0]["w"][2] == pytest.approx(-12e3)


# --- 面荷载 ---------------------------------------------------------------

def test_area_load_reaches_the_model(app, quiet):
    session = beam_session()
    dialog = AreaLoadDialog(session)
    dialog.members.setText("1, 2, 3")
    dialog.q.setValue(4.0)
    dialog.width.setValue(3.0)
    dialog.case.setCurrentText("DL")
    dialog._apply()

    assert not quiet["warn"], quiet["warn"]
    assert dialog.table.rowCount() == 3
    assert dialog.table.item(0, 2).text(), "每根梁的形状说明不能是空的"


@pytest.mark.parametrize("index", [0, 1, 2])
def test_every_load_path_the_dialog_offers_is_accepted_by_the_kernel(
        app, quiet, index):
    """下拉框里的每一项都必须是求解侧真认的值。

    这条不是形式主义：一开始我把 two_way 写进下拉框，而内核只认
    two_way_short / two_way_long——点下去直接报错，而下拉框看着很正常。
    """
    session = beam_session()
    dialog = AreaLoadDialog(session)
    dialog.members.setText("1, 2, 3")
    dialog.path.setCurrentIndex(index)
    dialog._apply()
    assert not quiet["warn"], (
        f"第 {index} 项传力方式内核不认：{quiet['warn']}")


def test_each_load_path_produces_its_own_distribution(app, quiet):
    """三种传力方式导出来的必须是不同的东西，否则这个选项没有意义。

    判据是形状说明与段数：单向板是一段均布，双向板短边梁是三角形、
    长边梁是梯形。三者相同就说明下拉框选了等于没选。
    """
    shapes = {}
    for index in range(3):
        session = beam_session()
        dialog = AreaLoadDialog(session)
        dialog.members.setText("1")
        dialog.path.setCurrentIndex(index)
        dialog._apply()
        shapes[index] = (dialog.table.item(0, 2).text(),
                         dialog.table.item(0, 1).text())

    assert "三角形" in shapes[1][0], f"双向板短边梁应是三角形：{shapes[1][0]}"
    assert "梯形" in shapes[2][0], f"双向板长边梁应是梯形：{shapes[2][0]}"
    assert len({s for s, _ in shapes.values()}) == 3, (
        f"三种传力方式给出了同样的分布：{shapes}")


# --- 边界条件管理器 -------------------------------------------------------

def test_the_bc_manager_lists_every_boundary_condition(app):
    session = beam_session()
    dialog = BCManagerDialog(session)
    assert dialog.table.rowCount() >= 2
    names = [dialog.table.item(r, 0).text()
             for r in range(dialog.table.rowCount())]
    assert all(n and n != "（未命名）" for n in names), (
        f"每条边界条件都该有名字，才删得掉：{names}")


def test_the_mask_is_shown_as_symbols_not_zeros_and_ones(app):
    """六个 0/1 排一行最容易看错，而看错了不会报错。"""
    session = beam_session()
    dialog = BCManagerDialog(session)
    mask = dialog.table.item(0, 3).text()
    assert set(mask.split()) <= {"●", "○"}
    assert len(mask.split()) == 6


def test_deleting_one_bc_does_not_take_the_others_with_it(app, quiet):
    """**默认名以前是常量 BC-1**，于是删「一条」会把同名的全删掉。

    删光了会被校验挡住，但只要还剩别的支座，它就悄悄删多了——而"少了
    几个约束"从结果里看不出来，只表现为位移偏大。
    """
    session = beam_session()
    dialog = BCManagerDialog(session)
    before = dialog.table.rowCount()
    dialog.table.selectRow(1)
    dialog._remove()
    assert not quiet["warn"], quiet["warn"]
    assert dialog.table.rowCount() == before - 1, "只该少掉选中的那一条"


def test_boundary_condition_names_do_not_collide(app):
    session = beam_session()
    names = [bc["name"] for bc
             in session.list_boundary_conditions().payload["boundary_conditions"]]
    assert len(names) == len(set(names)), f"BC 名字撞了：{names}"


# --- 通用闸门：提示文字不许被切掉 -----------------------------------------

#: 所有用 HintLabel 的对话框都进这张表。**新写一个对话框就在这里加一行**，
#: 否则下面那道闸门管不到它。
ALL_DIALOGS = [
    (CombinationDialog, (640, 580)),
    (LivePatternDialog, (600, 520)),
    (AreaLoadDialog, (600, 500)),
    (BCManagerDialog, (680, 440)),
]


def dialog_session():
    s = beam_session()
    s.add_step("s1", loads={"DL": "RAMP"})
    return s


@pytest.mark.parametrize(("cls", "size"), ALL_DIALOGS,
                         ids=[c.__name__ for c, _ in ALL_DIALOGS])
def test_no_hint_text_is_clipped(app, cls, size):
    """自动换行的 QLabel 默认会被压扁，折行之后最后一两行被切掉。

    它的 sizeHint 是按"排成一行"算的，布局照那个高度分配空间。被切掉的
    往往正是那句"选错不会报错"——而提示文字正是这个项目交付能力边界的
    主要位置，切掉了等于没写。实测规范组合少 9px、面荷载少 26px。

    判据是 heightForWidth，不是看截图：上一轮我把自己放大图像造成的失真
    当成了裁切。
    """
    from desktop.dialog_styles import HintLabel

    dialog = cls(dialog_session())
    dialog.resize(*size)
    dialog.show()
    app.processEvents()
    clipped = [(label.height(), label.heightForWidth(label.width()),
                label.text()[:40])
               for label in dialog.findChildren(HintLabel)
               if label.width() > 0
               and label.height() < label.heightForWidth(label.width())]
    assert not clipped, f"{cls.__name__} 的提示文字被切掉了：{clipped}"


@pytest.mark.parametrize(("cls", "size"), ALL_DIALOGS,
                         ids=[c.__name__ for c, _ in ALL_DIALOGS])
def test_no_dialog_leaks_markdown(app, cls, size):
    """`**重点**` 要渲染成加粗，不能把星号原样摆给用户。"""
    from PySide6.QtWidgets import QLabel

    dialog = cls(dialog_session())
    dialog.resize(*size)
    leaked = [label.text()[:50] for label in dialog.findChildren(QLabel)
              if "**" in label.text()]
    assert not leaked, f"{cls.__name__} 把 Markdown 星号漏给用户了：{leaked}"
