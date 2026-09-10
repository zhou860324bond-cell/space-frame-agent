"""桌面端 Qt 层的验证。

Qt 层是**薄的**——搭骨架、转发动作，不算也不画。所以这里验的是行为不是像素：
树填对没有、状态栏说的和模型一致没有、求解失败时有没有拦住、
切换模式会不会炸。

无头环境（QT_QPA_PLATFORM=offscreen）里没有 OpenGL，嵌入式 VTK 视口
渲染不出东西——**这是环境限制，不是代码问题**。所以这些测试一律不碰像素，
只碰逻辑。渲染那部分由 `test_desktop_scene.py` 在纯网格层面验，
那一层不需要 GL。
"""

import os

import pytest

pytest.importorskip("PySide6", reason="未安装 PySide6，跳过桌面端测试")
pytest.importorskip("pyvistaqt", reason="未安装 pyvistaqt，跳过桌面端测试")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyvista as pv                                        # noqa: E402

pv.OFF_SCREEN = True

from conftest import opengl_available                       # noqa: E402
pytestmark = pytest.mark.skipif(                             # noqa: E402
    not opengl_available(),
    reason="无可用 OpenGL，桌面端 VTK 视口无法初始化，跳过")

from PySide6.QtWidgets import QApplication                  # noqa: E402

from agent import Session                                   # noqa: E402
from desktop.main_window import MODES, MainWindow           # noqa: E402
from desktop.main_window import _parse_combo_expression     # noqa: E402

MATERIALS = [{"name": "Q355", "E": 2.06e11, "nu": 0.3, "density": 7850.0}]
SECTIONS = [{"name": "COLUMN", "A": 0.0147, "Iy": 4.2e-5, "Iz": 1.18e-3, "J": 9e-7},
            {"name": "RAFTER", "A": 0.0186, "Iy": 5.2e-5, "Iz": 2.47e-3, "J": 1.4e-6}]


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def built() -> Session:
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    g = s.generate_portal_frame(spans=[24.0], eave_height=7.5, ridge_rise=1.2,
                                column_section="COLUMN", rafter_section="RAFTER",
                                material="Q355", base="pinned")
    s.set_load_cases(cases=[{"name": "D", "member_loads":
                             [{"member": m, "w": [0, 0, -8e3]}
                              for m in g.payload["rafter_member_ids"]]}])
    return s


def solved(w: MainWindow) -> MainWindow:
    """点求解并等它跑完。

    `MainWindow.solve()` 现在是**异步**的——求解挪到了后台线程，
    否则网架、扫参、模态这类秒级的活儿会把窗口卡成"未响应"。
    所以测试里点完必须等，直接断言 `w.result` 只会拿到上一轮的状态。

    这个 helper 存在本身就是提示：**任何 `solve()` 之后立刻读结果的代码都是错的。**
    """
    w.solve()
    assert w.runner.wait(30000), "求解没在时限内跑完"
    return w


# ------------------------------------------------- 空模型

def test_an_empty_window_does_not_crash(qt_app):
    """开着空模型也得能站住——用户第一眼看到的就是这个状态。"""
    w = MainWindow()
    assert w.tree.topLevelItemCount() >= 0
    assert "空" in w.lbl_model.text()
    assert "未求解" in w.lbl_solve.text()
    assert not w.act_solve.isEnabled(), "没有模型时求解按钮该是灰的"


def test_solving_an_empty_model_is_refused(qt_app, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    shown = []
    monkeypatch.setattr(QMessageBox, "information",
                        lambda *a, **k: shown.append(a))
    w = MainWindow()
    solved(w)
    assert shown, "该弹一句话而不是默默什么都不做"
    assert w.result is None


# ------------------------------------------------- 有模型

def test_the_tree_lists_every_part(qt_app):
    w = MainWindow(built())
    assert w.empty_state.isHidden()
    labels = [w.tree.topLevelItem(k).text(0)
              for k in range(w.tree.topLevelItemCount())]
    for want in ("材料", "截面", "几何", "约束", "荷载工况", "建模过程"):
        assert any(want in lab for lab in labels), (want, labels)


def test_the_status_bar_matches_the_model(qt_app):
    s = built()
    w = MainWindow(s)
    assert f"{len(s.model['nodes'])} 节点" in w.lbl_model.text()
    assert f"{len(s.model['members'])} 杆件" in w.lbl_model.text()
    assert s.model.get("units", "N-m-Pa") in w.lbl_model.text()


def test_mm_model_selection_is_not_mislabelled_as_metres(qt_app):
    s = built()
    assert s.set_units("N-mm-MPa").ok
    w = MainWindow(s)
    node = s.model["nodes"][0]["id"]
    w._on_picked("node", node)
    assert "mm" in w.lbl_pick.text()


def test_the_status_bar_shows_the_last_build_step(qt_app):
    """建模过程那条信息切到哪个模式都在——不用回去翻。"""
    w = MainWindow(built())
    assert "设定荷载工况" in w.lbl_last.text()


def test_combo_expression_becomes_solver_factors(qt_app):
    factors = _parse_combo_expression("1.3×D + 1.5*L - 0.9×W", {"D", "L", "W"})
    assert factors == {"D": 1.3, "L": 1.5, "W": -0.9}


def test_combo_expression_rejects_an_unknown_case(qt_app):
    with pytest.raises(ValueError, match="不存在"):
        _parse_combo_expression("1.3×D + 1.5×L", {"D"})


def test_solving_switches_to_the_deformed_view(qt_app):
    """算完自动切到变形图。算完还停在模型视图，用户会以为没反应。"""
    w = MainWindow(built())
    solved(w)
    assert w.result is not None and w.result.ok
    assert w.mode == "变形"
    assert w.case == "D"
    assert "1 个工况" in w.lbl_solve.text()


def test_the_results_branch_appears_only_after_solving(qt_app):
    w = MainWindow(built())
    before = [w.tree.topLevelItem(k).text(0)
              for k in range(w.tree.topLevelItemCount())]
    assert not any("结果" == t for t in before)
    solved(w)
    after = [w.tree.topLevelItem(k).text(0)
             for k in range(w.tree.topLevelItemCount())]
    assert any("结果" == t for t in after)


def test_a_branch_that_appears_after_solving_starts_expanded(qt_app):
    """求解之后才出现的"结果"分支，必须是展开的。

    原来的做法是记住"上次展开了哪些"，于是新分支因为"上次不在展开集合里"
    而默认收起——**而它恰恰是用户此刻最想看的东西**。现在记的是"收起过哪些"。
    """
    w = solved(MainWindow(built()))
    node = next(w.tree.topLevelItem(k)
                for k in range(w.tree.topLevelItemCount())
                if w.tree.topLevelItem(k).text(0) == "结果")
    assert node.childCount(), "结果分支下该列出工况"
    assert node.isExpanded(), "新出现的分支不该默认收起"


def test_a_branch_the_user_collapsed_stays_collapsed(qt_app):
    """反过来也要成立：用户亲手收起的，刷新之后别自作主张展开。"""
    w = MainWindow(built())
    node = next(w.tree.topLevelItem(k)
                for k in range(w.tree.topLevelItemCount())
                if w.tree.topLevelItem(k).text(0).startswith("材料"))
    node.setExpanded(False)
    w.refresh()
    again = next(w.tree.topLevelItem(k)
                 for k in range(w.tree.topLevelItemCount())
                 if w.tree.topLevelItem(k).text(0).startswith("材料"))
    assert not again.isExpanded()


def test_tree_items_keep_the_full_text_in_a_tooltip(qt_app):
    """面板窄的时候文字会被省略号吃掉。**显示可以省略，数据不可以。**"""
    w = MainWindow(built())
    secs = next(w.tree.topLevelItem(k)
                for k in range(w.tree.topLevelItemCount())
                if w.tree.topLevelItem(k).text(0).startswith("截面"))
    child = secs.child(0)
    tip = child.toolTip(0)
    assert "Iy=" in tip and "J=" in tip, f"截面提示里没有完整特性：{tip!r}"


def test_the_check_tools_are_reachable_from_the_interface(qt_app):
    """三项校核不是附加功能，是结构程序该有的东西——界面上必须点得到。"""
    from desktop import commands

    w = MainWindow(built())
    for name in ("strength", "symmetry", "bandwidth"):
        assert name in w.actions_by_name, f"功能区里没有 {name}"
    handlers = {c.name: c.handler for c in commands.COMMANDS}
    for name in ("strength", "symmetry", "bandwidth"):
        assert callable(getattr(w, handlers[name], None))


def test_clearing_results_keeps_the_model(qt_app):
    w = solved(MainWindow(built()))
    before = w.session.model.copy()
    w.clear_results()
    assert w.result is None and w.session.solution is None
    assert w.session.model == before
    assert w.mode == "模型"


@pytest.mark.parametrize("mode", MODES)
def test_every_display_mode_survives(qt_app, mode):
    """四种模式都得能切过去而不炸。无头环境画不出东西，但代码路径要走通。"""
    w = MainWindow(built())
    solved(w)
    w.set_mode(mode)
    assert w.mode == mode
    assert w.mode_actions[mode].isChecked()


def test_the_mode_buttons_are_mutually_exclusive(qt_app):
    w = MainWindow(built())
    solved(w)
    w.set_mode("云图")
    checked = [n for n, a in w.mode_actions.items() if a.isChecked()]
    assert checked == ["云图"]


def test_analysis_mesh_preview_is_read_only_and_exposes_mapping(qt_app):
    s = built()
    member = s.model["members"][0]["id"]
    assert s.set_member_span_load(
        member, "point", [0.0, 0.0, -1000.0], a=2.0,
        case_name="D", name="Midspan-Point").ok
    w = solved(MainWindow(s))
    before_model = __import__("copy").deepcopy(s.model)
    before_state = (s.frame, s.solution, s.compilation, s.result_db)
    before_history = len(s.history.steps)

    w.show_analysis_mesh()

    assert w.mode == "分析网格"
    assert w.mode_actions["分析网格"].isChecked()
    assert w.results_dock.isVisibleTo(w)
    assert "物理杆件" in w.results.caption.text()
    assert w.results.table.rowCount() > len(s.model["members"])
    assert s.model == before_model
    assert (s.frame, s.solution, s.compilation, s.result_db) == before_state
    assert len(s.history.steps) == before_history


def test_modal_without_density_says_so_instead_of_crashing(qt_app, monkeypatch):
    """没有密度就没有质量矩阵。要说清楚，不能崩，也不能画个空视口了事。"""
    from PySide6.QtWidgets import QMessageBox
    shown = []
    monkeypatch.setattr(QMessageBox, "information",
                        lambda *a, **k: shown.append(a))
    s = Session()
    s.define_materials_and_sections(
        [{"name": "Q355", "E": 2.06e11, "nu": 0.3}], SECTIONS)   # 无密度
    g = s.generate_portal_frame(spans=[24.0], eave_height=7.5, ridge_rise=1.2,
                                column_section="COLUMN", rafter_section="RAFTER",
                                material="Q355", base="pinned")
    s.set_load_cases(cases=[{"name": "D", "member_loads":
                             [{"member": m, "w": [0, 0, -8e3]}
                              for m in g.payload["rafter_member_ids"]]}])
    w = MainWindow(s)
    solved(w)
    w.set_mode("模态")
    assert shown, "该说明为什么算不了"
    assert w.mode == "模型", "算不了就退回模型视图，不要停在一个空的模态视图上"


def test_a_singular_model_is_reported_not_crashed(qt_app):
    """算前校验拦住模型时，错误要摆进**结果面板**并且可点定位。

    以前是弹一个消息框写着"节点 3 缺少约束"——可用户在三维视图里
    根本找不到 3 号在哪。摆进表里、点一行就在视口高亮，这条信息才有用。
    """
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    s.add_nodes([[0, 0, 0], [6, 0, 0]])
    s.add_members([[1, 2]], section="RAFTER", material="Q355")
    # 一个约束都没有；这是允许存在、但不能求解的几何编辑态。
    w = solved(MainWindow(s))
    assert not w.result.ok
    assert w.results_dock.isVisibleTo(w), "错误清单该显示出来"
    assert "校验未通过" in w.results.caption.text()
    assert w.results.table.rowCount() > 0, "每条错误该是一行"


def test_new_model_clears_everything(qt_app):
    w = MainWindow(built())
    solved(w)
    w.chat.conversation = object()
    w.new_model()
    assert w.result is None and w.case is None
    assert "空" in w.lbl_model.text()
    assert w.chat.session is w.session
    assert w.chat.conversation is None
    assert w.lbl_last.text() == "最近：尚未建模"
    assert w.properties.session is w.session
    assert w.bc.session is w.session
    assert w.sketch.session is w.session
    assert w.section_opt.session is w.session


def test_sketch_uses_its_lowest_nodes_as_supports_and_replaces_every_reference(qt_app):
    """画布中心不是地面；草图平移后仍应能获得柱脚约束并继续人工建模。"""
    w = MainWindow(built())
    w._generate_from_sketch({
        "nodes": [(0.0, 4.0), (6.0, 4.0), (0.0, 8.0), (6.0, 8.0)],
        "lines": [(0, 2), (2, 3), (3, 1), (1, 0)],
        "plane": 0, "bays": 2, "bay_len": 5.0, "base": "pinned",
    })
    frame = w.session.preview_frame()
    supported = {item["node"] for item in w.session.model["supports"]}
    lowest = {nid for nid in frame.order()
              if abs(frame.nodes[nid].xyz[2] - 4.0) < 1e-9}
    assert supported == lowest
    assert len(w.session.model["members"]) == 12
    assert [s["name"] for s in w.session.model["sections"]] == ["COLUMN", "RAFTER"]
    assert w.chat.session is w.session
    assert w.properties.session is w.session


def test_picking_a_case_in_the_tree_switches_to_it(qt_app):
    s = built()
    s.set_load_cases(
        cases=[{"name": "D", "member_loads": [{"member": 1, "w": [0, 0, -8e3]}]},
               {"name": "L", "member_loads": [{"member": 1, "w": [0, 0, -5e3]}]}])
    w = MainWindow(s)
    solved(w)
    w._on_tree_action("loads", "L")
    assert w.case == "L"


def test_tree_lists_named_load_objects_and_opens_their_target(qt_app):
    s = built()
    member = s.model["load_cases"][0]["member_loads"][0]["member"]
    s.model["load_cases"][0]["member_loads"][0]["name"] = "Roof-UDL"
    w = MainWindow(s)

    def descendants(item):
        values = [item.text(0)]
        for index in range(item.childCount()):
            values.extend(descendants(item.child(index)))
        return values

    labels = []
    for index in range(w.tree.topLevelItemCount()):
        labels.extend(descendants(w.tree.topLevelItem(index)))
    assert any("Roof-UDL" in label and f"杆件 {member}" in label for label in labels)

    w._on_tree_action("load_object", {"case": "D", "kind": "member", "id": member})
    assert w.case == "D"
    assert w._selected_kind == "member" and w._selected_id == member
    assert w.bc._current_member == member


def test_tree_folders_dispatch_to_real_editors(qt_app, monkeypatch):
    w = MainWindow(built())
    called = []
    monkeypatch.setattr(w, "create_material", lambda: called.append("material"))
    monkeypatch.setattr(w, "create_section", lambda: called.append("section"))
    monkeypatch.setattr(w, "open_tables", lambda: called.append("tables"))
    w._on_tree_action("materials", None)
    w._on_tree_action("sections", None)
    w._on_tree_action("geometry", None)
    assert called == ["material", "section", "tables"]


def test_desktop_can_export_a_verified_agent_learning_trace(
        qt_app, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QMessageBox
    shown = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(QMessageBox, "information",
                        lambda *args, **kwargs: shown.append(args))
    w = MainWindow(built())
    w.export_learning_trace()
    traces = list((tmp_path / "learning_traces").glob("*.json"))
    assert len(traces) == 1
    assert "正样本" in shown[0][2]


def test_property_assignment_updates_material_and_section_together(
        qt_app, monkeypatch):
    from PySide6.QtWidgets import QDialog
    from desktop import assignment_dialog

    class FakeAssignmentDialog:
        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            return QDialog.DialogCode.Accepted

        def get_assignment(self):
            return "RAFTER", "Q235"

    monkeypatch.setattr(assignment_dialog, "AssignmentDialog", FakeAssignmentDialog)
    s = built()
    s.model["materials"].append(
        {"name": "Q235", "E": 2.0e11, "nu": 0.3, "density": 7850.0})
    member = s.model["members"][0]["id"]
    w = MainWindow(s)
    w._selected_kind, w._selected_id = "member", member
    w.assign_section()
    updated = next(item for item in s.model["members"] if item["id"] == member)
    assert updated["section"] == "RAFTER"
    assert updated["material"] == "Q235"


def test_the_camera_is_not_reset_on_every_redraw(qt_app):
    """每次重画都把相机弹回去的话，改一个参数视角就没了，用起来很恼火。"""
    w = MainWindow(built())
    solved(w)
    assert not w.viewport._first_render, "首帧之后就不该再自动摆相机了"


def test_selection_dependent_buttons_are_disabled_until_something_is_picked(qt_app):
    """必须先选中的按钮，没选中时要置灰。

    原来它们永远可点，点了只在状态栏闪一句五秒后消失的提示——用户看到的是
    "点了没反应"，这正是"很多功能都是摆设"那类抱怨的来源。前置条件应该看得见。
    """
    w = MainWindow()
    load = w.actions_by_name["create_load"]
    bc = w.actions_by_name["create_bc"]
    assert not load.isEnabled() and not bc.isEnabled()
    assert "请先" in load.toolTip()

    w._on_picked("node", 1)
    assert load.isEnabled() and bc.isEnabled(), "选中节点后两个都可用"

    w._on_picked("member", 1)
    assert load.isEnabled(), "杆件可以施加载荷"
    assert not bc.isEnabled(), "边界条件只能加在节点上"
    assert "节点" in bc.toolTip()


def test_property_edit_keeps_the_object_selected_and_highlighted(qt_app):
    s = built()
    w = MainWindow(s)
    member = s.model["members"][0]["id"]
    w._on_picked("member", member)
    sections = [item["name"] for item in s.model["sections"]]
    replacement = next(name for name in sections
                       if name != s.model["members"][0]["section"])

    w.properties._widgets["section"].setCurrentText(replacement)

    assert (w._selected_kind, w._selected_id) == ("member", member)
    assert w.viewport.selection == ("member", member)
    assert w.properties.kind == "member" and w.properties.ident == member


def test_viewport_context_menu_tracks_the_selected_object(qt_app):
    w = MainWindow(built())
    empty = [action.text() for action in w._viewport_context_menu().actions()]
    assert "选择节点" in empty and "适应窗口" in empty

    w._on_picked("node", 1)
    node = [action.text() for action in w._viewport_context_menu().actions()]
    assert "节点 1" in node
    assert "创建边界条件" in node and "创建载荷" in node
    assert "属性指派" not in node

    member = w.session.model["members"][0]["id"]
    w._on_picked("member", member)
    member_actions = [action.text() for action in
                      w._viewport_context_menu().actions()]
    assert f"杆件 {member}" in member_actions
    assert {"属性指派", "创建铰接", "创建载荷", "删除选中"}.issubset(
        member_actions)


def test_model_diagnosis_carries_object_references_for_viewport_highlight(
        qt_app, monkeypatch):
    s = built()
    w = MainWindow(s)
    s.model["nodes"].append({"id": 999, "x": 2.0, "y": 1.0, "z": 9.0})
    captured = {}
    monkeypatch.setattr(
        w, "_show_diagnose_result",
        lambda issues, suggestions, locations: captured.update(
            issues=issues, suggestions=suggestions, locations=locations))

    w.diagnose_model()

    row = next(i for i, text in enumerate(captured["issues"])
               if "孤立节点" in text)
    assert captured["locations"][row] == [("node", 999)]
    w.locate_problem(captured["locations"][row])
    assert w.viewport.problem_refs == [("node", 999)]
