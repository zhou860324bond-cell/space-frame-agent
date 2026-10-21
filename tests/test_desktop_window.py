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


def test_export_screenshot_keeps_the_empty_state_card(qt_app, monkeypatch, tmp_path):
    """防止 VTK 截图回贴时盖住空模型引导卡片，造成手册图缺失入口。"""
    import numpy as np
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QImage

    w = MainWindow()
    w.show()
    qt_app.processEvents()
    monkeypatch.setattr(w.viewport.plotter, "screenshot",
                        lambda **_kw: np.full((40, 40, 3), (255, 0, 0),
                                                dtype=np.uint8))
    path = tmp_path / "empty-state.png"
    w.export_screenshot(str(path))
    picture = QImage(str(path))
    point = w.empty_state.mapTo(w, w.empty_state.rect().center())
    ratio = picture.width() / w.width()
    pixel = picture.pixelColor(QPoint(int(point.x() * ratio),
                                      int(point.y() * ratio)))
    assert pixel.red() < 250 or pixel.green() > 10 or pixel.blue() > 10
    w.close()


def test_export_screenshot_positions_tool_drawers_in_window_coordinates(qt_app, monkeypatch, tmp_path):
    """主窗离开原点时，独立 Tool 抽屉截图不能再次叠加屏幕坐标而偏移。"""
    import numpy as np
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QImage

    w = MainWindow(built())
    w.move(150, 90)
    w.show()
    w.timeline_dock.dock_panel()
    w.timeline_dock.show()
    qt_app.processEvents()
    monkeypatch.setattr(w.viewport.plotter, "screenshot",
                        lambda **_kw: np.full((40, 40, 3), (255, 0, 0), dtype=np.uint8))
    path = tmp_path / "drawer.png"
    w.export_screenshot(str(path))
    picture = QImage(str(path))
    point = w.mapFromGlobal(w.left_drawer.mapToGlobal(QPoint(20, 100)))
    ratio = picture.width() / w.width()
    pixel = picture.pixelColor(int(point.x() * ratio), int(point.y() * ratio))
    assert pixel.green() > 10 or pixel.blue() > 10
    w.close()


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
    # 与 Abaqus 模型树同序：对象分支（截面指派、边界条件、荷载）放在几何与分析步之后
    for want in ("材料", "截面", "截面指派", "集合", "几何", "分析步", "边界条件", "荷载",
                 "建模过程"):
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


def test_tree_lists_named_load_objects_and_opens_their_target(qt_app, monkeypatch):
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

    opened = []
    monkeypatch.setattr(w, "open_load_manager", lambda: opened.append("loads"))
    w._on_tree_action("load_object", {"case": "D", "kind": "member", "id": member})
    assert w.case == "D"
    assert w._selected_kind == "member" and w._selected_id == member
    assert opened == ["loads"], "荷载统一在荷载管理器里编辑（旧边界表单已删除）"


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


class _Accept:
    """替身对话框：exec 直接确定，values/vector 返回预先给定的值。"""

    def __init__(self, *args, values=None, vector=None, **kwargs):
        self._values, self._vector = values, vector
        self.name = type("N", (), {"setEnabled": lambda *_: None})()

    def exec(self):
        from PySide6.QtWidgets import QDialog
        return QDialog.DialogCode.Accepted

    def values(self):
        return self._values

    def vector(self):
        return self._vector


@pytest.fixture(autouse=True)
def _no_modal_boxes(monkeypatch):
    """对象命令失败时会弹 QMessageBox；无头环境里没人点，测试会永远挂住。"""
    from desktop import window_objects
    monkeypatch.setattr(window_objects.QMessageBox, "warning",
                        staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(window_objects.QMessageBox, "information",
                        staticmethod(lambda *a, **k: None))


def _stub(monkeypatch, name, **kwargs):
    from desktop import window_objects
    monkeypatch.setattr(window_objects, name,
                        lambda *a, **k: _Accept(*a, **{**k, **kwargs}))


def test_assign_section_picks_a_region_first_then_creates_one_object(qt_app, monkeypatch):
    """Abaqus 的截面指派：先在视口选区域（可多选），再选截面；结果是一个可管理的对象。

    原来要先单选一根杆件，多根只能重复整套流程，指派也不是对象。
    """
    s = built()
    s.model["materials"].append({"name": "Q235", "E": 2.0e11, "nu": 0.3, "density": 7850.0})
    w = MainWindow(s)
    _stub(monkeypatch, "AssignSectionDialog", values=("SA-椽", "RAFTER", "Q235"))
    w.assign_section()
    assert w.region_pick.active and w.viewport.region_pick == "member"
    assert "截面指派" in w.lbl_prompt.text() and "拖框" in w.lbl_prompt.text()
    targets = [m["id"] for m in s.model["members"][:2]]
    w.viewport.region_selection.apply("member", targets)
    w.region_pick.finish()
    assert not w.region_pick.active
    for member in s.model["members"]:
        if member["id"] in targets:
            assert (member["section"], member["material"]) == ("RAFTER", "Q235")
            assert member["assignment"] == "SA-椽"
    assert [f["name"] for f in s.model["features"]["section_assignments"]] == ["SA-椽"]


def test_the_camera_is_not_reset_on_every_redraw(qt_app):
    """每次重画都把相机弹回去的话，改一个参数视角就没了，用起来很恼火。"""
    w = MainWindow(built())
    solved(w)
    assert not w.viewport._first_render, "首帧之后就不该再自动摆相机了"


def test_object_commands_are_clickable_without_a_prior_selection(qt_app):
    """命令在前、区域在后：没选中任何东西也能点，提示里说明怎么选。"""
    w = MainWindow()
    for name in ("create_load", "create_bc", "assign_section"):
        action = w.actions_by_name[name]
        assert action.isEnabled()
        assert "集合" in action.toolTip() and "Shift" in action.toolTip()


def test_create_bc_follows_the_abaqus_sequence_and_takes_many_nodes(qt_app, monkeypatch):
    """创建 → 选区域（多选）→ 编辑：一次给多个柱脚加约束，结果是一个边界条件对象。"""
    s = built()
    w = MainWindow(s)
    _stub(monkeypatch, "CreateObjectDialog", values=("BC-柱脚", "Initial", "PINNED"))
    _stub(monkeypatch, "EditBCDialog", values=([1, 1, 1, 0, 0, 0], None))
    warnings = []
    from desktop import window_objects
    monkeypatch.setattr(window_objects.QMessageBox, "warning",
                        staticmethod(lambda *a, **k: warnings.append(a[2])))
    w.create_bc()
    assert w.region_pick.active and w.viewport.region_pick == "node"
    # 门架模型每个节点都有面外约束的直接支座；清掉后才能用对象重新约束
    s.model["supports"] = []
    feet = [1, 2]
    w.viewport.region_selection.apply("node", feet)
    w.region_pick.finish()
    assert not warnings, warnings
    assert [f["name"] for f in s.model["features"]["bcs"]] == ["BC-柱脚"]
    owned = [x for x in s.model["supports"] if x.get("owners") == ["BC-柱脚"]]
    assert sorted(x["node"] for x in owned) == sorted(feet)


def test_the_selection_before_the_command_becomes_the_default_region(qt_app, monkeypatch):
    """先选后点的老习惯保留：命令开始时已选中的同类对象直接带进区域。"""
    s = built()
    w = MainWindow(s)
    member = s.model["members"][0]["id"]
    w._on_picked("member", member)
    w.assign_section()
    assert w.viewport.region_selection.ids == [member]


def test_using_a_set_makes_the_region_a_live_reference(qt_app, monkeypatch):
    s = built()
    w = MainWindow(s)
    s.define_set(name="全部杆件", member_ids=[m["id"] for m in s.model["members"]])
    s.set_load_cases([{"name": "DL"}])
    _stub(monkeypatch, "CreateObjectDialog", values=("屋面", "DL", "line_load"))
    _stub(monkeypatch, "EditLoadDialog", vector=[0.0, 0.0, -1e3])
    from desktop import region_prompt
    monkeypatch.setattr(region_prompt.QInputDialog, "getItem",
                        staticmethod(lambda *a, **k: ("全部杆件", True)))
    w.create_load()
    w.region_pick.choose_set()
    feature = s.model["features"]["loads"][0]
    assert feature["region"] == {"set": "全部杆件"}


def test_escape_and_a_new_command_cancel_a_region_pick(qt_app, monkeypatch):
    """Esc 取消选择；开始另一个对象命令时上一个选择作废，不会把区域交给错的命令。"""
    s = built()
    w = MainWindow(s)
    _stub(monkeypatch, "CreateObjectDialog", values=("BC-1", "Initial", "ENCASTRE"))
    w.create_bc()
    w.cancel_interaction()
    assert not w.region_pick.active and w.viewport.region_pick is None
    created = []
    w.create_bc()
    w.region_pick.start("member", "截面指派", created.append)
    assert w.viewport.region_pick == "member"
    w.viewport.region_selection.apply("member", [s.model["members"][0]["id"]])
    w.region_pick.finish()
    assert created and "members" in created[0]
    assert not (s.model.get("features") or {}).get("bcs"), "被顶掉的边界条件命令不能再落地"


def test_region_pick_rejects_finishing_with_nothing_selected(qt_app):
    w = MainWindow(built())
    delivered = []
    w.region_pick.start("node", "测试", delivered.append)
    w.region_pick.finish()
    assert not delivered and w.region_pick.active


def test_solving_fills_the_result_summary_without_popping_it_open(qt_app):
    """求解完先把结果画在模型上；总览表填好但不弹出，想看时点右侧「结果」。"""
    w = MainWindow(built())
    solved(w)
    assert w.result.ok
    assert w.results_dock.isHidden(), "求解完不该自动弹出一块抽屉挡住模型"
    assert "求解完成" in w.results.caption.text()
    assert w.results.table.rowCount() == len(w.result.payload["cases"])


def test_picking_while_viewing_results_does_not_pop_the_property_drawer(qt_app):
    """看结果时点选是在查数，再弹一个属性抽屉只会把模型又挤掉一块。"""
    s = built()
    w = MainWindow(s)
    solved(w)
    assert w.mode == "变形"
    w.tree_dock.hide()
    w._on_picked("member", s.model["members"][0]["id"])
    assert w.props_dock.isHidden()


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


def test_the_contour_is_drawn_from_banded_cell_colours(qt_app):
    """把 show_contour 真跑一遍，盯住送进渲染器的到底是什么。

    无头环境画不出像素，但**送进去的网格和参数是可以查的**——这一层出错
    （标量名写错、颜色挂在点上、级数没传下去）在有屏幕的机器上也只是
    "图看着怪"，没人查得出来是哪一步。
    """
    window = solved(MainWindow(built()))
    calls: list[dict] = []
    plotter = window.viewport.plotter
    plotter.add_mesh = lambda mesh=None, **kw: calls.append(
        {"mesh": mesh, **kw})
    import desktop.viewport as vp
    from desktop import scene

    was = vp.CAN_RENDER
    vp.CAN_RENDER = True
    try:
        window.viewport.show_contour(window.session.frame,
                                     window.session.solution, "D", "Mz",
                                     levels=8)
    finally:
        vp.CAN_RENDER = was

    contour = calls[0]
    assert contour["scalars"] == "Mz" + scene.BAND_SUFFIX
    assert contour["n_colors"] == 8
    mesh = contour["mesh"]
    assert contour["scalars"] in mesh.cell_data
    assert contour["scalars"] not in mesh.point_data
    # 默认打光——圆管得看得出是圆的。环境光要托得住底（背光面不能黑掉，
    # 否则同一个数值在两面差出一级颜色），但也不能高到把明暗差压没：
    # 那正是"打了光还是看不出是圆的"的原因。
    assert contour["lighting"] is True
    assert 0.3 <= contour["ambient"] <= 0.5
    assert contour["diffuse"] > contour["ambient"]


def test_the_contour_members_are_pixel_width_lit_lines_not_fat_tubes(qt_app):
    """云图杆件的粗细按屏幕像素定，不按模型尺寸定。"""
    # 按世界尺寸建管时只能靠加粗（CONTOUR_TUBE_RATIO 曾到模型显示的 4.5 倍），
    # 一切到云图杆件就胖一大圈。现在按屏幕像素画成带光照的线管：粗细与
    # 模型尺寸、镜头远近无关，这里守的是"别退回世界尺寸的粗管"。
    window = solved(MainWindow(built()))
    calls: list[dict] = []
    window.viewport.plotter.add_mesh = lambda mesh=None, **kw: calls.append(
        {"mesh": mesh, **kw})
    import desktop.viewport as vp

    was = vp.CAN_RENDER
    vp.CAN_RENDER = True
    try:
        for levels in (0, 8):
            calls.clear()
            window.viewport.show_contour(window.session.frame,
                                         window.session.solution, "D", "Mz",
                                         levels=levels)
            contour = calls[0]
            assert contour["render_lines_as_tubes"] is True
            assert contour["line_width"] == vp.CONTOUR_LINE_PX >= 5
            assert contour["mesh"].n_lines and not contour["mesh"].n_faces_strict
    finally:
        vp.CAN_RENDER = was


def test_the_continuous_contour_colours_points_and_skips_the_overflow_layer(qt_app):
    """连续模式（levels=0）：标量挂在点上、映射前插值，颜色沿杆平滑过渡。

    分级把合弯矩这类先降后升的量切成一圈圈色环，整张图像彩色条纹——
    这是"云图好丑、一根杆一个颜色"的来源。连续模式下也不再压一层橙色的
    量程外标记：那里本来就是色标顶端的颜色。
    """
    window = solved(MainWindow(built()))
    calls: list[dict] = []
    plotter = window.viewport.plotter
    plotter.add_mesh = lambda mesh=None, **kw: calls.append({"mesh": mesh, **kw})
    import desktop.viewport as vp

    was = vp.CAN_RENDER
    vp.CAN_RENDER = True
    try:
        window.viewport.show_contour(window.session.frame, window.session.solution,
                                     "D", "M", levels=0)
    finally:
        vp.CAN_RENDER = was
    contour = calls[0]
    assert contour["scalars"] == "M"
    assert "M" in contour["mesh"].point_data
    assert contour["interpolate_before_map"] is True
    assert contour["n_colors"] == 256
    assert not any(c.get("name") == "_contour_out_of_range" for c in calls)


def test_the_result_panel_defaults_to_continuous_colouring(qt_app):
    window = MainWindow(built())
    assert window.results.levels.currentData() == 0
    assert window.result_display_options["levels"] == 0


def test_the_force_diagram_mode_draws_diagrams_and_keeps_the_picked_component(
        qt_app):
    """「内力图」是一个显示模式；在内力图里换分量不会被踢回云图。"""
    window = solved(MainWindow(built()))
    drawn = []
    window.viewport.show_force_diagram = (
        lambda frame, solution, case, component: drawn.append(component))
    window.show_force_diagram()
    assert window.mode == "内力图"
    assert window.mode_actions["内力图"].isChecked()
    assert drawn[-1] == window.component
    window._set_component("Vz")
    assert window.mode == "内力图" and drawn[-1] == "Vz"


def _checkable() -> Session:
    """带许用应力与截面几何的模型——没有它们强度验算做不了。"""
    s = Session()
    s.define_materials_and_sections(
        [{"name": "Q355", "E": 2.06e11, "nu": 0.3, "density": 7850.0,
          "allow_tension": 305e6, "allow_compression": 305e6,
          "yield_stress": 355e6}],
        [{"name": "COLUMN", "A": 0.0147, "Iy": 4.2e-5, "Iz": 1.18e-3, "J": 9e-7,
          "cy": 0.15, "cz": 0.1},
         {"name": "RAFTER", "A": 0.0186, "Iy": 5.2e-5, "Iz": 2.47e-3, "J": 1.4e-6,
          "cy": 0.2, "cz": 0.1}])
    g = s.generate_portal_frame(spans=[24.0], eave_height=7.5, ridge_rise=1.2,
                                column_section="COLUMN", rafter_section="RAFTER",
                                material="Q355", base="pinned")
    s.set_load_cases(cases=[{"name": "D", "member_loads":
                             [{"member": m, "w": [0, 0, -8e3]}
                              for m in g.payload["rafter_member_ids"]]}])
    return s


def test_the_utilization_mode_runs_the_check_once_and_draws_it(qt_app):
    """应力比图后台跑一次验算；同一次求解再切过去直接用缓存，结果表也填好。"""
    w = solved(MainWindow(_checkable()))
    drawn = []
    w.viewport.show_utilization = lambda frame, mapping, payload: drawn.append(payload)
    w.show_utilization()
    assert w.runner.wait(30000)
    qt_app.processEvents()
    assert w.mode == "应力比" and drawn
    assert drawn[-1]["members"], "验算结果要带逐杆的利用率"
    assert "强度验算" in w.results.caption.text()
    runs = []
    w.session.check_strength = lambda **_kw: runs.append(1)
    w.set_mode("模型")
    w.show_utilization()                   # 同一次求解：不重算
    assert w.mode == "应力比" and not runs


def test_the_utilization_mode_says_what_is_missing_instead_of_drawing_nothing(
        qt_app, monkeypatch):
    """材料没给许用应力时说清楚去哪补，并退回模型显示，不留一个空视口。"""
    from PySide6.QtWidgets import QMessageBox

    told = []
    monkeypatch.setattr(QMessageBox, "information",
                        lambda _parent, title, text: told.append(text))
    w = solved(MainWindow(built()))        # built() 的材料没有许用应力
    w.show_utilization()
    assert w.runner.wait(30000)
    qt_app.processEvents()
    assert told and "allow" in told[-1] + str(told)
    assert w.mode != "应力比"


def test_an_automatic_utilization_refresh_never_pops_a_modal_dialog(
        qt_app, monkeypatch):
    """停在应力比上重新求解时，重画会自动补算；算不了只在状态栏说，不弹框。

    模态框从后台回调里弹出来会和事件循环嵌套——这条路径曾在全量测试里把
    进程直接带崩（Windows 0xc0000374 堆损坏）。
    """
    from PySide6.QtWidgets import QMessageBox

    popped = []
    monkeypatch.setattr(QMessageBox, "information",
                        lambda *args: popped.append(args))
    w = solved(MainWindow(built()))        # 没有许用应力，验算必然失败
    w.set_mode("应力比")                     # 不经按钮，直接切模式
    qt_app.processEvents()
    assert w.runner.wait(30000)
    qt_app.processEvents()
    assert not popped
    assert w.mode == "模型"
    assert "无法显示应力比" in w.statusBar().currentMessage()


def test_joint_solid_asks_for_the_level_count_and_shows_progress(qt_app, monkeypatch):
    """节点实体是分钟级作业：开算前说清快速/收敛判断的区别，算的时候一直
    显示走到哪一档了——不能只有一句"计算中"。"""
    from agent import ToolResult

    s = built()
    w = solved(MainWindow(s))
    seen = {}

    def fake(node_id, case=None, levels=1, progress=None, **_kw):
        seen["levels"] = levels
        progress("第 2/3 档：求解 12345 个自由度…")
        return ToolResult(False, {"error": "测试用：不真算"})

    monkeypatch.setattr(w.session, "analyze_joint_solid", fake)
    shown = []
    original = w.statusBar().showMessage
    monkeypatch.setattr(w.statusBar(), "showMessage",
                        lambda text, *a: (shown.append(text), original(text, *a)))
    monkeypatch.setattr(w, "_ask_joint_levels", lambda _node: 3)
    w._on_picked("node", 1)
    w.run_solid_joint()
    assert w.runner.wait(30000)
    qt_app.processEvents()
    assert seen["levels"] == 3
    assert any("第 2/3 档" in text for text in shown)


def test_cancelling_the_joint_solid_choice_runs_nothing(qt_app, monkeypatch):
    s = built()
    w = solved(MainWindow(s))
    ran = []
    monkeypatch.setattr(w.session, "analyze_joint_solid", lambda **kw: ran.append(kw))
    monkeypatch.setattr(w, "_ask_joint_levels", lambda _node: None)
    w._on_picked("node", 1)
    w.run_solid_joint()
    assert not w.runner.busy and not ran


def test_turning_off_contour_shading_gives_flat_colour(qt_app):
    """要精确对色标读数时可以关掉打光，这时必须是纯平涂。"""
    window = solved(MainWindow(built()))
    calls: list[dict] = []
    window.viewport.plotter.add_mesh = lambda mesh=None, **kw: calls.append(kw)
    import desktop.viewport as vp

    assert window.viewport.set_contour_shading(False) is True
    assert window.viewport.set_contour_shading(False) is False, "没变就不该说变了"
    was = vp.CAN_RENDER
    vp.CAN_RENDER = True
    try:
        window.viewport.show_contour(window.session.frame,
                                     window.session.solution, "D", "Mz")
    finally:
        vp.CAN_RENDER = was
    assert calls[0]["lighting"] is False
    assert "ambient" not in calls[0]


def test_the_contour_palette_defaults_to_smooth_turbo(qt_app):
    """云图默认用平滑彩虹（turbo），Abaqus 彩虹与蓝—灰—红仍可选。

    经典彩虹中段有一大片刺眼的亮绿亮黄，连续着色时沿杆渐变会在那一段
    突然"跳亮"；turbo 明度过渡均匀。换色系是**显示选项**，不该改动任何
    数值：这里只认色标本身变了没有。
    """
    from desktop import theme

    window = solved(MainWindow(built()))
    assert window.viewport.contour_palette == "turbo"
    assert theme.palette_cmap("turbo")(0.0) != theme.palette_cmap("rainbow")(0.0)
    rainbow = theme.palette_cmap("rainbow")
    diverging = theme.palette_cmap("diverging", "Mz")
    assert rainbow(0.0) != diverging(0.0)
    assert window.viewport.set_contour_palette("diverging") is True
    assert window.viewport.set_contour_palette("diverging") is False
    assert window.viewport.set_contour_palette("不存在的色系") is False
    # 合量定义上非负，发散色标的中点代表零，在它身上没有意义——自动退回顺序色标
    assert theme.palette_cmap("diverging", "M")(0.5) == theme.sequential_cmap()(0.5)


def test_the_stress_contour_is_refused_at_the_click_not_after_switching(
        qt_app, monkeypatch):
    """截面缺 cy/cz 时应力云图算不出来。**在点下去的那一刻就说清楚**——
    别让用户切过去之后对着一个空视口和一个没反应的菜单项。"""
    from PySide6.QtWidgets import QMessageBox

    from desktop import scene

    window = solved(MainWindow(built()))
    said: list[str] = []
    monkeypatch.setattr(QMessageBox, "information",
                        lambda *a, **k: said.append(str(a[2])))
    before = window.component
    window._set_component(scene.STRESS)
    assert said and "cy" in said[0], "要说清楚缺什么，而不是没反应"
    assert window.component == before, "画不出来就别切过去"


def test_the_colour_bar_is_bound_to_the_contour_layer_not_the_overflow_layer(
        qt_app):
    """色标颜色必须取自云图那一层的查找表，而不是最后加进来的那一层。

    原来用 VTK 色标，它绑的是**最后一个加进来的网格**的映射器：等把"量程外"
    那层纯色网格加完再加色标，色标画的就是那层的默认查找表——一条 0…1 的
    彩虹，和图上任何东西都对不上。现在色标由 Qt 画，颜色显式从云图那一层的
    映射器读，这里守住"读的是哪一层"。
    """
    window = solved(MainWindow(built()))
    plotter = window.viewport.plotter

    class Table:
        def __init__(self, colours):
            self.colours = colours

        def GetNumberOfTableValues(self):
            return len(self.colours)

        def GetTableValue(self, i):
            return (*self.colours[i], 1.0)

    class Actor:
        def __init__(self, colours):
            self.table = Table(colours)

        def GetMapper(self):
            return self

        def GetLookupTable(self):
            return self.table

    contour = [(0.1, 0.2, 0.9), (0.9, 0.3, 0.1)]
    plotter.add_mesh = lambda mesh=None, **kw: Actor(
        contour if "scalars" in kw else [(1.0, 0.0, 1.0)] * 4)
    plotter.add_point_labels = lambda *a, **k: None

    import desktop.viewport as vp
    was = vp.CAN_RENDER
    vp.CAN_RENDER = True
    try:
        window.viewport.show_contour(window.session.frame,
                                     window.session.solution, "D", "Mz", levels=12)
    finally:
        vp.CAN_RENDER = was

    scale = window.viewport.color_scale
    assert scale.active
    import numpy as np

    got = np.array([c.getRgbF()[:3] for c in scale.colors])
    assert got.shape == (2, 3) and np.allclose(got, contour, atol=1 / 255)
    assert scale.banded, "分级云图的色标也要分级，否则读不出每一级的区间"
    assert "Mz" in scale.title


def test_colour_scale_ticks_keep_both_ends_and_never_overlap(qt_app):
    """刻度太密时隔一个标一个，但**两端必标**：只标中间、不标最大值的色标
    读不出量程。"""
    from desktop.viewport_legend import ColorScale

    scale = ColorScale()
    scale.set_scale("σ (MPa)", [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0)] * 12, (-40.0, 40.0),
                    banded=True)
    scale.resize(scale.width(), 160)
    shown = []
    painter_text = []

    from PySide6.QtGui import QPainter
    original = QPainter.drawText

    def capture(self, *args):
        painter_text.append(args[-1])
        return original(self, *args)

    QPainter.drawText = capture
    try:
        scale.grab()
    finally:
        QPainter.drawText = original
    shown = [t for t in painter_text if t != "σ (MPa)"]
    assert shown[0] == "−40" and shown[-1] == "40"
    assert len(shown) < 25, "160 像素高放不下 25 个刻度，必须隔着标"


def test_peak_tag_states_length_unit():
    """视口峰值标签曾写「杆件 24, x = 3」，读不出是 3 m 还是 3 mm。"""
    from desktop.viewport import peak_tag

    assert peak_tag("M = ", 128.1, "kN·m", 24, 3.0, "m").endswith("杆件 24, x = 3 m")
    assert peak_tag("M = ", 128.1, "kN·m", 24, 3000.0, "mm").endswith("x = 3e+03 mm")


def test_result_marker_imports_pyvista_itself(qt_app):
    """放结果标记这条路径必须自己把 pyvista 导进来。

    `desktop/viewport.py` 里 pyvista 一律用时再导——它和 VTK 加起来启动开销
    不小，无头机器上还根本装不上。但 `set_result_marker` 原先漏了那一行，
    引用的是别的方法里的局部 `pv`。

    漏了这一行**测不出来**，这才是要补这条测试的原因：回归跑在
    `QT_QPA_PLATFORM=offscreen` 下，`CAN_RENDER` 是 False，函数第一句就
    return 了；只有真机上点"结果探针"或"定位极值"才会撞出 NameError。
    也就是说这个 bug 对用户可见、对回归不可见。所以这里把 CAN_RENDER
    显式扳成 True，逼它真的走到造球那一步。

    （原件是 ruff 的 F821 翻出来的，不是人看出来的。）
    """
    window = solved(MainWindow(built()))
    meshes: list[dict] = []
    plotter = window.viewport.plotter
    plotter.add_mesh = lambda mesh=None, **kw: meshes.append({"mesh": mesh, **kw})
    plotter.add_point_labels = lambda *a, **kw: None
    import desktop.viewport as vp

    was = vp.CAN_RENDER
    vp.CAN_RENDER = True
    try:
        window.viewport.set_result_marker((0.0, 0.0, 0.0), "PROBE M1")
    finally:
        vp.CAN_RENDER = was

    assert meshes, "标记没送进渲染器"
    assert meshes[0]["name"] == "_result_marker"
    # 真造出了一个球，而不是 None 混过去
    assert meshes[0]["mesh"].n_points > 0


def _tree_items(tree):
    out, stack = [], [tree.topLevelItem(k) for k in range(tree.topLevelItemCount())]
    while stack:
        item = stack.pop()
        out.append(item)
        stack.extend(item.child(k) for k in range(item.childCount()))
    return out


def test_model_tree_shows_objects_with_the_manager_context_menu(qt_app, monkeypatch):
    """对象进模型树：单击高亮区域，右键是管理器的同一组操作（Abaqus 模型树的用法）。"""
    from desktop import model_tree, window_objects
    s = built()
    s.define_set(name="椽", member_ids=[m["id"] for m in s.model["members"]][:2])
    assert s.create_section_assignment("SA-椽", "椽", "RAFTER", "Q355").ok
    w = MainWindow(s)
    feature = next(i for i in _tree_items(w.tree)
                   if i.data(0, model_tree.ACTION) == "feature")
    assert feature.data(0, model_tree.PAYLOAD) == ("section_assignments", "SA-椽", False)
    assert [a for a, _ in w.tree.context_actions(feature)] == [
        "edit", "region", "copy", "rename", "suppress", "delete"]
    branch = feature.parent()
    assert [a for a, _ in w.tree.context_actions(branch)] == ["create", "manager"]

    w.tree.feature_selected.emit("section_assignments", "SA-椽")
    assert w.viewport.region_selection.kind == "member"
    assert len(w.viewport.region_selection.ids) == 2

    monkeypatch.setattr(window_objects.QMessageBox, "question",
                        staticmethod(lambda *a, **k: window_objects.QMessageBox.StandardButton.Yes))
    w.tree.feature_action.emit("section_assignments", "delete", "SA-椽")
    assert not (s.model.get("features") or {}).get("section_assignments")


def test_the_prompt_resets_after_a_region_is_delivered(qt_app):
    """真窗口实测发现：点「完成」后提示区仍停在「选择节点…」，像是还在等输入。"""
    w = MainWindow(built())
    w.region_pick.start("node", "测试", lambda region: None)
    w.viewport.region_selection.apply("node", [1])
    w.region_pick.finish()
    assert "选择" not in w.lbl_prompt.text()


def test_assign_beam_orientation_follows_the_region_first_flow(qt_app, monkeypatch):
    s = built()
    w = MainWindow(s)
    _stub(monkeypatch, "OrientationDialog", values=("Ori-1", [0.0, 1.0, 0.0]))
    w.actions_by_name["beam_orientation"].trigger()
    assert w.region_pick.active and w.viewport.region_pick == "member"
    target = s.model["members"][0]["id"]
    w.viewport.region_selection.apply("member", [target])
    w.region_pick.finish()
    member = next(m for m in s.model["members"] if m["id"] == target)
    assert member["ref_vector"] == [0.0, 1.0, 0.0] and member["orientation"] == "Ori-1"


def test_create_load_can_name_a_new_case_on_the_spot(qt_app, monkeypatch):
    """荷载属于工况；原来没有工况时只弹提示让用户去别处建，现在输入新名字即就地新建。"""
    s = built()
    w = MainWindow(s)
    _stub(monkeypatch, "CreateObjectDialog", values=("屋面活载", "LL", "line_load"))
    _stub(monkeypatch, "EditLoadDialog", vector=[0.0, 0.0, -500.0])
    w.create_load()
    w.viewport.region_selection.apply("member", [s.model["members"][0]["id"]])
    w.region_pick.finish()
    assert "LL" in [c["name"] for c in s.model["load_cases"]]
    assert [f["name"] for f in s.model["features"]["loads"]] == ["屋面活载"]


def test_seed_edges_follows_the_region_flow_and_seed_part_sets_the_default(qt_app, monkeypatch):
    """网格模块的布种：全局段数与逐杆覆盖（Abaqus Seed Part / Seed Edges）。"""
    from desktop import window_objects
    s = built()
    w = MainWindow(s)
    answers = iter([3, 8])
    monkeypatch.setattr(window_objects.QInputDialog, "getInt",
                        staticmethod(lambda *a, **k: (next(answers), True)))
    w.seed_part()
    assert s.model["mesh"]["divisions"] == 3
    target = s.model["members"][0]["id"]
    w.seed_edges()
    w.viewport.region_selection.apply("member", [target])
    w.region_pick.finish()
    assert s.model["mesh"]["members"] == {str(target): 8}
    w.clear_seeds()
    assert "mesh" not in s.model


def _synchronous_runner(w, monkeypatch):
    """作业在后台线程里跑；测试里改成同步执行，回调立刻发生。"""
    def submit(fn, on_done=None, on_failed=None, on_progress=None):
        on_done(fn())
        return True
    monkeypatch.setattr(w.runner, "submit", submit)


def test_job_manager_records_how_and_what_was_solved(qt_app, monkeypatch):
    """作业管理器：数据检查、提交、消息、结果与模型版本对应。

    原来 F5 直接求解，算过什么、出了什么警告只在状态栏闪一下；改了模型之后旧结果
    还对不对应当前模型，只能靠记忆。
    """
    from desktop.job_manager import Job
    s = built()
    w = MainWindow(s)
    _synchronous_runner(w, monkeypatch)
    job = Job("Job-1", "cases", "linear")
    w.jobs.append(job)
    w.job_action("check", job)
    assert job.status == "数据检查通过"
    w.job_action("submit", job)
    assert job.status == "完成" and job.finished
    assert any("平衡校核通过" in m for m in job.messages)
    assert any("静默失败检查" in m for m in job.messages)

    infos = []
    from desktop import window_jobs
    monkeypatch.setattr(window_jobs.QMessageBox, "information",
                        staticmethod(lambda *a, **k: infos.append(a[2])))
    w.job_action("results", job)
    assert not infos and w.mode == "变形"
    s.model["nodes"][0]["x"] += 0.1                     # 改了模型
    w.job_action("results", job)
    assert infos and "旧模型" not in infos[-1] and "重新提交" in infos[-1]


def test_a_steps_job_reports_frequency_and_buckling_steps(qt_app, monkeypatch):
    from desktop.job_manager import Job
    s = built()
    case = s.model["load_cases"][0]["name"] if s.model.get("load_cases") else "default"
    assert s.add_step("S1", loads={case: "RAMP"}).ok
    assert s.add_step("F1", analysis="frequency", num_modes=2).ok
    w = MainWindow(s)
    _synchronous_runner(w, monkeypatch)
    job = Job("Job-步", "steps")
    w.jobs.append(job)
    w.job_action("submit", job)
    assert job.status == "完成", job.messages
    assert any("F1（频率）" in m and "Hz" in m for m in job.messages)


def test_display_group_hides_members_from_drawing_and_picking(qt_app):
    """显示组：仅显示 / 隐藏 / 全部显示；隐藏的杆件不进绘制用的 Frame，也就不参与拾取。"""
    s = built()
    w = MainWindow(s)
    ids = [m["id"] for m in s.model["members"]]
    w.display_only()
    w.viewport.region_selection.apply("member", ids[:1])
    w.region_pick.finish()
    assert w.viewport.hidden_members == set(ids[1:])
    view = w.viewport.visible_frame(w.session.preview_frame())
    assert set(view.members) == {ids[0]}
    assert set(view.nodes) == set(w.session.preview_frame().nodes), "节点不过滤，位移下标不能错位"
    w.display_all()
    assert not w.viewport.hidden_members


def test_path_plot_draws_along_a_chain_of_members(qt_app, monkeypatch):
    from desktop import window_viz
    s = built()
    w = MainWindow(s)
    solved(w)
    shown = []

    import desktop.path_plot as path_plot
    original = path_plot.PathPlotDialog

    class Capture:
        def __init__(self, window, ordered):
            self.dialog = original(window, ordered)
            shown.append(self.dialog.draw())

        def exec(self):
            return 0
    monkeypatch.setattr(path_plot, "PathPlotDialog", Capture)
    first = s.model["members"][0]
    neighbour = next(m for m in s.model["members"][1:]
                     if {m["i"], m["j"]} & {first["i"], first["j"]})
    w.path_plot()
    w.viewport.region_selection.apply("member", [first["id"], neighbour["id"]])
    w.region_pick.finish()
    assert shown and shown[0]["points"] > 0
    assert window_viz is not None


def test_animation_toggle_needs_a_result_view(qt_app, monkeypatch):
    from desktop import window_viz
    w = MainWindow(built())
    told = []
    monkeypatch.setattr(window_viz.QMessageBox, "information",
                        staticmethod(lambda *a, **k: told.append(a[2])))
    w.actions_by_name["animate"].setChecked(True)
    w.toggle_animation()
    assert told and not w.actions_by_name["animate"].isChecked()


def test_part_commands_partition_datum_and_measure(qt_app, monkeypatch):
    """Part 模块的三个命令都走提示区拾取：分割杆件、两节点中点基准点、测距离。"""
    from desktop import window_objects
    s = built()
    w = MainWindow(s)
    monkeypatch.setattr(window_objects.QInputDialog, "getDouble",
                        staticmethod(lambda *a, **k: (0.5, True)))
    member = s.model["members"][0]
    before = len(s.model["nodes"])
    w.partition_member_ui()
    w.viewport.region_selection.apply("member", [member["id"]])
    w.region_pick.finish()
    assert len(s.model["nodes"]) == before + 1

    choices = iter(["两节点中点（在视口选两个节点）", "两节点距离"])
    monkeypatch.setattr(window_objects.QInputDialog, "getItem",
                        staticmethod(lambda *a, **k: (next(choices), True)))
    w.datum_point_ui()
    w.viewport.region_selection.apply("node", [member["i"], member["j"]])
    w.region_pick.finish()
    assert list(s.datum_points()) == ["Datum-1"]
    w.redraw()
    assert "Datum-1" in w.viewport.datum_points, "重画时把基准点同步给视口（显示与吸附）"
    w.measure_ui()
    w.viewport.region_selection.apply("node", [member["i"], member["j"]])
    w.region_pick.finish()
    assert "距离" in w.lbl_prompt.text()


def test_create_constraint_picks_master_then_slaves(qt_app, monkeypatch):
    """Interaction 模块的创建约束：名称与类型 → 选一个主节点 → 选从节点区域 → 编辑。

    结果是一个约束对象：视口画出主从连线，模型树有「约束」分支，求解时偏心点随主节点动。
    """
    s = built()
    eave = next(n for n in s.model["nodes"] if n["z"] == pytest.approx(7.5))
    assert s.add_nodes([[eave["x"], 0.6, eave["z"]]]).ok
    slave = max(n["id"] for n in s.model["nodes"])
    w = MainWindow(s)
    _stub(monkeypatch, "CreateObjectDialog", values=("刚臂", "", "beam"))
    _stub(monkeypatch, "EditConstraintDialog", values=("beam", eave["id"], None))
    w.create_constraint_ui()
    assert w.region_pick.active and "主节点" in w.lbl_prompt.text()
    w.viewport.region_selection.apply("node", [eave["id"]])
    w.region_pick.finish()
    assert w.region_pick.active and "从节点" in w.lbl_prompt.text()
    w.viewport.region_selection.apply("node", [slave])
    w.region_pick.finish()
    assert [f["name"] for f in s.model["features"]["constraints"]] == ["刚臂"]
    w.redraw()
    assert [link[0] for link in w.viewport.constraint_links] == ["刚臂"]
    labels = [w.tree.topLevelItem(k).text(0) for k in range(w.tree.topLevelItemCount())]
    assert "约束（1）" in labels
    assert s.solve_model().ok


def test_bc_manager_modifies_and_deactivates_in_the_selected_step(qt_app, monkeypatch):
    """边界条件管理器的步矩阵：点选某一步的格子，编辑即在该步修改，失活从该步起生效。"""
    from desktop.object_dialogs import ObjectManagerDialog
    s = Session()
    assert s.add_nodes([[0, 0, 0], [6, 0, 0]]).ok
    assert s.define_materials_and_sections(
        [{"name": "S", "E": 2e11, "nu": 0.3}],
        [{"name": "R", "A": 1e-2, "Iy": 2e-5, "Iz": 8e-5, "J": 1e-5}]).ok
    assert s.add_members([[1, 2]]).ok
    assert s.create_section_assignment("SA", [1], "R", "S").ok
    assert s.create_boundary_condition("BC-左", [1], preset="ENCASTRE").ok
    assert s.create_boundary_condition("BC-右", [2], preset="ENCASTRE").ok
    assert s.set_load_cases([{"name": "DL"}]).ok
    assert s.create_load("均布", "DL", "line_load", [1], w=[0, 0, -1e4]).ok
    assert s.add_step("S1", loads={"DL": "RAMP"}).ok and s.add_step("S2").ok and s.add_step("S3").ok
    w = MainWindow(s)
    manager = ObjectManagerDialog(w, "bcs")
    assert manager.steps == ["Initial", "S1", "S2", "S3"]
    row = next(r for r in range(manager.table.rowCount()) if manager.table.item(r, 0).text() == "BC-右")
    first = len(manager.COLUMNS["bcs"])
    _stub(monkeypatch, "EditBCDialog", values=([1, 1, 1, 0, 0, 0], None))
    manager.table.setCurrentCell(row, first + 2)                     # S2 列
    manager._run("edit")
    manager.table.setCurrentCell(row, first + 3)                     # S3 列
    manager._run("deactivate")
    feature = next(f for f in s.model["features"]["bcs"] if f["name"] == "BC-右")
    assert feature["modifications"]["S2"]["fix"] == [1, 1, 1, 0, 0, 0]
    assert feature["deactivated"] == "S3"
    cells = [manager.table.item(row, first + k).text() for k in range(4)]
    assert cells == ["创建", "传播", "修改", "失活"]
    assert s.solve_steps().ok


def test_gravity_and_batch_assignment_create_objects_like_their_abaqus_counterparts(qt_app, monkeypatch):
    """重力按钮与批量属性指派不再写逐杆直接条目：得到的是载荷管理器、截面指派管理器里的对象。

    原来同一件事有两条路，结果是两种数据：一种能在管理器里编辑、抑制，另一种只能去表格里删。
    """
    s = built()
    w = MainWindow(s)
    _stub(monkeypatch, "CreateObjectDialog", values=("Gravity-1", "D", "gravity"))
    _stub(monkeypatch, "EditLoadDialog", vector=[0.0, 0.0, -9.80665])
    w.apply_gravity()
    assert w.region_pick.active
    assert sorted(w.viewport.region_selection.ids) == sorted(m["id"] for m in s.model["members"]), \
        "重力的区域默认全部杆件"
    w.region_pick.finish()
    load = next(f for f in s.model["features"]["loads"] if f["name"] == "Gravity-1")
    assert load["type"] == "gravity"

    from desktop import assignment_dialog
    target = [m["id"] for m in s.model["members"][:2]]

    class _Batch:
        def __init__(self, *a, **k):
            self.target = type("T", (), {"member_ids": lambda _self: target})()

        def exec(self):
            from PySide6.QtWidgets import QDialog
            return QDialog.DialogCode.Accepted

        def get_assignment(self):
            return "COLUMN", "Q355"

    monkeypatch.setattr(assignment_dialog, "BatchAssignmentDialog", _Batch)
    w.assign_member_group()
    assignment = s.model["features"]["section_assignments"][0]
    assert assignment["region"] == {"members": sorted(target)}
    assert "support" not in w.actions_by_name, "支座管理器与边界条件管理器是同一个，只留一个入口"


def test_results_tree_lists_steps_frames_and_modes(qt_app):
    """结果树按 分析步 → 帧 / 模态 展开；点增量帧画的是那一帧的位移，点模态画那一步的振型。"""
    s = built()
    assert s.add_step("静力", loads={"D": "RAMP"}).ok
    assert s.add_step("二阶", analysis="pdelta", increments=3).ok
    assert s.add_step("频率", analysis="frequency", num_modes=2).ok
    w = MainWindow(s)
    got = s.solve_steps()
    assert got.ok, got.payload
    w._steps_solved(got)
    root = next(w.tree.topLevelItem(k) for k in range(w.tree.topLevelItemCount())
                if w.tree.topLevelItem(k).text(0) == "结果")
    steps = [root.child(k).text(0) for k in range(root.childCount())]
    assert steps == ["静力　线性静力", "二阶　P-Δ", "频率　频率"]
    assert root.child(1).childCount() == 3 and root.child(2).childCount() == 2
    w._on_tree_action("result_frame", ("二阶", 0))
    assert w.result_frame == 0
    shown = w._frame_solution()[w.case].U
    assert (shown == s.solution[w.case].frames[0][1]).all()
    w._on_tree_action("result_mode", ("频率", 1))
    assert "第 2 阶" in w.statusBar().currentMessage()


def test_linear_pattern_from_the_assembly_page_and_delete_in_the_manager(qt_app, monkeypatch):
    """装配页：选部件杆件 → 线性阵列参数 → 副本；模型树列实例；实例管理器整份删除。"""
    from desktop import window_assembly
    from desktop.assembly_dialogs import InstanceManagerDialog
    s = built()
    before = len(s.model["members"])
    w = MainWindow(s)
    monkeypatch.setattr(window_assembly, "InstanceDialog", lambda *a, **k: _Accept(values={
        "name": "沿Y", "direction1": [0, 1, 0], "spacing1": 6.0, "number1": 3,
        "direction2": [0, 0, 0], "spacing2": 0.0, "number2": 1}))
    w.linear_pattern_ui()
    assert w.region_pick.active and "部件" in w.lbl_prompt.text()
    w.viewport.region_selection.apply("member", [m["id"] for m in s.model["members"]])
    w.region_pick.finish()
    assert len(s.model["members"]) == 3 * before
    labels = [w.tree.topLevelItem(k).text(0) for k in range(w.tree.topLevelItemCount())]
    assert "装配（2 个实例）" in labels
    monkeypatch.setattr(window_assembly.QMessageBox, "question",
                        staticmethod(lambda *a, **k: window_assembly.QMessageBox.StandardButton.Yes))
    manager = InstanceManagerDialog(w)
    manager.table.selectRow(0)
    manager._delete(pattern=True)
    assert len(s.model["members"]) == before and "assembly" not in s.model


def test_picks_in_a_refined_result_view_report_physical_members(qt_app):
    """P-Δ、分析步把杆件剖成多段求解：结果视图里点选、框选、显示组都要按物理杆件。

    真窗口验收发现：14 根杆件的模型点中一根，状态栏和探针写的是「杆件 32」——
    那是分析杆段号，用户在模型里找不到。
    """
    s = built()
    assert s.add_step("二阶", analysis="pdelta", loads={"D": "RAMP"}, increments=2).ok
    w = MainWindow(s)
    got = s.solve_steps()
    assert got.ok, got.payload
    w._steps_solved(got)
    w.redraw()
    owner = w.viewport.element_owner
    physical = {int(m["id"]) for m in s.model["members"]}
    assert set(owner.values()) == physical and len(owner) > len(physical)
    member = min(physical)
    segments = w.viewport._elements([member])
    assert len(segments) > 1 and all(owner[e] == member for e in segments)
    w.viewport.set_hidden_members({member})
    view = w.viewport.visible_frame(s.frame)
    assert not set(segments) & set(view.members), "隐藏物理杆件要隐藏它的全部分析杆段"
    last = segments[-1]
    w.probe_member_result(last, s.frame.nodes[s.frame.members[last].j].xyz)
    caption = w.results.caption.text()
    assert f"杆件 {member}（分析杆段 {last}）" in caption
    whole = next(m for m in s.model["members"] if m["id"] == member)
    ends = [next(n for n in s.model["nodes"] if n["id"] == whole[e]) for e in ("i", "j")]
    length = sum((ends[1][k] - ends[0][k]) ** 2 for k in "xyz") ** 0.5
    assert f"x={length:.4g}/{length:.4g}" in caption, "位置按整根物理杆件量"


def test_hiding_a_member_in_model_view_after_solving_hides_all_its_segments(qt_app):
    """求解后切回「模型」显示，画的仍是分析模型：隐藏一根柱要整根隐藏，而不是只藏一段。"""
    s = built()
    assert s.add_step("二阶", analysis="pdelta", loads={"D": "RAMP"}, increments=2).ok
    w = MainWindow(s)
    got = s.solve_steps()
    assert got.ok
    w._steps_solved(got)
    w.set_mode("模型")
    member = min(int(m["id"]) for m in s.model["members"])
    assert len(w.viewport._elements([member])) > 1
    w.viewport.set_hidden_members({member})
    view = w.viewport.visible_frame(s.frame)
    assert not set(w.viewport._elements([member])) & set(view.members)


def _section_dialog_returning(section: dict):
    from PySide6.QtWidgets import QDialog

    class _Dialog:
        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            return QDialog.DialogCode.Accepted

        def get_section(self):
            return dict(section)
    return _Dialog


def test_created_section_keeps_its_real_geometry(qt_app, monkeypatch):
    """防止「创建截面」写回模型时丢掉 geometry。

    真实问题：白名单只放行 A/I/J/cy/cz 等数值，对话框建的工字钢、矩形管存进模型
    就没有轮廓，节点实体报“缺少真实轮廓尺寸”，只有圆管、圆钢能从 A/I 反推。
    """
    from desktop import section_dialog

    raw = section_dialog.calc_section("I 型钢", {"h": .4, "b": .2, "tw": .008, "tf": .013, "r": .016})
    raw.update(name="HN400", type="I 型钢", params={})
    monkeypatch.setattr(section_dialog, "SectionDialog", _section_dialog_returning(raw))
    w = MainWindow(built())
    w.create_section()
    stored = next(s for s in w.session.model["sections"] if s["name"] == "HN400")
    assert stored["geometry"] == {"type": "i", "dimensions": {
        "h": .4, "b": .2, "tw": .008, "tf": .013, "r": .016}}
    assert "type" not in stored and "params" not in stored


def test_editing_a_section_replaces_every_derived_field(qt_app, monkeypatch):
    """防止同名截面重建时旧字段残留：原先用 dict.update，旧 geometry、cy/cz、
    Ay/Az 会留在新尺寸上，节点实体随即报面积不一致、强度验算用错纤维距离。"""
    from desktop import section_dialog

    s = built()
    old = next(sec for sec in s.model["sections"] if sec["name"] == "COLUMN")
    old.update(geometry={"type": "i", "dimensions": {"h": .7, "b": .25, "tw": .01, "tf": .016}},
               cy=.35, cz=.125, Ay=1e-3, Az=1e-3)
    raw = section_dialog.calc_section("槽钢", {"h": .25, "b": .09, "tw": .008, "tf": .012})
    raw.update(name="COLUMN", type="槽钢", params={})
    monkeypatch.setattr(section_dialog, "SectionDialog", _section_dialog_returning(raw))
    w = MainWindow(s)
    w.create_section()
    stored = next(sec for sec in w.session.model["sections"] if sec["name"] == "COLUMN")
    assert stored["geometry"]["type"] == "channel"
    assert not {"cy", "cz", "Ay", "Az"} & set(stored)
    assert stored["A"] == pytest.approx(raw["A"], rel=1e-12)
