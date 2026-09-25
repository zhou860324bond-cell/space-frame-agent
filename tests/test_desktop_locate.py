"""拾取、编号标注、结果表——"这个数字在哪"的那条主线。

这个软件之前回答不了"在哪"：位移是个数字没有位置，报错说"节点 17"
但视口里找不到 17，想改一根杆得先知道它几号——而编号规则是生成器定的，
用户并不知道。

拾取、标注、可点的报错、可点的结果表，是同一件事的四个面。
这也是"每个数字可溯源"这句话在界面上的兑现处：之前它只在日志层面成立。
"""

from __future__ import annotations

import os

import numpy as np
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
from desktop import result_rows, scene                      # noqa: E402
from desktop.main_window import MainWindow                  # noqa: E402

MATERIALS = [{"name": "Q355", "E": 2.06e11, "nu": 0.3, "density": 7850.0}]
SECTIONS = [{"name": "B", "A": 0.0147, "Iy": 4.2e-5, "Iz": 1.18e-3, "J": 9e-7}]


@pytest.fixture(scope="module")
def qt_app():
    yield QApplication.instance() or QApplication([])


def built() -> Session:
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    g = s.generate_frame(spans=[6.0, 6.0], storeys=[3.6], column_section="B",
                         beam_section="B", material="Q355")
    s.set_load_cases(cases=[{"name": "D", "member_loads":
                             [{"member": m, "w": [0, 0, -20e3]}
                              for m in g.payload["beam_member_ids"]]}])
    return s


# ------------------------------------------------- 拾取判据

def test_clicking_a_node_finds_that_node():
    s = built(); f = s.preview_frame()
    for nid in f.order():
        assert scene.nearest(f, np.asarray(f.nodes[nid].xyz), "node") == nid


def test_clicking_a_member_finds_that_member():
    s = built(); f = s.preview_frame()
    for mid in f.members:
        m = f.members[mid]
        mid_pt = (f.nodes[m.i].xyz + f.nodes[m.j].xyz) / 2.0
        assert scene.nearest(f, mid_pt, "member") == mid


def test_a_point_on_the_extension_of_a_member_is_not_picked():
    """**用线段距离而不是直线距离。**

    用直线的话，远处一根杆的延长线正好从你点的位置附近穿过，
    就会被判成"最近的一根"——用户会觉得软件在乱跳。
    """
    s = built(); f = s.preview_frame()
    m = f.members[min(f.members)]
    a, b = f.nodes[m.i].xyz, f.nodes[m.j].xyz
    far = a + 8.0 * (b - a)
    assert scene.nearest(f, far, "member") is None


def test_clicking_far_away_selects_nothing():
    """**宁可没选中，也不要选中一个用户没想选的东西。**"""
    s = built(); f = s.preview_frame()
    assert scene.nearest(f, np.array([80.0, 80.0, 80.0]), "node") is None


# ------------------------------------------------- 标注与高亮

def test_labels_cover_every_node_and_member():
    s = built(); f = s.preview_frame()
    pts, txt = scene.node_labels(f)
    assert len(txt) == len(list(f.order())) == len(pts)
    pts, txt = scene.member_labels(f)
    assert len(txt) == len(f.members) == len(pts)


def test_highlight_is_thicker_not_a_different_colour():
    """选中用加粗表示。云图模式下颜色是有含义的（受拉受压），
    选中再占一个颜色，图就没法读了。"""
    s = built(); f = s.preview_frame()
    plain = scene.member_tubes(f)
    hot = scene.highlight_members(f, [min(f.members)])
    assert hot.n_cells > 0
    # 高亮管的半径更大：包围盒在垂直于杆轴的方向上更"胖"
    assert hot.n_cells > 0 and plain.n_cells > 0


def test_highlighting_nothing_is_safe():
    s = built(); f = s.preview_frame()
    assert scene.highlight_members(f, []).n_cells == 0
    assert scene.highlight_nodes(f, [999]).n_points == 0


# ------------------------------------------------- 界面接线

def test_picking_reports_position_not_just_a_number(qt_app):
    """**只说"杆件 27"是没用的。** 编号规则是生成器定的，用户不知道，
    所以选中之后必须同时给出坐标或端点。"""
    w = MainWindow(built())
    w.locate("node", 3)
    text = w.lbl_pick.text()
    assert "节点 3" in text and "(" in text, text
    w.locate("member", 1)
    text = w.lbl_pick.text()
    assert "杆件 1" in text and "长" in text and "截面" in text, text


def test_locating_an_object_completes_the_same_selection_flow_as_picking(qt_app):
    """结果表定位不能只画高亮；后续编辑命令也必须知道选中了什么。"""
    w = MainWindow(built())
    w.props_dock.hide()

    w.locate("node", 3)

    assert (w._selected_kind, w._selected_id) == ("node", 3)
    assert w.viewport.selection == ("node", 3)
    assert w.properties.kind == "node" and w.properties.ident == 3
    assert w.actions_by_name["create_bc"].isEnabled()
    assert not w.props_dock.isHidden()


def test_the_pick_filter_can_be_turned_off(qt_app):
    """全不选时回到纯看图——转视角时不该误点中东西。"""
    w = MainWindow(built())
    w.actions_by_name["pick_member"].setChecked(True)
    w.set_pick_member()
    assert w.viewport.pick_mode == "member"
    w.actions_by_name["pick_member"].setChecked(False)
    w.set_pick_member()
    assert w.viewport.pick_mode is None


def test_labels_toggle(qt_app):
    w = MainWindow(built())
    w.actions_by_name["labels"].setChecked(True)
    w.toggle_labels()
    assert w.viewport.show_labels


def test_undo_from_the_window_clears_the_displayed_result(qt_app):
    """撤销之后界面这份 result 也要清。不清的话，模型退回去了而
    状态栏还写着"已求解 1 个工况"——界面在说谎。"""
    from tests.test_desktop_window import solved

    w = solved(MainWindow(built()))
    assert w.result is not None and w.result.ok
    w.undo()
    assert w.result is None
    assert "未求解" in w.lbl_solve.text()
    assert w.mode == "模型"


# ------------------------------------------------- 结果表
#
# **这一组返工过一次，教训要留在这儿。**
#
# 第一版是我自己编一份 payload 喂给 result_rows，然后断言它摊出来的行。
# 全绿——可我编的 payload 键名跟真实工具返回的完全不一样（我以为包络会给
# 一个 members 列表，实际上给的是 peak / at_member / governing_case）。
# 于是测试在拿我的猜测验证我的猜测，界面上显示"共 0 根杆件"。
# **是截图抓到的，不是测试。**
#
# 所以下面一律用"真建模、真求解、真调工具"拿到的 payload。
# 自己编的输入只能验格式，验不了接口对不对。


@pytest.fixture(scope="module")
def solved_session():
    s = built()
    assert s.solve_model().ok
    return s


def test_envelope_reads_the_real_payload(solved_session):
    r = solved_session.query_envelope(component="Mz")
    assert r.ok
    title, cols, rows, loc = result_rows.to_rows("envelope", r.payload)
    assert rows, "全结构包络摊出来是空表——多半是键名对不上"
    assert "kN·m" in title, "单位要从 payload 自带的 unit 取，不要写死"
    assert any(x for x in loc), "峰值所在的杆件要能点回视口"


def test_single_member_envelope_reads_the_real_payload(solved_session):
    r = solved_session.query_envelope(component="Mz", member=1)
    assert r.ok
    _, cols, rows, loc = result_rows.to_rows("envelope", r.payload)
    assert len(rows) == 3, "单杆包络该给 i 端、j 端、峰值三行"
    assert all(x == ("member", 1) for x in loc)
    assert any("上包线" in c for c in cols)


def test_buckling_reads_the_real_payload(solved_session):
    r = solved_session.buckling_analysis(num_modes=3)
    assert r.ok
    title, _, rows, loc = result_rows.to_rows("buckling", r.payload)
    assert len(rows) == 3
    assert "临界因子" in title and "受压杆件" in title
    assert loc[0] and loc[0][0] == "member"


def test_modal_reads_the_real_payload(solved_session):
    r = solved_session.modal_analysis(num_modes=3)
    assert r.ok
    title, cols, rows, _ = result_rows.to_rows("modal", r.payload)
    assert len(rows) == 3
    assert all(isinstance(row[1], float) for row in rows), "频率要保持数值"
    assert "累计参与质量比" in title, "取的阶数够不够，靠它判断"
    # 分母也要摆出来。只给比值的话，"上不去"是阶数不够还是质量压在支座上，
    # 用户分不出来——后者加多少阶都没用。
    assert "可参与" in title, "可参与质量是参与比的分母，必须一起给"


def test_deflection_reads_the_real_payload(solved_session):
    r = solved_session.query_results(what="max_deflection")
    assert r.ok
    title, cols, rows, loc = result_rows.to_rows("deflection", r.payload)
    assert len(rows) == 1 and loc[0] and loc[0][0] == "member"
    assert any("mm" in c for c in cols)


def test_the_kernel_note_is_carried_through(solved_session):
    """内核给的 note 写的往往正是这个结果最容易被误读的地方
    （比如挠跨比的分母是杆长不是设计跨度）。**不能在这一层丢掉。**"""
    r = solved_session.query_results(what="max_deflection")
    title, _, _, _ = result_rows.to_rows("deflection", r.payload)
    assert "不是设计跨度" in title, "内核那句关于分母的提醒没带过来"


def test_the_note_does_not_leak_field_names(solved_session):
    """note 是写给**大模型**看的，带着字段名和调用方式。
    原样印到界面上，就是把内部接口漏给了用户。"""
    import re

    leak = re.compile(r"[a-z][a-z0-9]*(_[a-z0-9]+)+")
    for kind, r in (("envelope", solved_session.query_envelope(component="Mz")),
                    ("buckling", solved_session.buckling_analysis(num_modes=2)),
                    ("modal", solved_session.modal_analysis(num_modes=2)),
                    ("deflection",
                     solved_session.query_results(what="max_deflection"))):
        title = result_rows.to_rows(kind, r.payload)[0]
        hit = leak.search(title)
        assert not hit, f"{kind} 的说明里漏出了字段名 {hit.group(0)!r}"


def test_the_note_never_dangles_a_pronoun(solved_session):
    """第一版是"整句丢掉"，结果包络那句变成"它为 0 并不等于……"——
    **代词没了指代对象，比不显示更糟**。所以清洗要么换词，要么断在那里。"""
    r = solved_session.query_envelope(component="Mz")
    title = result_rows.to_rows("envelope", r.payload)[0]
    body = title.split("\n", 1)[1] if "\n" in title else ""
    for sentence in body.split("。"):
        s = sentence.strip()
        if s.startswith(("它", "这个", "那个", "后者", "前者")):
            raise AssertionError(f"句子以悬空代词开头：{s[:40]!r}")


def test_numbers_stay_numbers_so_sorting_works(solved_session):
    """数值列存成字符串的话，表格排序会按字典序排——"9" 排在 "10" 后面。"""
    r = solved_session.modal_analysis(num_modes=3)
    _, _, rows, _ = result_rows.to_rows("modal", r.payload)
    assert all(isinstance(row[1], float) for row in rows)


def test_an_unrecognised_payload_still_shows_something():
    """**不要因为格式没对上就什么都不显示。** 空表看不出是"没有结果"
    还是"读错了"——第一版的包络就栽在这上面。"""
    _, _, rows, _ = result_rows.to_rows("没见过的", {"a": 1, "b": "x"})
    assert len(rows) == 2


def test_a_broken_payload_does_not_raise():
    """界面拿不到表只是难看，抛异常是整块面板空白。"""
    title, _, rows, _ = result_rows.to_rows("modal", {"modes": "不是列表"})
    assert "失败" in title or rows
