"""工作平面与画图状态机。

**这两块是「三维里画图」的全部难点，也全都能在没有窗口的情况下测。**

视口里点一下给到的是一条射线，不是一个点；吸附会悄悄改坐标；
连杆连到一半换模式会留下半根杆。这些都是逻辑，不是渲染——
放进 Qt 里去测，只会把逻辑错误藏在渲染问题后面。
"""

from __future__ import annotations

import math

import pytest

from draft import FrameDraft
from drawing import DrawingSession
from workplane import NODE_SNAP, OffPlane, WorkPlane, plane_through

CARRIED = {"units": "N-m-Pa",
           "materials": [{"name": "Q355", "E": 2.06e11, "nu": 0.3}],
           "sections": [{"name": "COLUMN", "A": 0.01, "Iy": 1e-4,
                         "Iz": 1e-4, "J": 1e-5}]}


def session(grid: float = 0.5, mode: str = "node") -> DrawingSession:
    s = DrawingSession(FrameDraft(carried=dict(CARRIED)),
                       WorkPlane("XZ", 0.0, grid), section="COLUMN")
    s.set_mode(mode)
    return s


# --------------------------------------------------------- 射线 → 平面

def test_a_click_becomes_one_definite_point_on_the_plane():
    """屏幕上一个像素对应射线上无数个点，工作平面挑出唯一那个。"""
    wp = WorkPlane("XZ", offset=0.0)
    # 相机在 y = -20 处正对着看
    assert wp.intersect((3.2, -20.0, 4.1), (0, 1, 0)) == (3.2, 0.0, 4.1)
    # 斜着看也一样能定位：t = 10
    hit = wp.intersect((0.0, -10.0, 0.0), (0.5, 1.0, 0.25))
    assert hit == pytest.approx((5.0, 0.0, 2.5))


def test_a_second_bay_is_drawn_on_its_own_plane():
    """offset 就是「第几榀」。拉伸成多榀后，每一榀是一个平面。"""
    wp = WorkPlane("XZ", offset=7.5)
    assert wp.intersect((3.0, -20.0, 4.0), (0, 1, 0))[1] == 7.5


def test_looking_along_the_plane_refuses_instead_of_guessing():
    """视线与平面平行时不能猜——猜出来的是个莫名其妙的远处节点。"""
    with pytest.raises(OffPlane, match="转开"):
        WorkPlane("XZ").intersect((0, 0, 0), (1, 0, 0))


def test_the_plane_kind_decides_which_coordinate_is_frozen():
    for kind, frozen in (("XY", 2), ("XZ", 1), ("YZ", 0)):
        wp = WorkPlane(kind, offset=2.0)
        assert wp.project((9.0, 9.0, 9.0))[frozen] == 2.0


def test_an_unknown_plane_is_rejected_at_construction():
    with pytest.raises(ValueError):
        WorkPlane("斜面")


# --------------------------------------------------------- 吸附

def test_snapping_prefers_an_existing_node_over_the_grid():
    """**顺序反了就永远连不上柱顶。**

    层高 3.6、网格 1.0 时，柱顶不在网格点上。先吸网格的话，
    点柱顶会被拉到 4.0，连出来的梁和柱是断开的。
    """
    wp = WorkPlane("XZ", 0.0, grid=1.0)
    nodes = [{"id": 7, "x": 0.0, "y": 0.0, "z": 3.6}]
    point, hit, how = wp.snap((0.05, 0.0, 3.62), nodes)
    assert hit == 7 and point == (0.0, 0.0, 3.6)
    assert "节点 7" in how


def test_snapping_falls_back_to_the_grid_when_no_node_is_near():
    wp = WorkPlane("XZ", 0.0, grid=0.5)
    point, hit, how = wp.snap((2.34, 0.0, 1.19), [])
    assert (point, hit) == ((2.5, 0.0, 1.0), None)
    assert "0.5" in how


def test_a_node_just_outside_the_tolerance_is_not_grabbed():
    wp = WorkPlane("XZ", 0.0, grid=0.0)
    nodes = [{"id": 1, "x": 0.0, "y": 0.0, "z": 0.0}]
    _, hit, _ = wp.snap((NODE_SNAP * 2, 0.0, 0.0), nodes)
    assert hit is None, "吸附范围太大，想在柱子旁边新建点就永远建不了"


def test_grid_lines_lie_on_the_plane():
    """画不出来的网格没法瞄准，而画错平面的网格比没有更糟。"""
    wp = WorkPlane("XZ", offset=7.5, grid=1.0)
    lines = wp.grid_lines(half_size=3.0)
    assert lines
    assert all(a[1] == 7.5 and b[1] == 7.5 for a, b in lines)


def test_the_plane_can_align_itself_to_an_existing_bay():
    """省掉手填 offset：填错一个数，画的东西落在另一榀上，图上看不出来。"""
    nodes = [{"x": 0, "y": 7.5, "z": 0}, {"x": 6, "y": 7.5, "z": 0},
             {"x": 0, "y": 0.0, "z": 0}]
    assert plane_through(nodes, "XZ") == 7.5


# --------------------------------------------------------- 画

def test_drawing_a_portal_frame_by_clicking():
    """把「画一榀门架」这件事整个走一遍。"""
    s = session(grid=0.5, mode="member")
    # **四下画完一榀。** 连成一根之后终点自动接着画下一根，
    # 每根都重点一次起点的话要八下，而且重点时点偏就会多出重合节点
    clicks = [(0, 0, 0), (0, 0, 4.0), (8.0, 0, 4.0), (8.0, 0, 0)]
    for p in clicks:
        assert s.click(p).ok
    assert len(s.draft.members) == 3
    assert len(s.draft.nodes) == 4, "柱顶被两根杆共用，不该出现重合节点"

    s.set_mode("support")
    for p in ((0, 0, 0), (8.0, 0, 0)):
        assert s.click(p).ok
    assert s.draft.can_finish(), s.draft.finish_diagnostics()


def test_clicking_empty_space_in_member_mode_places_the_node_first():
    """画图要连贯：不该逼用户先切到放点模式再切回来。"""
    s = session(mode="member")
    r = s.click((0, 0, 0))
    assert r.ok and r.created == ("node", 1) and r.changed


def test_clicking_the_same_node_twice_does_not_make_a_zero_length_member():
    s = session(mode="member")
    s.click((0, 0, 0))
    r = s.click((0.01, 0, 0.01))            # 点偏了一点，还是同一个节点
    assert not r.ok and "同一个节点" in r.message
    assert s.pending_node == 1, "多半只是点偏了，起点不该丢"
    assert not s.draft.members


def test_a_duplicate_member_is_refused_with_a_reason():
    s = session(mode="member")
    s.click((0, 0, 0)); s.click((0, 0, 3.0))
    s.cancel_pending()
    s.click((0, 0, 0)); r = s.click((0, 0, 3.0))
    assert not r.ok and "已经有杆件" in r.message
    assert len(s.draft.members) == 1
    assert s.pending_node is None, "被拒之后要回到干净状态，不然下一下更奇怪"


def test_switching_mode_drops_a_half_drawn_member():
    """**留着半根杆，下一模式的第一次点击会莫名连出一根。**"""
    s = session(mode="member")
    s.click((0, 0, 0))
    assert s.pending_node is not None
    s.set_mode("node")
    assert s.pending_node is None


def test_escape_cancels_the_pending_member():
    s = session(mode="member")
    s.click((0, 0, 0))
    assert s.cancel_pending() is True
    assert s.cancel_pending() is False, "没东西可取消时要如实回答，界面据此决定要不要提示"


def test_support_mode_toggles_and_only_lands_on_nodes():
    s = session(mode="member")
    s.click((0, 0, 0)); s.click((0, 0, 3.0))
    s.set_mode("support")

    assert not s.click((5.0, 0, 5.0)).ok, "支座不能加在空气里"
    assert s.click((0, 0, 0)).ok
    assert s.draft.supports[0]["fix"] == [1] * 6
    r = s.click((0, 0, 0))                  # 再点一次取消
    assert r.ok and "取消" in r.message and not s.draft.supports


def test_delete_mode_hits_the_member_body_not_just_its_midpoint():
    """用户点的是杆身。按到中点的距离判，长杆几乎点不中。"""
    s = session(mode="member")
    s.click((0, 0, 0)); s.click((0, 0, 10.0))
    s.set_mode("delete")
    r = s.click((0.1, 0, 8.5))              # 靠近上端，离中点 3.5 m
    assert r.ok and ("member", 1) in r.removed
    assert not s.draft.members


def test_deleting_a_node_says_which_members_went_with_it():
    s = session(mode="member")
    s.click((0, 0, 0)); s.click((0, 0, 3.0)); s.click((3.0, 0, 3.0))
    s.set_mode("delete")
    r = s.click((0, 0, 3.0))
    assert r.ok and "连带" in r.message
    assert len(r.removed) == 3              # 一个节点 + 两根杆


def test_undo_during_drawing_steps_back_one_click():
    """**画图中的撤销和提交后的撤销粒度不同，是故意的。**

    画的时候你想退上一根杆；画完之后你想退掉整次修改。
    """
    s = session(mode="member")
    s.click((0, 0, 0)); s.click((0, 0, 3.0))       # 杆 1
    s.click((3.0, 0, 3.0))                          # 杆 2（接着画）
    s.click((3.0, 0, 0.0))                          # 杆 3
    assert len(s.draft.members) == 3

    assert s.undo_click()
    assert len(s.draft.members) == 2, "退一次只该退掉最后那根杆"
    while s.undo_click():
        pass
    assert not s.draft.nodes and not s.draft.members
    assert s.undo_click() is False


def test_a_refused_click_is_not_undoable_on_its_own():
    """没生效的点击不该占一格撤销——按一次撤销什么都没变，最让人困惑。"""
    s = session(mode="support")
    steps = s.steps
    assert not s.click((9.0, 0, 9.0)).ok
    assert s.steps == steps


def test_the_status_line_says_where_you_are_and_what_is_next():
    s = session(mode="member")
    assert "XZ" in s.status() and "连杆件" in s.status()
    s.click((0, 0, 0))
    assert "Esc" in s.status(), "半根杆的状态必须写在脸上，否则下一下点出来的东西没人预料得到"


def test_snapping_is_announced_when_it_moves_the_point():
    """吸附是隐形的。0.5 m 网格上点 3.6 会落到 3.5——不说就是坑。"""
    s = session(grid=0.5, mode="member")
    s.click((0, 0, 0))
    r = s.click((0, 0, 3.6))
    assert "3.5" in r.message and "网格" in r.message


def test_chaining_can_be_broken_and_resumed_elsewhere():
    """**必须有个明确的出口。** 不然想在别处另起一根时，
    第一下会从上一根的终点连过去，画出一根谁也没打算要的杆。
    """
    s = session(mode="member")
    s.click((0, 0, 0)); s.click((0, 0, 3.0))
    assert s.pending_node is not None, "画完一根应该接着画下一根"

    s.cancel_pending()
    s.click((9.0, 0, 0.0))                  # 在远处另起一根
    assert len(s.draft.members) == 1, "断链之后第一下不该连出杆件"
    s.click((9.0, 0, 3.0))
    assert len(s.draft.members) == 2
    ends = {(m["i"], m["j"]) for m in s.draft.members}
    assert (2, 3) not in ends and (3, 2) not in ends, "把两处画的东西连起来了"


def test_drawing_a_two_bay_frame_takes_no_redundant_clicks():
    """一条连贯的折线画完两跨，节点数正好，不多不少。"""
    s = session(grid=0.5, mode="member")
    for p in [(0, 0, 0), (0, 0, 4.0), (6.0, 0, 4.0), (6.0, 0, 0)]:
        s.click(p)
    s.cancel_pending()
    for p in [(6.0, 0, 4.0), (12.0, 0, 4.0), (12.0, 0, 0)]:
        s.click(p)
    assert len(s.draft.nodes) == 6, [(n["x"], n["z"]) for n in s.draft.nodes]
    assert len(s.draft.members) == 5, (
        "两跨刚架：左柱/左梁/中柱（第一跨3根）+ 右梁/右柱（第二跨2根）= 5根，"
        "中柱顶被两跨共用，不会多出第6根"
    )
    assert not s.draft.coincident_nodes(), "中柱顶被两跨共用，不该出现重合节点"
