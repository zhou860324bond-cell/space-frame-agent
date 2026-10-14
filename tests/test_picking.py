"""屏幕空间拾取与多选。

防的问题：原拾取先要点中 VTK 表面，再按「模型尺寸的 10%」在世界坐标里找最近对象——
容差跟着模型走而不是跟着手走，大模型里选到邻居、细杆点不中；一次只能选一个，
给二十个柱脚加约束要把整套流程走二十遍。
"""

from __future__ import annotations

import numpy as np
import pytest

from desktop import picking


def test_projection_maps_the_view_frustum_to_pixels():
    """单位矩阵相机：NDC (-1,-1)→左下角，(1,1)→右上角；y 轴翻成屏幕向下。"""
    pts = picking.project(np.eye(4), (200, 100), [[-1, -1, 0], [1, 1, 0], [0, 0, 0]])
    np.testing.assert_allclose(pts, [[0, 100], [200, 0], [100, 50]])


def test_points_behind_the_camera_are_not_pickable():
    matrix = np.eye(4)
    matrix[3] = [0, 0, -1, 0]                  # w = -z：z>0 的点在相机背后
    pts = picking.project(matrix, (100, 100), [[0, 0, 1.0]])
    assert np.isnan(pts).all()
    assert picking.nearest_point(pts, (50, 50)) is None


def test_click_tolerance_is_in_pixels_not_model_units():
    points = np.array([[10.0, 10.0], [40.0, 10.0]])
    assert picking.nearest_point(points, (17, 12)) == 0
    assert picking.nearest_point(points, (25, 10)) is None, "离两个点都超过 10 像素"


def test_a_member_is_picked_anywhere_along_its_screen_segment():
    segments = np.array([[[0, 0], [100, 0]], [[0, 50], [100, 50]]], dtype=float)
    assert picking.nearest_segment(segments, (60, 4)) == 0
    assert picking.nearest_segment(segments, (60, 46)) == 1
    assert picking.nearest_segment(segments, (60, 25)) is None


def test_box_selects_only_what_is_fully_inside():
    """与 Abaqus 默认一致：框住两端的杆件才算选中。"""
    segments = np.array([[[10, 10], [20, 20]], [[10, 10], [80, 80]]], dtype=float)
    assert picking.segments_in_rect(segments, (0, 0, 30, 30)).tolist() == [True, False]
    assert picking.points_in_rect(np.array([[5, 5], [50, 5]]), (30, 0, 0, 30)).tolist() == \
        [True, False], "反向拖框同样有效"


def test_selection_modes_replace_add_remove_and_kind_switch():
    sel = picking.Selection()
    sel.apply("node", [1, 2])
    sel.apply("node", [3, 2], "add")
    assert sel.ids == [1, 2, 3]
    sel.apply("node", [2], "remove")
    assert sel.ids == [1, 3]
    sel.apply("member", [7], "add")
    assert (sel.kind, sel.ids) == ("member", [7]), "换类型重新开始，节点与杆件不混在一个区域"
    with pytest.raises(ValueError):
        sel.apply("member", [1], "toggle")


def test_viewport_click_and_box_use_screen_geometry(qt_app, monkeypatch):
    """视口的点选与拖框在屏幕坐标上判定；无头环境下用替身几何验证逻辑。"""
    from agent import Session
    from desktop import scene
    from desktop.viewport import Viewport
    session = Session()
    session.add_nodes([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    view = Viewport()
    view._frame = scene.draft_frame(session.model)
    geometry = ([1, 2, 3], np.array([[10.0, 10.0], [50.0, 10.0], [90.0, 10.0]]),
                [], np.zeros((0, 2, 2)))
    monkeypatch.setattr(view, "_screen_geometry", lambda: geometry)
    seen = []
    view.region_changed.connect(lambda kind, ids: seen.append(list(ids)))
    view.start_region_pick("node")
    view.region_click(52, 12)
    view.region_click(12, 9, "add")
    view.region_box((40, 0, 100, 20), "remove")
    assert seen[-1] == [1]
    view.region_click(300, 300, "add")
    assert view.region_selection.ids == [1], "加选时点空不改选择"
    view.stop_region_pick()
    assert view.region_pick is None and not view.region_selection


@pytest.fixture(scope="module")
def qt_app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_path_ordering_follows_connectivity_and_rejects_branches():
    """路径曲线要把选中的杆件排成首尾相连的链；分叉或断开时说清原因。"""
    from desktop.path_plot import order_path
    members = {1: (1, 2), 2: (3, 2), 3: (3, 4), 4: (2, 5), 5: (7, 8)}
    assert order_path(members, [3, 1, 2]) == [(1, False), (2, True), (3, False)]
    with pytest.raises(ValueError, match="分叉"):
        order_path(members, [1, 2, 4])
    with pytest.raises(ValueError, match="不连续"):
        order_path(members, [1, 5])
