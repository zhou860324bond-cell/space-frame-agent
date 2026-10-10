"""模型树名称筛选、精确编号和视口联动回归。"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyvistaqt")

from PySide6.QtWidgets import QApplication  # noqa: E402
from desktop.main_window import MainWindow  # noqa: E402
from desktop.model_tree import ACTION, PAYLOAD, ModelTree, ModelTreePanel  # noqa: E402
from tools.render_smoke import build_session  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_number_search_is_exact_and_name_search_keeps_ancestors(app):
    """搜索节点 1 不得混入节点 10；名称筛选仍应保留可见的分类路径。"""
    session = build_session()
    tree = ModelTree()
    panel = ModelTreePanel(tree)
    tree.rebuild(session)
    panel.search.setText("节点 1")
    assert len(panel.matches) == 1
    item = panel.matches[0]
    assert item.data(0, PAYLOAD) == ("node", 1)
    assert not item.parent().isHidden() and not item.parent().parent().isHidden()
    assert tree._objects[("node", 10)].isHidden()
    panel.search.setText("COL")
    assert any(item.data(0, ACTION) == "locate_object" for item in panel.matches)
    panel.search.setText("节点 9999")
    assert not panel.matches and "没有匹配项" in panel.status.text()


def test_search_enter_and_viewport_pick_share_selection(app):
    """编号定位和视口拾取应使用同一对象选择，不得重复创建选择状态。"""
    window = MainWindow(build_session())
    window.tree_panel.search.setText("杆件 2")
    window.tree_panel.locate_match()
    assert window.viewport.selection == ("member", 2)
    assert window.tree.currentItem().data(0, PAYLOAD) == ("member", 2)
    window.tree_panel.search.clear()
    window.viewport.picked.emit("node", 10)
    assert window.tree.currentItem().data(0, PAYLOAD) == ("node", 10)
    assert window.properties.kind == "node"


def test_tree_cache_preserves_selection_but_detects_same_count_model_edits(app):
    """无修改刷新不能丢失树选择；同数量的坐标或截面修改仍须更新树。"""
    session = build_session()
    tree = ModelTree()
    tree.rebuild(session)
    tree.select_object("node", 1)
    item = tree.currentItem()
    tree.rebuild(session)
    assert tree.currentItem() is item
    session.model["nodes"][0]["x"] = 7
    tree.rebuild(session)
    assert "(7，" in tree._objects[("node", 1)].text(0)
    assert tree.currentItem().data(0, PAYLOAD) == ("node", 1)
    session.model["members"][0]["section"] = "BEAM"
    tree.rebuild(session)
    assert "BEAM" in tree._objects[("member", 1)].text(0)


def test_clearing_search_restores_collapsed_branches_and_rebuild_keeps_filter(app):
    """清除搜索应恢复用户收起状态；模型变化后不能悄悄解除筛选。"""
    session = build_session()
    tree = ModelTree()
    panel = ModelTreePanel(tree)
    tree.rebuild(session)
    geometry = next(tree.topLevelItem(i) for i in range(tree.topLevelItemCount())
                    if tree.topLevelItem(i).text(0) == "几何")
    geometry.setExpanded(False)
    panel.search.setText("节点 1")
    assert geometry.isExpanded()
    session.model["nodes"][0]["x"] = 9
    tree.rebuild(session)
    assert len(panel.matches) == 1 and "(9，" in panel.matches[0].text(0)
    panel.search.clear()
    geometry = tree._objects[("node", 1)].parent().parent()
    assert not geometry.isExpanded()
    assert all(not item.isHidden() for item in panel._items())
