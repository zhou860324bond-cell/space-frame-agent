"""结果摘要与窄屏判读回归，防止精简界面隐藏校核信息或误报通过。"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt                                      # noqa: E402
from PySide6.QtWidgets import QApplication, QToolButton             # noqa: E402

from desktop import result_rows                                   # noqa: E402
from desktop.panels import ResultPanel                             # noqa: E402
from agent import Session                                          # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def payload(load=200e3, yield_stress=235e6):
    session = Session()
    material = {"name": "M", "E": 2e11, "nu": 0.3, "allow_tension": 100e6}
    if yield_stress is not None:
        material["yield_stress"] = yield_stress
    assert session.set_model(model={
        "schema_version": 1, "units": "N-m-Pa",
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                  {"id": 2, "x": 0.5 if load < 0 else 5, "y": 0, "z": 0}],
        "members": [{"id": 1, "i": 1, "j": 2, "section": "S", "material": "M"}],
        "materials": [material],
        "sections": [{"name": "S", "A": 4e-3, "Iy": 8e-6, "Iz": 8e-6,
                      "J": 1.6e-5, "cy": 0.05, "cz": 0.05}],
        "supports": [{"node": 1, "fix": [1, 1, 1, 1, 1, 1]},
                     {"node": 2, "fix": [0, 1, 1, 1, 1, 1]}],
        "load_cases": [{"name": "P", "nodal_loads": [{"node": 2, "load": [load, 0, 0, 0, 0, 0]}]}],
    }).ok
    assert session.solve_model().ok
    result = session.check_strength()
    assert result.ok
    return result.payload


def present(panel, data):
    title, columns, rows, locators = result_rows.to_rows("strength", data)
    panel.show_rows(title, columns, rows, locators,
                    result_rows.row_marks("strength", data),
                    overview=result_rows.strength_overview(data),
                    primary_columns={"杆件", "截面", "应力比 σ/[σ]", "折算比", "N/(φA)/f", "结论"})


@pytest.mark.parametrize("ratio", [0.99999, 1.00001])
def test_summary_verdict_uses_unrounded_tool_ratio(ratio):
    """真实轴力校核在阈值两侧显示值都约为 1，摘要不能用显示值重新判断。"""
    data = payload(ratio * 100e6 * 4e-3)
    cards = result_rows.strength_overview(data)
    assert cards[0]["value"] == "1"
    assert (cards[0]["state"] == "fail") == (ratio > 1)
    assert cards[0]["target"] == ("member", data["worst_strength"]["member"])


def test_inconclusive_compression_does_not_claim_every_member_passed():
    """材料缺少屈服应力时，粗短压杆未判定，标题不能仍宣称全部通过。"""
    data = payload(-200e3, yield_stress=None)
    assert data["inconclusive_members"] == [1]
    title, *_ = result_rows.to_rows("strength", data)
    assert "全部通过" not in title
    assert "可判定杆件均通过" in title
    assert result_rows.strength_overview(data)[3]["value"] == "1"


def test_key_columns_keep_shear_and_code_stability_visible(app):
    """精简结果列时必须同时保留剪切折算比和规范稳定比，避免只读正应力。"""
    panel = ResultPanel()
    present(panel, payload())
    visible = {panel.table.horizontalHeaderItem(i).text()
               for i in range(panel.table.columnCount()) if not panel.table.isColumnHidden(i)}
    assert visible == {"杆件", "截面", "应力比 σ/[σ]", "折算比", "N/(φA)/f", "结论"}
    panel.all_columns.setChecked(True)
    assert not any(panel.table.isColumnHidden(i) for i in range(14))
    panel.show_rows("位移", ["节点", "位移"], [[1, 0.1]])
    assert not panel.table.isColumnHidden(1)
    assert panel.all_columns.isHidden()
    assert panel.summary.isHidden()


def test_calculation_notes_preserve_tool_limits_and_errors_are_expanded(app):
    """折叠冗长说明不能丢失工具的适用限制；计算失败的下一步应直接可见。"""
    panel = ResultPanel()
    data = payload()
    present(panel, data)
    assert panel.notes.isHidden()
    assert data["limitation"].replace("**", "") in panel.notes.toPlainText()
    panel.details_button.click()
    assert not panel.notes.isHidden()
    panel.show_message("验算未完成", "材料参数不全，请补充屈服应力后重新验算。")
    assert not panel.notes.isHidden()
    assert "请补充屈服应力" in panel.notes.toPlainText()
    assert panel.summary.isHidden()


@pytest.mark.parametrize("load,yield_stress,card,mark", [
    (600e3, 235e6, 2, "fail"), (-200e3, None, 3, "unclear")])
def test_summary_links_use_tool_member_and_filter_after_sort(app, load, yield_stress, card, mark):
    """摘要定位和超限/未判定清单必须关联真实物理杆件，不能依赖表格行号。"""
    panel = ResultPanel()
    data = payload(load, yield_stress)
    present(panel, data)
    located = []
    panel.locate.connect(lambda *target: located.append(target))
    panel.table.sortItems(0, Qt.SortOrder.DescendingOrder)
    panel.summary_cards[0].findChild(QToolButton).click()
    assert located == [("member", 1)]
    panel.summary_cards[card].findChild(QToolButton).click()
    assert panel.row_filter.currentData() == mark
    assert not panel.table.isRowHidden(0)


def test_narrow_summary_keeps_metrics_above_readable_table(app):
    """窄屏摘要应换为两列，不能将指标挤成空白条。"""
    panel = ResultPanel()
    present(panel, payload())
    panel.resize(450, 420)
    panel.show()
    app.processEvents()
    assert panel.summary_grid.getItemPosition(2)[:2] == (1, 0), (panel.width(), panel.minimumSizeHint().width())
    assert all(card.height() >= 92 for card in panel.summary_cards)
    assert panel.table.height() >= 60
    panel.close()
