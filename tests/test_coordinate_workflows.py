"""坐标定位、区间终点、完整 CSV 及真实结果点查询的回归（移植自主工作区 7804922）。

按坐标施加跨间荷载的对话框测试未移植：master 的创建载荷已改为对象流程，
目前不含跨间荷载类型，需先扩展荷载对象。
"""

from copy import deepcopy
import csv
import io
import json

import numpy as np
import pytest

from agent import Session
from coordinates import parse_coordinate, station_on_segment
from csv_export import export_csv
from model_tables import (SPAN_LOAD_COLUMNS, from_csv, from_tables, to_csv,
                          to_tables)
from sections import rectangle
from units import convert_model


def cantilever():
    """建立可与悬臂闭合解比较的真实求解模型。"""
    session = Session()
    payload = {"units": "N-m-Pa",
               "materials": [{"name": "S", "E": 2.1e11, "nu": 0.3}],
               "sections": [rectangle("B", .2, .4)],
               "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                         {"id": 2, "x": 6, "y": 0, "z": 0}],
               "members": [{"id": 1, "i": 1, "j": 2, "section": "B", "material": "S"}],
               "supports": [{"node": 1, "fix": [1] * 6}],
               "load_cases": [{"name": "P", "nodal_loads": [
                   {"node": 2, "load": [0, 0, -1000, 0, 0, 0]}]},
                   {"name": "Q", "nodal_loads": [
                       {"node": 2, "load": [0, 0, -2000, 0, 0, 0]}]}],
               "combos": [{"name": "PQ", "factors": {"P": 1, "Q": 1}}]}
    assert session.set_model(model=payload).ok
    assert session.solve_model().ok
    return session


@pytest.fixture
def qt_app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize("scale", [1, 1000])
def test_oblique_coordinate_station_is_unit_invariant(scale):
    """防止斜杆坐标被当作 X 距离或在 mm 模型中使用米。"""
    assert station_on_segment(np.array([1.5, 2, 0]) * scale,
                              [0, 0, 0], np.array([3, 4, 0]) * scale) == 2.5 * scale


@pytest.mark.parametrize("point", [[3, .001, 0], [-1, 0, 0], [7, 0, 0], [np.nan, 0, 0]])
def test_coordinate_load_rejects_off_axis_and_outside(point):
    """防止坐标定位静默投影离杆点或将越界荷载吸附到杆端。"""
    with pytest.raises(ValueError):
        station_on_segment(point, [0, 0, 0], [6, 0, 0])


@pytest.mark.parametrize("kind", ["partial", "partial_trapezoid", "moment", "trapezoid"])
def test_table_roundtrip_keeps_span_ends_and_other_modules(kind):
    """防止表格编辑丢弃 b、w2、力矩位置及分析步、幅值等非表格内容。"""
    session = cantilever()
    model = deepcopy(session.model)
    model["steps"] = [{"name": "Later", "custom": 123}]
    model["amplitudes"] = [{"name": "History", "values": [0, 1]}]
    entry = {"member": 1, "kind": kind, "w1": [0, 0, -10], "name": "Span"}
    if kind in {"partial", "partial_trapezoid"}:
        entry.update(a=1.2, b=4.7)
    if kind == "moment":
        entry["a"] = 2.1
    if kind in {"trapezoid", "partial_trapezoid"}:
        entry["w2"] = [0, 0, -30]
    model["load_cases"][0]["member_spans"] = [entry]
    back = from_tables(model, to_tables(model))
    assert back["load_cases"][0]["member_spans"] == [entry]
    assert back["steps"] == model["steps"]
    assert back["amplitudes"] == model["amplitudes"]


def test_old_span_csv_and_new_trapezoid_endpoint():
    """防止增加 b 后旧 CSV 无法导入，同时确保梯形区间终点参与求解。"""
    model = cantilever().model
    row = {"case": "P", "member": 1, "kind": "trapezoid",
           "w1x": 0, "w1y": 0, "w1z": -10,
           "w2x": 0, "w2y": 0, "w2z": -20, "a": 1}
    columns = tuple(c for c in SPAN_LOAD_COLUMNS if c != "b")
    restored = from_csv(to_csv([row], columns), SPAN_LOAD_COLUMNS)
    tables = to_tables(model)
    tables["member_spans"] = restored
    assert from_tables(model, tables)["load_cases"][0]["member_spans"][0]["kind"] == "trapezoid"
    restored[0]["b"] = 5
    span = from_tables(model, tables)["load_cases"][0]["member_spans"][0]
    assert (span["kind"], span["a"], span["b"]) == ("partial_trapezoid", 1, 5)


def test_kind_dropdown_and_row_deletion_keep_correct_values(qt_app):
    """防止删除行后类型下拉框仍写入旧行号，或空白行被默认类型创建。"""
    from desktop.dialogs import TableDialog
    dialog = TableDialog(cantilever())
    table = dialog.tables["member_spans"]
    column = SPAN_LOAD_COLUMNS.index("kind")
    combo = table.cellWidget(1, column)
    assert [combo.itemData(i) for i in range(combo.count())] == ["", "trapezoid", "point"]
    combo.setCurrentIndex(2)
    table.removeRow(0)
    assert dialog._rows_of("member_spans") == [{c: "point" if c == "kind" else ""
                                              for c in SPAN_LOAD_COLUMNS}]


def test_string_false_support_is_not_fixed(qt_app):
    """防止桌面表格中的字符串 False 被 bool() 转换成约束开启。"""
    from desktop.dialogs import TableDialog
    session = cantilever()
    session.model["supports"][0]["fix"][5] = 0
    dialog = TableDialog(session)
    model = from_tables(session.model, dialog.collect())
    assert model["supports"][0]["fix"] == [1, 1, 1, 1, 1, 0]


def csv_rows(text):
    return list(csv.DictReader(io.StringIO(text.lstrip("\ufeff"))))


def test_full_csv_all_cases_fields_modules_and_no_resolve(monkeypatch):
    """防止 CSV 只导出当前页或当前工况，也防止导出覆盖原求解结果。"""
    session = cantilever()
    solution = session.solution
    monkeypatch.setattr(session, "solve_model", lambda: pytest.fail("导出不应重新求解"))
    analyses = {"buckling": (session._model_key(), {"factors": [3.14]})}
    text = export_csv(session, analyses=analyses)
    assert text.startswith("\ufeff") and "\r\n" in text
    rows = csv_rows(text)
    modules = {row["module"] for row in rows}
    assert {f"model.{key}" for key in session.model} <= modules
    for case in ("P", "Q", "PQ"):
        assert any(f"/steps/{case}/" in row["path"] and row["status"] == "COMPLETED"
                   for row in rows if row["module"] == "results")
    assert any(row["module"] == "analysis.buckling" and json.loads(row["value"]) == 3.14
               for row in rows)
    assert any(row["module"] == "analysis.modal" and row["status"] == "NOT_AVAILABLE"
               for row in rows)
    assert session.solution is solution


def test_csv_changed_draft_excludes_original_results(qt_app):
    """防止未应用的表格修改与原模型的位移内力被导出为同一份有效结果。"""
    from desktop.dialogs import TableDialog
    from PySide6.QtWidgets import QTableWidgetItem
    session = cantilever()
    dialog = TableDialog(session)
    baseline = from_tables(session.model, dialog.collect())
    assert any(row["status"] == "COMPLETED" for row in csv_rows(export_csv(session, model=baseline)))
    dialog.tables["nodes"].setItem(1, 1, QTableWidgetItem("7"))
    edited = from_tables(session.model, dialog.collect())
    rows = csv_rows(export_csv(session, model=edited))
    assert not any(row["status"] == "COMPLETED" for row in rows)
    assert any(row["module"] == "model.nodes" and row["path"] == "/1/x"
               and json.loads(row["value"]) == 7 for row in rows)


def test_coordinate_results_actual_node_and_interior_displacement(qt_app):
    """防止坐标查询把杆中位移当节点线性插值，或乘上变形显示放大系数。"""
    from desktop.main_window import MainWindow
    session = cantilever()
    window = MainWindow(session)
    window.result = session.solve_summary
    window.case = "Q"
    window.query_coordinate("3, 0, 0")
    probe = window._last_probe
    assert window.results.notes.isHidden()
    assert not window.results.details_button.isHidden()
    section = session.frame.sections["B"]
    expected = -2000 * 3**2 * (3 * 6 - 3) / (6 * 2.1e11 * section.Iz) * 1000
    assert probe["global_displacement"][2] == pytest.approx(expected, rel=1e-4)
    assert probe["stress"] is not None
    window.scale = 10000
    window.query_coordinate("6, 0, 0")
    assert "节点 2" in window.results.caption.text()
    assert "Q" in window.results.caption.text()
    window.close()


def test_coordinate_intersection_requires_member_and_rejects_nan():
    """防止交叉点查询任意采用第一根杆件，或非有限坐标进入结果计算。"""
    from desktop.result_inspector import coordinate_target
    from frame3d import Member, Node
    frame = cantilever().frame
    frame.nodes[3] = Node(3, 3, -1, 0)
    frame.nodes[4] = Node(4, 3, 1, 0)
    frame.members[2] = Member(2, 3, 4, "B", "S")
    with pytest.raises(ValueError, match="唯一"):
        coordinate_target(frame, [3, 0, 0])
    assert coordinate_target(frame, [3, 0, 0], [2]) == ("member", 2)
    with pytest.raises(ValueError):
        parse_coordinate("nan, 0, 0")


def test_coordinate_query_mm_matches_si(qt_app):
    """防止坐标输入使用 mm 后杆件位置、位移或应力的换算发生漂移。"""
    from desktop.main_window import MainWindow
    from desktop.result_inspector import probe_member
    session = cantilever()
    before = probe_member(session.frame, session.solution, 1, [3, 0, 0], "P")
    assert session.set_model(model=convert_model(session.model, "N-mm-MPa")).ok
    assert session.solve_model().ok
    window = MainWindow(session)
    window.result = session.solve_summary
    window.case = "P"
    window.query_coordinate("3000,0,0")
    assert window._last_probe["global_displacement"] == pytest.approx(before["global_displacement"])
    assert window._last_probe["stress"]["max"] == pytest.approx(before["stress"]["max"])
    window.close()


def test_deformed_mouse_member_pick_uses_original_station(qt_app):
    """防止点击放大后的梁管时以显示坐标定位到错误杆件或错误截面。"""
    from desktop import scene
    from desktop.main_window import MainWindow
    session = cantilever()
    window = MainWindow(session)
    window.result = session.solve_summary
    line = scene.member_polylines(session.frame, session.solution, "P", scale=10000)
    index = len(line.points) // 2

    class Picker:
        def GetDataSet(self):
            return line

        def GetPointId(self):
            return index

    window.viewport._frame = session.frame
    window.viewport.pick_mode = "member"
    readings = []
    window.viewport.probed.connect(lambda mid, point: readings.append((mid, point)))
    window.viewport._on_pick(line.points[index], Picker())
    assert readings[0][0] == 1
    assert readings[0][1] == pytest.approx([3, 0, 0])
    assert abs(line.points[index, 2]) > 1
    window.close()


def test_full_csv_button_exports_input_and_remembered_analysis(qt_app, monkeypatch, tmp_path):
    """防止全部模块按钮遗漏已完成分析或输出未经 BOM 编码的中文 CSV。"""
    from desktop.main_window import MainWindow
    from PySide6.QtWidgets import QFileDialog
    session = cantilever()
    window = MainWindow(session)
    window.result = session.solve_summary
    window._sync_result_validity()
    window._remember_analysis("buckling", {"factors": [2.5]})
    path = tmp_path / "全部模块.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(path), ""))
    window.export_all_csv()
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    rows = csv_rows(path.read_text(encoding="utf-8-sig"))
    assert any(r["module"] == "analysis.buckling" and r["status"] == "COMPLETED" for r in rows)
    old = session._model_key()
    assert session.set_nodal_load(2, [0, 0, -3000, 0, 0, 0], "P").ok
    assert session._model_key() != old
    window.export_all_csv()
    assert not any(r["status"] == "COMPLETED"
                   for r in csv_rows(path.read_text(encoding="utf-8-sig")))
    window.close()


def test_split_member_coordinate_selects_correct_analysis_segment(qt_app):
    """防止杆中集中力自动剖分后，按物理杆件号查询总是读取第一段。"""
    from desktop.main_window import MainWindow
    session = cantilever()
    assert session.set_member_span_load(1, "point", [0, 0, -500], a=2, case_name="P").ok
    assert session.solve_model().ok
    window = MainWindow(session)
    window.result = session.solve_summary
    window.case = "P"
    window.query_coordinate("4,0,0", "1")
    assert window._last_probe["point"] == pytest.approx([4, 0, 0])
    assert window._last_probe["x"] == pytest.approx(2)
    window.query_coordinate("2,0,0", "1")
    assert window._last_probe["x"] == pytest.approx(0)
    window.close()
