"""荷载对象的沿杆定位类型：杆上集中力、杆上力矩、区间均布、区间梯形。

防的问题：#32 把「创建载荷」改成对象流程后，荷载对象只有集中力、力矩、满跨线荷载、
重力四种，杆中集中力、局部堆载、梯形导荷、跨间力偶从界面上建不出来，只能绕道
Agent 的 set_member_span_load 或模型表格；主工作区原有的「按全局坐标定位荷载」
也因此没有入口。这里把四种类型补进 features，再生成到 member_spans，并在
「编辑荷载」对话框里提供距 i 端距离与全局坐标两种定位方式。

判据：闭合解（悬臂、简支的反力与端弯矩、悬臂端挠度），单位不变性，旧对象往返
不变，界面流程把坐标换成的 a、b 写进对象。
"""

from __future__ import annotations

import json
import os
from copy import deepcopy

import numpy as np
import pytest

import features
from agent import Session
from model_io import from_dict
from units import convert_model

E, NU = 2.06e11, 0.3
A_, IY, IZ, J = 8e-3, 2e-5, 1.5e-4, 5e-7
L = 6.0


def _beam(fix_i, fix_j=None, length: float = L) -> Session:
    """沿全局 X 的单根杆件 1→2；竖向荷载绕局部 z（强轴 Iz）弯曲。"""
    s = Session()
    payload = {"units": "N-m-Pa",
               "materials": [{"name": "S", "E": E, "nu": NU}],
               "sections": [{"name": "H", "A": A_, "Iy": IY, "Iz": IZ, "J": J}],
               "nodes": [{"id": 1, "x": 0.0, "y": 0.0, "z": 0.0},
                         {"id": 2, "x": length, "y": 0.0, "z": 0.0}],
               "members": [{"id": 1, "i": 1, "j": 2, "section": "H", "material": "S"}],
               "supports": [{"node": 1, "fix": fix_i}]
               + ([{"node": 2, "fix": fix_j}] if fix_j else []),
               "load_cases": [{"name": "DL"}]}
    assert s.set_model(model=payload).ok
    return s


def _cantilever() -> Session:
    return _beam([1] * 6)


def _simply_supported() -> Session:
    # i 端铰（含扭转约束），j 端竖向与侧向滚动；平面内绕 Y 转动两端自由
    return _beam([1, 1, 1, 1, 0, 0], [0, 1, 1, 0, 0, 0])


def _reaction(s: Session, node: int) -> np.ndarray:
    assert s.solve_model().ok, s.last_result if hasattr(s, "last_result") else ""
    frame = from_dict(s.model)
    return s.solution["DL"].R[frame.node_dofs(node)]


P, W1, W2, A, B, M0 = 1.2e4, 3.0e3, 7.0e3, 1.5, 4.0, 2.5e4


# (类型, create_load 参数, 合力 Fz, 合力对原点的矩 ∫x·q dx)
LOADS = {
    "member_point": ({"force": [0, 0, -P], "a": A}, -P, -P * A),
    "partial_line": ({"w": [0, 0, -W1], "a": A, "b": B},
                     -W1 * (B - A), -W1 * (B - A) * (A + B) / 2),
    "partial_trapezoid": ({"w": [0, 0, -W1], "w2": [0, 0, -W2], "a": A, "b": B},
                          -(W1 + W2) * (B - A) / 2,
                          -(B - A) * (W1 * (2 * A + B) + W2 * (A + 2 * B)) / 6),
}


@pytest.mark.parametrize("load_type", sorted(LOADS))
def test_cantilever_reactions_match_statics(load_type):
    """悬臂固定端反力 = −合力，绕 Y 的反力矩 = 合力对固定端的矩（x 沿杆，Fz 向下）。"""
    s = _cantilever()
    values, resultant, moment = LOADS[load_type]
    assert s.create_load("L", "DL", load_type, [1], **values).ok
    r = _reaction(s, 1)
    # 外荷载对原点的矩：(x,0,0)×(0,0,Fz) = (0, −x·Fz, 0)；反力矩与之相反
    assert r[2] == pytest.approx(-resultant, rel=1e-10)
    assert r[4] == pytest.approx(moment, rel=1e-10)
    assert np.allclose(r[[0, 1, 3, 5]], 0.0, atol=1e-9 * abs(resultant) * L)


def test_cantilever_member_moment_reaction_and_tip_rotation():
    """跨间力偶 M0（绕 Y）作用在 a：固定端反力矩 −M0、无反力；端转角 M0·a/(E·Iz)。"""
    s = _cantilever()
    assert s.create_load("M", "DL", "member_moment", [1], moment=[0, M0, 0], a=A).ok
    r = _reaction(s, 1)
    assert r[4] == pytest.approx(-M0, rel=1e-10)
    assert np.allclose(r[:3], 0.0, atol=1e-9 * M0 / L)
    frame = from_dict(s.model)
    ry = s.solution["DL"].U[frame.node_dofs(2)[4]]
    assert ry == pytest.approx(M0 * A / (E * IZ), rel=1e-9)


def test_cantilever_member_point_tip_deflection():
    """杆中集中力的悬臂端挠度 P·a²(3L−a)/(6EI)：位置 a 真的落在 a，而不是杆端。"""
    s = _cantilever()
    assert s.create_load("P", "DL", "member_point", [1], force=[0, 0, -P], a=A).ok
    assert s.solve_model().ok
    frame = from_dict(s.model)
    uz = s.solution["DL"].U[frame.node_dofs(2)[2]]
    assert uz == pytest.approx(-P * A ** 2 * (3 * L - A) / (6 * E * IZ), rel=1e-9)


@pytest.mark.parametrize("load_type", sorted(LOADS))
def test_simply_supported_reactions_follow_lever_rule(load_type):
    """简支梁两端竖向反力按杠杆原理分配：R_j = 合力矩 / L，R_i = 合力 − R_j。"""
    s = _simply_supported()
    values, resultant, moment = LOADS[load_type]
    assert s.create_load("L", "DL", load_type, [1], **values).ok
    ri, rj = _reaction(s, 1), _reaction(s, 2)
    assert rj[2] == pytest.approx(-moment / L, rel=1e-10)
    assert ri[2] == pytest.approx(-(resultant - moment / L), rel=1e-10)


def _all_types(s: Session) -> None:
    assert s.create_load("P", "DL", "member_point", [1], force=[0, 0, -P], a=A).ok
    assert s.create_load("M", "DL", "member_moment", [1], moment=[0, M0, 0], a=B).ok
    assert s.create_load("Q", "DL", "partial_line", [1], w=[0, -W1, 0], a=A, b=B).ok
    assert s.create_load("T", "DL", "partial_trapezoid", [1], w=[0, 0, -W1],
                         w2=[0, 0, -W2], a=0.0, b=L).ok


def test_span_load_objects_are_unit_invariant():
    """同一模型换成 N-mm-MPa：对象的 a、b、w2 与派生条目一起换，位移与反力逐位一致。

    顺带防的问题：convert_model 曾把跨间力偶（member_spans kind=moment）当线荷载换算，
    毫米制下力偶小 10⁶ 倍且不报错；这里另放一条直接定义的力偶覆盖它。
    """
    s = _cantilever()
    _all_types(s)
    assert s.set_member_span_load(1, "moment", [M0, 0, 0], a=A, case_name="DL",
                                  name="直接力偶").ok
    si = deepcopy(s.model)
    mm = Session()
    assert mm.set_model(model=convert_model(si, "N-mm-MPa")).ok
    loads = {f["name"]: f for f in mm.model["features"]["loads"]}
    assert loads["Q"]["a"] == pytest.approx(A * 1e3) and loads["Q"]["b"] == pytest.approx(B * 1e3)
    assert loads["T"]["w2"] == pytest.approx([0, 0, -W2 * 1e-3])
    assert s.solve_model().ok and mm.solve_model().ok
    frame = from_dict(si)
    u_si, u_mm = s.solution["DL"].U, mm.solution["DL"].U
    r_si, r_mm = s.solution["DL"].R, mm.solution["DL"].R
    for node in (1, 2):
        d = frame.node_dofs(node)
        np.testing.assert_allclose(u_mm[d[:3]], u_si[d[:3]] * 1e3, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(u_mm[d[3:]], u_si[d[3:]], rtol=1e-9, atol=1e-15)
        np.testing.assert_allclose(r_mm[d[:3]], r_si[d[:3]], rtol=1e-9, atol=1e-6)
        np.testing.assert_allclose(r_mm[d[3:]], r_si[d[3:]] * 1e3, rtol=1e-9, atol=1e-3)


def test_regenerate_writes_member_spans_and_round_trips(tmp_path):
    """对象 → member_spans 条目（带 owner）；存盘再读、再生成两次结果都不变。"""
    s = _cantilever()
    _all_types(s)
    spans = [e for e in s.model["load_cases"][0]["member_spans"] if e.get("owner")]
    by_owner = {e["owner"]: e for e in spans}
    assert by_owner["P"] == {"member": 1, "kind": "point", "w1": [0.0, 0.0, -P], "a": A,
                             "owner": "P"}
    assert by_owner["M"]["kind"] == "moment" and "b" not in by_owner["M"]
    assert (by_owner["Q"]["kind"], by_owner["Q"]["a"], by_owner["Q"]["b"]) == ("partial", A, B)
    assert by_owner["T"]["w2"] == [0.0, 0.0, -W2] and by_owner["T"]["b"] == L
    again = deepcopy(s.model)
    assert features.regenerate(again) == [] and again == s.model
    path = tmp_path / "模型.json"
    path.write_text(json.dumps(s.model, ensure_ascii=False), encoding="utf-8")
    loaded = Session()
    assert loaded.set_model(model=json.loads(path.read_text(encoding="utf-8"))).ok
    assert loaded.model == s.model


def test_existing_load_objects_are_unchanged():
    """旧四种荷载对象的字段与派生条目和扩展前完全一样（不需要迁移）。"""
    s = _cantilever()
    assert s.create_load("F", "DL", "concentrated_force", [2], force=[0, 0, -P]).ok
    assert s.create_load("W", "DL", "line_load", [1], w=[0, 0, -W1]).ok
    assert s.model["features"]["loads"] == [
        {"name": "F", "case": "DL", "type": "concentrated_force", "force": [0.0, 0.0, -P],
         "region": {"nodes": [2]}},
        {"name": "W", "case": "DL", "type": "line_load", "w": [0.0, 0.0, -W1],
         "region": {"members": [1]}}]
    case = s.model["load_cases"][0]
    assert case["nodal_loads"] == [{"node": 2, "load": [0.0, 0.0, -P, 0.0, 0.0, 0.0],
                                    "owner": "F"}]
    assert case["member_loads"] == [{"member": 1, "w": [0.0, 0.0, -W1], "owner": "W"}]
    assert "member_spans" not in case or not case["member_spans"]


def test_positions_are_checked_member_by_member():
    """区域里杆件长短不一：放得下的逐根派生；有放不下的整条拒绝并点名那根杆件。"""
    s = _cantilever()
    assert s.add_nodes([[L, 0, 3.0]]).ok
    assert s.add_members([[2, 3]]).ok              # 杆件 2 长 3 m
    s.model["members"][1].update(section="H", material="S")
    ok = s.create_load("P1", "DL", "member_point", [1, 2], force=[0, 0, -P], a=2.0)
    assert ok.ok
    assert sorted(e["member"] for e in s.model["load_cases"][0]["member_spans"]
                  if e.get("owner") == "P1") == [1, 2]
    bad = s.create_load("P2", "DL", "member_point", [1, 2], force=[0, 0, -P], a=4.0)
    assert not bad.ok
    text = json.dumps(bad.payload, ensure_ascii=False)
    assert "杆件 2" in text and "杆件 1 " not in text
    assert all(f["name"] != "P2" for f in s.model["features"]["loads"])


def test_create_load_rejects_missing_or_foreign_position():
    """沿杆荷载缺 a / b 要说清楚；满跨线荷载不接受 a，免得以为施加在了局部。"""
    s = _cantilever()
    missing = s.create_load("Q", "DL", "partial_line", [1], w=[0, 0, -W1], a=A)
    assert not missing.ok and "b" in missing.payload["error"]
    foreign = s.create_load("W", "DL", "line_load", [1], w=[0, 0, -W1], a=A)
    assert not foreign.ok and "a" in foreign.payload["error"]
    reversed_ = s.create_load("R", "DL", "partial_line", [1], w=[0, 0, -W1], a=B, b=A)
    assert not reversed_.ok
    negative = s.create_load("N", "DL", "member_point", [1], force=[0, 0, -P], a=-1.0)
    assert not negative.ok


def test_edit_feature_moves_the_load_and_regenerates():
    """管理器里改 a、b：派生条目跟着变，而不是叠加一条新的。"""
    s = _cantilever()
    assert s.create_load("Q", "DL", "partial_line", [1], w=[0, 0, -W1], a=A, b=B).ok
    assert s.edit_feature("load", "Q", {"a": 0.5, "b": 2.0}).ok
    spans = [e for e in s.model["load_cases"][0]["member_spans"] if e.get("owner") == "Q"]
    assert [(e["a"], e["b"]) for e in spans] == [(0.5, 2.0)]


# ---------------------------------------------------------------- 界面

@pytest.fixture
def qt_app():
    pytest.importorskip("PySide6", reason="未安装 PySide6，跳过桌面端测试")
    pytest.importorskip("pyvistaqt", reason="未安装 pyvistaqt，跳过桌面端测试")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _ends(length: float = L):
    return {1: (np.array([0.0, 0.0, 0.0]), np.array([length, 0.0, 0.0]))}


def test_dialog_converts_coordinates_to_stations(qt_app):
    """全局坐标经 station_on_segment 换成距 i 端距离；离杆、越界的点给出原因、不关闭。"""
    from desktop.object_dialogs import EditLoadDialog
    dialog = EditLoadDialog("Q", "partial_line", [0, 0, -1e3], "N-m-Pa", members=_ends())
    dialog.position.setCurrentIndex(dialog.position.findData("coordinate"))
    dialog.start_coordinate.setText("1, 0, 0")
    dialog.end_coordinate.setText("3，0，0")
    dialog._accept()
    assert dialog.result() == dialog.DialogCode.Accepted
    assert dialog.span() == {"a": pytest.approx(1.0), "b": pytest.approx(3.0)}
    for text in ("1, 0.2, 0", "7, 0, 0"):
        dialog = EditLoadDialog("P", "member_point", [0, 0, -1e3], "N-m-Pa", members=_ends())
        dialog.position.setCurrentIndex(dialog.position.findData("coordinate"))
        dialog.start_coordinate.setText(text)
        dialog._accept()
        assert dialog.result() != dialog.DialogCode.Accepted
        assert "中心线" in dialog.hint.text()


def test_dialog_rejects_coordinates_on_a_multi_member_region(qt_app):
    """区域有多根杆件时，坐标只能落在其中一根上：拒绝并点名它，提示改区域或改用距离。"""
    from desktop.object_dialogs import EditLoadDialog
    members = {**_ends(), 2: (np.array([L, 0.0, 0.0]), np.array([L, 0.0, 3.0]))}
    dialog = EditLoadDialog("P", "member_point", [0, 0, -1e3], "N-m-Pa", members=members)
    dialog.position.setCurrentIndex(dialog.position.findData("coordinate"))
    dialog.start_coordinate.setText("6, 0, 1")
    dialog._accept()
    assert dialog.result() != dialog.DialogCode.Accepted
    assert "杆件 [2]" in dialog.hint.text() and "距 i 端距离" in dialog.hint.text()
    dialog.position.setCurrentIndex(dialog.position.findData("distance"))
    dialog.start.setValue(4.0)                       # 杆件 2 只有 3 m
    dialog._accept()
    assert dialog.result() != dialog.DialogCode.Accepted and "[2]" in dialog.hint.text()
    dialog.start.setValue(2.0)
    dialog._accept()
    assert dialog.result() == dialog.DialogCode.Accepted and dialog.span() == {"a": 2.0}


@pytest.mark.parametrize("load_type, kind", [("member_point", "point"),
                                             ("member_moment", "moment"),
                                             ("partial_line", "partial"),
                                             ("partial_trapezoid", "partial_trapezoid")])
def test_create_load_window_writes_coordinate_selection(qt_app, monkeypatch, load_type, kind):
    """「创建载荷」走完对象流程：坐标换成的 a、b 写进对象，派生到 member_spans，旧结果作废。

    主工作区同名测试的对象化版本：防止对话框算出了位置，窗口却仍按满跨荷载提交。
    """
    from desktop import window_objects
    from desktop.main_window import MainWindow
    from desktop.object_dialogs import EditLoadDialog
    s = _cantilever()
    assert s.create_load("Tip", "DL", "concentrated_force", [2], force=[0, 0, -P]).ok
    assert s.solve_model().ok
    window = MainWindow(s)

    class Create:
        def __init__(self, *args, **kwargs):
            self.kind = None

        def exec(self):
            from PySide6.QtWidgets import QDialog
            return QDialog.DialogCode.Accepted

        def values(self):
            return "Coordinate", "DL", load_type

    def accept(dialog):
        dialog.position.setCurrentIndex(dialog.position.findData("coordinate"))
        dialog.start_coordinate.setText("1,0,0")
        dialog.end_coordinate.setText("3,0,0")
        dialog.values_[1 if load_type == "member_moment" else 2].setValue(-1000)
        if dialog.end_values:
            dialog.end_values[2].setValue(-2000)
        dialog._accept()
        return dialog.result()

    monkeypatch.setattr(window_objects, "CreateObjectDialog", Create)
    monkeypatch.setattr(EditLoadDialog, "exec", accept)
    window.create_load()
    assert window.region_pick.active and window.viewport.region_pick == "member"
    window.viewport.region_selection.apply("member", [1])
    window.region_pick.finish()
    feature = next(f for f in s.model["features"]["loads"] if f["name"] == "Coordinate")
    assert feature["type"] == load_type and feature["a"] == pytest.approx(1.0)
    span = next(e for e in s.model["load_cases"][0]["member_spans"]
                if e.get("owner") == "Coordinate")
    assert (span["kind"], span["a"]) == (kind, pytest.approx(1.0))
    if load_type.startswith("partial"):
        assert span["b"] == pytest.approx(3.0)
    else:
        assert "b" not in span and "b" not in feature
    if load_type == "partial_trapezoid":
        assert span["w2"] == [0.0, 0.0, -2000.0]
    assert s.solution is None
    window.close()
