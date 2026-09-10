"""单杆内力图面板。

三维云图看的是全局分布，这里看的是**沿杆长的那条曲线**：在哪儿过零、
峰值在什么位置、各组合的包络有多宽。判断"这根梁跨中够不够"靠的是后者，
两者互相替代不了。

这块也是移植对账查出来的漏项——正则一开始报了个假的 ✔（桌面端确实
import 了 `member_diagram`，但只是给云图取标量）。**正则的 ✔ 不能当证据。**

最要紧的一条测试是：**面板算出来的包络必须和内核 `query_envelope`
给的数字一模一样。** 两条路算出不同的数，用户信哪个都是错的。
"""

from __future__ import annotations

import os
from copy import deepcopy

import numpy as np
import pytest

pytest.importorskip("PySide6", reason="未安装 PySide6，跳过桌面端测试")
pytest.importorskip("matplotlib", reason="未安装 matplotlib")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication                  # noqa: E402

from agent import Session                                   # noqa: E402
from desktop.diagram_panel import (COMPONENTS, DiagramPanel,
                                   diagram_features)         # noqa: E402

MATERIALS = [{"name": "Q355", "E": 2.06e11, "nu": 0.3, "density": 7850.0}]
SECTIONS = [{"name": "B", "A": 0.0147, "Iy": 4.2e-5, "Iz": 1.18e-3, "J": 9e-7}]


@pytest.fixture(scope="module")
def qt_app():
    yield QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def solved():
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    g = s.generate_frame(spans=[6.0], storeys=[3.6], column_section="B",
                         beam_section="B", material="Q355")
    beams = g.payload["beam_member_ids"]
    s.set_load_cases(
        cases=[{"name": "D", "member_loads":
                [{"member": m, "w": [0, 0, -20e3]} for m in beams]},
               {"name": "L", "member_loads":
                [{"member": m, "w": [0, 0, -10e3]} for m in beams]}],
        combos=[{"name": "1.3D+1.5L", "factors": {"D": 1.3, "L": 1.5}}])
    assert s.solve_model().ok
    return s, beams[0]


def panel(session, member: int, envelope: bool = False) -> DiagramPanel:
    p = DiagramPanel()
    p.attach(session, "D")
    p.member.setCurrentText(str(member))
    p.envelope.setChecked(envelope)
    return p


# ------------------------------------------------- 数值正确性

def test_the_curve_is_in_display_units_not_model_units(qt_app, solved):
    """**换算走 units.py 那一套，不要自己再写一份。**

    `member_diagram` 给的是模型内部单位（N·m）；直接标成 kN·m 就差一千倍，
    而图照样画得出来，只是数全错。这个 bug 犯过一次：
    包络带宽显示成 70787 kN·m，实际是 70.8。
    """
    s, mid = solved
    d = panel(s, mid).series()
    assert d["unit"] == "kN·m"
    kernel = s.query_envelope(component="Mz", member=mid, ).payload
    # 当前工况的峰值不会超过跨所有组合的包络峰值
    assert float(np.abs(d["y"]).max()) <= kernel["peak"] + 1e-6


def test_the_panel_envelope_matches_the_kernel_envelope(qt_app, solved):
    """**两条路算出不同的数，用户信哪个都是错的。**

    面板自己逐点取各组合上下界，内核 `query_envelope` 另有一套实现。
    两者必须给出同一个峰值。
    """
    s, mid = solved
    d = panel(s, mid, envelope=True).series()
    mine = float(max(np.abs(d["upper"]).max(), np.abs(d["lower"]).max()))
    kernel = s.query_envelope(component="Mz", member=mid).payload["peak"]
    assert mine == pytest.approx(kernel, rel=1e-6), (mine, kernel)


def test_mm_model_diagram_x_axis_is_still_reported_in_physical_metres(qt_app, solved):
    original, mid = solved
    session = Session()
    assert session.set_model(deepcopy(original.model)).ok
    assert session.set_units("N-mm-MPa").ok
    assert session.solve_model().ok
    data = panel(session, mid).series()
    expected = session.query_diagram(component="Mz", member=mid).payload["length_m"]
    assert data["x_unit"] == "m"
    assert data["x"][-1] == pytest.approx(expected)


def test_the_envelope_brackets_every_single_case(qt_app, solved):
    """包络必须真的把每个工况都包住——上界不小于任何一条曲线。"""
    s, mid = solved
    env = panel(s, mid, envelope=True).series()
    for case in s.solution.all_results():
        p = panel(s, mid)
        p.case = case
        one = p.series()
        assert np.all(one["y"] <= env["upper"] + 1e-6), case
        assert np.all(one["y"] >= env["lower"] - 1e-6), case


def test_which_combination_governs_is_reported(qt_app, solved):
    """**这才是包络真正有用的地方**：跨中和支座常常不是同一个组合说了算。
    只给上下界不说是谁控制的，用户还得自己一个个试。"""
    s, mid = solved
    d = panel(s, mid, envelope=True).series()
    names = set(d["governs_max"]) | set(d["governs_min"])
    assert names <= set(s.solution.all_results())
    assert names, "没报出任何控制组合"


@pytest.mark.parametrize("label", list(COMPONENTS))
def test_every_component_can_be_drawn(qt_app, solved, label):
    """六个分量都要能画。轴力和弯矩的单位不同，别只测常用的那个。"""
    s, mid = solved
    p = panel(s, mid)
    p.component.setCurrentText(label)
    d = p.series()
    assert d is not None and len(d["x"]) == len(d["y"])
    expect = "kN·m" if COMPONENTS[label] in ("T", "My", "Mz", "M") else "kN"
    assert d["unit"] == expect


# ------------------------------------------------- 界面行为

def test_no_result_means_no_crash(qt_app):
    """还没求解就打开这块面板是常事，不能炸。"""
    s = Session()
    p = DiagramPanel()
    p.attach(s, None)
    assert p.series() is None
    p.redraw()                                  # 不抛异常即可
    assert "尚无" in p.caption.text()


def test_switching_model_refreshes_the_member_list(qt_app, solved):
    """换模型时杆件下拉必须重填，否则会留着上一个模型的编号。"""
    s, mid = solved
    p = DiagramPanel()
    p.attach(s, "D")
    assert p.member.count() == len(s.frame.members)
    p.attach(Session(), None)
    assert p.member.count() == 0


def test_the_caption_says_where_the_peak_is(qt_app, solved):
    """只报一个峰值数字没用——**得说清楚在哪根杆、哪个位置**。"""
    s, mid = solved
    p = panel(s, mid)
    p.redraw()
    text = p.caption.text()
    assert f"杆件 {mid}" in text and "x =" in text and "kN·m" in text


def test_signed_diagram_states_local_axis_and_right_hand_rule(qt_app, solved):
    """空间梁不能笼统声称弯矩画在受拉侧，必须说明局部轴与正号。"""
    s, mid = solved
    p = panel(s, mid)
    p.redraw()
    assert "local x:" in p.caption.text()
    assert "局部 z 轴" in p.caption.text()
    assert "右手定则" in p.caption.text()
    p.component.setCurrentText("轴力 N（受拉为正）")
    p.redraw()
    assert "受拉" in p.caption.text()
    assert "右手定则" not in p.caption.text()


def test_diagram_features_find_zero_extrema_and_a_jump():
    features = diagram_features(
        [0.0, 1.0, 2.0, 2.0 + 1e-9, 3.0],
        [-2.0, 0.0, 3.0, -1.0, 2.0])
    assert features["max"] == (2.0, 3.0)
    assert features["min"] == (0.0, -2.0)
    assert features["zeros"] == pytest.approx([1.0, 7.0 / 3.0])
    assert features["jumps"] == pytest.approx([2.0 + 0.5e-9])
