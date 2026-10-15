"""初始几何缺陷 + 二阶分析。

防的问题：线性屈曲只给完善结构的上限，钢结构稳定承载力要按「有缺陷 + 二阶」算；
原来没有任何施加缺陷的入口，P-Δ 只能在完善几何上做，压杆在达到分岔荷载之前
没有侧向变形，看不出稳定问题。

闭合解：两端铰支压杆，初始弯曲取一阶屈曲模态（正弦）、跨中幅值 e₀，轴压
P = α·Pcr 下跨中**附加**挠度 = e₀·α/(1−α)。α = 0.5 时恰好等于 e₀。
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from agent import Session
from units import convert_model

E, NU = 2.0e11, 0.3
A, IY, IZ, J = 0.01, 1.0e-5, 2.0e-5, 5.0e-6
LENGTH = 8.0


def _column(load: float) -> dict:
    return {
        "units": "N-m-Pa",
        "materials": [{"name": "Steel", "E": E, "nu": NU}],
        "sections": [{"name": "S", "A": A, "Iy": IY, "Iz": IZ, "J": J}],
        "nodes": [{"id": 1, "x": 0.0, "y": 0.0, "z": 0.0},
                  {"id": 2, "x": LENGTH, "y": 0.0, "z": 0.0}],
        "members": [{"id": 1, "i": 1, "j": 2, "section": "S", "material": "Steel"}],
        "supports": [{"node": 1, "fix": [1, 1, 1, 1, 0, 0]},
                     {"node": 2, "fix": [0, 1, 1, 0, 0, 0]}],
        "nodal_loads": [{"node": 2, "load": [-load, 0, 0, 0, 0, 0]}],
    }


def _critical_load() -> float:
    """加密模型上的数值 Pcr：比对附加挠度时用同一个离散，误差只剩二阶迭代本身。"""
    reference = 1.0e4
    got = Session(model=_column(reference)).buckling_analysis(num_modes=1)
    assert got.ok, got.payload
    return got.payload["critical_factor"] * reference


def _midspan_translation(session: Session) -> np.ndarray:
    mapping = session.compilation.mapping
    node = next(n for n, (member, ratio) in mapping.generated_nodes.items()
                if member == 1 and abs(ratio - 0.5) < 1e-9)
    k = session.frame.index_of()[node]
    return session.solution.cases[session.solution.primary].U[6 * k:6 * k + 3]


@pytest.mark.parametrize("divisions, rel", [
    # 不布种：默认 8 段，实测比值 0.987
    (None, 1.5e-2),
    # 布种 16 段，实测 0.997；O(h²) 收敛，见 imperfection.MIN_DIVISIONS
    (16, 5e-3),
])
def test_sinusoidal_imperfection_is_amplified_by_one_over_one_minus_alpha(divisions, rel):
    pcr = _critical_load()
    e0, alpha = LENGTH / 1000.0, 0.5
    session = Session(model=_column(alpha * pcr))
    if divisions:
        assert session.seed_mesh(divisions=divisions).ok
    assert session.set_imperfection(mode=1, amplitude=e0).ok
    got = session.solve_model(analysis="pdelta", increments=20, tolerance=1e-10)
    assert got.ok, got.payload

    lateral = np.linalg.norm(_midspan_translation(session)[1:])
    exact = e0 * alpha / (1 - alpha)
    assert lateral == pytest.approx(exact, rel=rel)
    assert lateral < exact, "折线逼近从下方收敛：粗网格偏小，所以默认段数不能太少"
    info = got.payload["imperfection"]
    assert info["mode"] == 1 and info["amplitude"] == pytest.approx(e0)
    assert "杆件 1" in info["peak_at"], "缺陷峰值在跨中，应报物理杆件位置"


def test_without_imperfection_a_perfect_column_has_no_lateral_response():
    """对照组：同样的轴压，完善几何下 P-Δ 没有侧移——这正是缺陷要补上的东西。"""
    session = Session(model=_column(0.5 * _critical_load()))
    assert session.solve_model(analysis="pdelta").ok
    assert np.linalg.norm(_midspan_translation(session)[1:]) < 1e-12


def test_negative_amplitude_flips_the_deflection():
    pcr = _critical_load()
    up, down = Session(model=_column(0.5 * pcr)), Session(model=_column(0.5 * pcr))
    assert up.set_imperfection(amplitude=0.008).ok
    assert down.set_imperfection(amplitude=-0.008).ok
    assert up.solve_model(analysis="pdelta").ok and down.solve_model(analysis="pdelta").ok
    # 只比侧向：轴向缩短与缺陷方向无关，不翻转
    np.testing.assert_allclose(_midspan_translation(up)[1:],
                               -_midspan_translation(down)[1:], rtol=1e-6, atol=1e-12)


def test_linear_analysis_ignores_the_imperfection_and_says_so():
    session = Session(model=_column(1.0e4))
    assert session.set_imperfection(amplitude=0.008).ok
    got = session.solve_model()
    assert got.ok
    assert "imperfection" not in got.payload
    assert "只在 pdelta" in got.payload["imperfection_note"]


def test_imperfection_amplitude_survives_the_unit_conversion():
    """幅值是长度：毫米制下不换算就小了 1000 倍，缺陷形同没有。"""
    pcr = _critical_load()
    si = _column(0.5 * pcr)
    si["imperfection"] = {"mode": 1, "amplitude": 0.008}
    mm = convert_model(deepcopy(si), "N-mm-MPa")
    a, b = Session(model=si), Session(model=mm)
    assert a.solve_model(analysis="pdelta").ok and b.solve_model(analysis="pdelta").ok
    np.testing.assert_allclose(np.linalg.norm(_midspan_translation(b)) / 1e3,
                               np.linalg.norm(_midspan_translation(a)), rtol=1e-6)


def test_set_imperfection_validates_its_inputs():
    session = Session(model=_column(1.0e4))
    assert not session.set_imperfection().ok, "幅值两个都没给"
    assert not session.set_imperfection(amplitude=0.01, height_ratio=0.004).ok
    assert not session.set_imperfection(amplitude=0.01, case="不存在").ok
    assert "imperfection" not in session.model

    # 平面结构 z 向没有跨度时，按最大尺寸取「高度」
    got = session.set_imperfection(height_ratio=1 / 250)
    assert got.ok
    assert session.model["imperfection"]["amplitude"] == pytest.approx(LENGTH / 250)
    assert session.set_imperfection(clear=True).payload["cleared"]
    assert "imperfection" not in session.model
