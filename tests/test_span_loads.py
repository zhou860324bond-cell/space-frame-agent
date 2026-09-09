"""梯形与集中杆间荷载的验证。

固端力公式一旦写错，结果**看着完全正常**——量级对、图形也像那么回事，
只是数不对。所以这里每一条都对着闭合解或静力恒等式验，没有一条是"跑通即通过"。

分三层：
1. 公式层：等效节点荷载的合力与合力矩必须等于荷载本身的合力与合力矩；
2. 结构层：简支、固接、悬臂三种边界下对经典闭合解；
3. 一致性：均布的两种写法必须给出同一个数，叠加必须等于分别算再相加。
"""

import numpy as np
import pytest

from frame3d import (Frame, Material, Member, Node, Section, check_equilibrium,
                     check_model, fixed_end_equivalent, solve)
from internal_forces import member_diagram
from span_loads import POINT, TRAPEZOID, UNIFORM, SpanLoad, fixed_end, span_force

# 符号约定：这套几何下全局 z 向的荷载落在杆件局部 y 上，因此产生 Mz。
# 向下（W、P 取负）的荷载给出**正的** Mz。既有测试一律取绝对值回避了符号，
# 这里把它写明并当作判据——符号错误正是最容易漏掉的那一类。
E, L = 2.1e11, 6.0
W, P = -20e3, -30e3
SEC = Section("S", 0.01, 4e-5, 3e-4, 8e-7)
MAT = Material("STEEL", E, 0.3)
I3 = np.eye(3)


def bar(n: int = 1, cuts: tuple[float, ...] = ()) -> Frame:
    """一根沿全局 x 的杆，可指定额外的分割位置（用于验网格无关性）。"""
    stations = sorted({0.0, L, *(L * k / n for k in range(n + 1)), *cuts})
    f = Frame()
    for k, s in enumerate(stations, 1):
        f.nodes[k] = Node(k, s, 0.0, 0.0)
    for k in range(1, len(stations)):
        f.members[k] = Member(k, k, k + 1, "S", "STEEL")
    f.sections["S"] = SEC
    f.materials["STEEL"] = MAT
    return f


def pin_pin(f: Frame) -> Frame:
    last = max(f.nodes)
    f.supports[1] = (1, 1, 1, 1, 0, 0)
    f.supports[last] = (0, 1, 1, 1, 0, 0)
    return f


def clamp_clamp(f: Frame) -> Frame:
    f.supports[1] = (1,) * 6
    f.supports[max(f.nodes)] = (1,) * 6
    return f


def peak(f: Frame, component: str = "Mz") -> tuple[float, float]:
    """全结构该分量的极值 (位置, 值)，位置换算到整根杆的坐标。"""
    sol = solve(f)
    best = (0.0, 0.0)
    for mid in f.members:
        d = member_diagram(f, sol, mid, stations=401)
        x0 = f.nodes[f.members[mid].i].x
        pos, value = d.extreme(component)
        if abs(value) > abs(best[1]):
            best = (x0 + pos, value)
    return best


# ------------------------------------------------- 公式层

@pytest.mark.parametrize("w", [(0, 0, W), (3e3, -2e3, 5e3)])
def test_uniform_span_load_matches_the_original_implementation(w):
    """均布走新路径必须与既有 fixed_end_equivalent 逐位相同。

    这是整个改动的安全网：老路径的结果已经对着闭合解和 Abaqus 验过了。
    """
    new = fixed_end(SpanLoad(UNIFORM, w), L, I3)
    assert np.array_equal(new, fixed_end_equivalent(L, I3 @ np.asarray(w, float)))


def test_a_trapezoid_with_equal_ends_is_the_uniform_load():
    w = (0.0, 0.0, W)
    assert np.array_equal(fixed_end(SpanLoad(TRAPEZOID, w, w), L, I3),
                          fixed_end(SpanLoad(UNIFORM, w), L, I3))


def _statics_y(p: np.ndarray) -> tuple[float, float]:
    """等效节点荷载的 (合力 Fy, 对 i 端的合力矩)。"""
    return p[1] + p[7], p[7] * L + p[5] + p[11]


def test_triangular_load_is_statically_equivalent():
    """三角形 0→q：合力 qL/2，对 i 端矩 qL²/3。"""
    force, moment = _statics_y(fixed_end(SpanLoad(TRAPEZOID, (0, 0, 0), (0, W, 0)), L, I3))
    assert force == pytest.approx(W * L / 2)
    assert moment == pytest.approx(W * L ** 2 / 3)


@pytest.mark.parametrize("a", [0.0, 1.5, 3.0, 4.5, L])
def test_a_point_load_is_statically_equivalent(a):
    """任意位置的集中力：合力 P，对 i 端矩 P·a。"""
    force, moment = _statics_y(fixed_end(SpanLoad(POINT, (0, P, 0), a=a), L, I3))
    assert force == pytest.approx(P)
    assert moment == pytest.approx(P * a)


def test_the_z_direction_carries_the_opposite_moment_sign():
    """局部 y 与 z 两个方向的弯矩项符号相反。写成同号是最常见的错法，
    表现是悬臂自由端弯矩不归零。"""
    py = fixed_end(SpanLoad(POINT, (0, P, 0), a=2.0), L, I3)
    pz = fixed_end(SpanLoad(POINT, (0, 0, P), a=2.0), L, I3)
    assert pz[4] == pytest.approx(-py[5])
    assert pz[10] == pytest.approx(-py[11])


def test_span_force_reduces_to_the_uniform_formulas():
    """均布下 S = w·x、M = w·x²/2 —— 原来那套公式必须是新公式的特例。"""
    x = np.linspace(0, L, 7)
    S, M = span_force(SpanLoad(UNIFORM, (0, 0, W)), L, I3, x)
    assert S[2] == pytest.approx(W * x)
    assert M[2] == pytest.approx(W * x ** 2 / 2)


def test_span_force_of_a_point_load_is_a_step():
    x = np.array([0.0, 1.0, 1.999, 2.001, 4.0, L])
    S, M = span_force(SpanLoad(POINT, (0, P, 0), a=2.0), L, I3, x)
    assert S[1] == pytest.approx([0, 0, 0, P, P, P])
    assert M[1] == pytest.approx([0, 0, 0, P * 0.001, P * 2.0, P * 4.0], abs=1e-9)


# ------------------------------------------------- 结构层

@pytest.mark.gold
def test_simply_supported_point_load_at_midspan():
    """M_max = PL/4，出现在跨中。"""
    f = pin_pin(bar(1))
    f.member_spans[1] = [SpanLoad(POINT, (0.0, 0.0, P), a=L / 2)]
    x, moment = peak(f)
    assert moment == pytest.approx(-P * L / 4, rel=1e-9)
    assert x == pytest.approx(L / 2, abs=L / 400)


@pytest.mark.parametrize("a", [1.0, 2.0, 4.5])
def test_simply_supported_point_load_off_centre(a):
    """M = P·a·b/L，出现在荷载作用点。偏心位置才验得出 a、b 有没有写反。"""
    f = pin_pin(bar(1))
    f.member_spans[1] = [SpanLoad(POINT, (0.0, 0.0, P), a=a)]
    x, moment = peak(f)
    assert moment == pytest.approx(-P * a * (L - a) / L, rel=1e-9)
    assert x == pytest.approx(a, abs=L / 400)


def test_fixed_fixed_point_load_at_midspan():
    """两端固接、跨中集中力：端弯矩与跨中弯矩都是 PL/8，符号相反。"""
    f = clamp_clamp(bar(1))
    f.member_spans[1] = [SpanLoad(POINT, (0.0, 0.0, P), a=L / 2)]
    d = member_diagram(f, solve(f), 1, stations=401)
    assert d.Mz[0] == pytest.approx(P * L / 8, rel=1e-9)
    assert d.Mz[-1] == pytest.approx(P * L / 8, rel=1e-9)
    mid = int(np.argmin(np.abs(d.x - L / 2)))
    assert d.Mz[mid] == pytest.approx(-P * L / 8, rel=1e-6)


def test_cantilever_point_load_leaves_the_far_end_free():
    """固端弯矩 P·a，而荷载点以外的一段弯矩恒为零 —— 后半条最能验出符号错误。"""
    a = 4.0
    f = bar(1)
    f.supports[1] = (1,) * 6              # 只固定一端，另一端自由
    f.member_spans[1] = [SpanLoad(POINT, (0.0, 0.0, P), a=a)]
    d = member_diagram(f, solve(f), 1, stations=401)
    assert d.Mz[0] == pytest.approx(P * a, rel=1e-9)
    beyond = d.Mz[d.x > a + 1e-6]
    assert np.abs(beyond).max() < abs(P * a) * 1e-9


def test_fixed_fixed_triangular_load_end_moments():
    """两端固接、三角形荷载：|M_i| = wL²/30，|M_j| = wL²/20。

    两个数不同，正好验出 30 和 20 有没有写反 —— 均布荷载两端等大，验不出来。

    注意两端**同号**：教科书里的固端弯矩 wL²/30 与 −wL²/20 是按杆端转向写的，
    而这里画的是截面弯矩，两端都是负弯矩（上部受拉），同号才对。
    跨中集中力那条测试也是同样的形态。
    """
    f = clamp_clamp(bar(1))
    f.member_spans[1] = [SpanLoad(TRAPEZOID, (0.0, 0.0, 0.0), (0.0, 0.0, W))]
    d = member_diagram(f, solve(f), 1, stations=401)
    assert d.Mz[0] == pytest.approx(W * L ** 2 / 30, rel=1e-9)
    assert d.Mz[-1] == pytest.approx(W * L ** 2 / 20, rel=1e-9)
    assert np.sign(d.Mz[0]) == np.sign(d.Mz[-1])


@pytest.mark.gold
def test_simply_supported_triangular_load_peak():
    """M_max = wL²/(9√3)，位置 x = L/√3。位置本身就是一条独立判据。"""
    f = pin_pin(bar(1))
    f.member_spans[1] = [SpanLoad(TRAPEZOID, (0.0, 0.0, 0.0), (0.0, 0.0, W))]
    x, moment = peak(f)
    assert moment == pytest.approx(-W * L ** 2 / (9 * np.sqrt(3)), rel=1e-5)
    assert x == pytest.approx(L / np.sqrt(3), abs=L / 200)


def test_the_shear_jump_across_a_point_load_equals_the_load():
    f = pin_pin(bar(1))
    f.member_spans[1] = [SpanLoad(POINT, (0.0, 0.0, P), a=2.5)]
    d = member_diagram(f, solve(f), 1, stations=51)
    before = d.Vy[d.x < 2.5][-1]
    after = d.Vy[d.x > 2.5][0]
    assert abs(after - before) == pytest.approx(abs(P), rel=1e-6)


@pytest.mark.parametrize("cuts", [(), (2.0,), (2.0, 4.5), (1.0, 2.0, 3.0, 4.0, 5.0)])
def test_the_point_load_answer_is_independent_of_the_mesh(cuts):
    """把杆件切成几段，跨中弯矩必须一模一样 —— 解析恢复的意义就在这里。"""
    a = 2.0
    f = pin_pin(bar(1, cuts))
    host = next(mid for mid, m in f.members.items()
                if f.nodes[m.i].x <= a <= f.nodes[m.j].x)
    f.member_spans[host] = [SpanLoad(POINT, (0.0, 0.0, P),
                                     a=a - f.nodes[f.members[host].i].x)]
    _, moment = peak(f)
    assert moment == pytest.approx(-P * a * (L - a) / L, rel=1e-9)


# ------------------------------------------------- 一致性

def test_the_two_ways_of_writing_a_uniform_load_agree():
    """member_loads 与 member_spans 两种写法必须给出同一个数。"""
    a = pin_pin(bar(1))
    a.member_loads[1] = (0.0, 0.0, W)
    b = pin_pin(bar(1))
    b.member_spans[1] = [SpanLoad(UNIFORM, (0.0, 0.0, W))]
    assert (member_diagram(a, solve(a), 1).Mz
            == pytest.approx(member_diagram(b, solve(b), 1).Mz))


def test_loads_on_one_member_superpose():
    """同一根杆上多项荷载叠加，必须等于分别算再相加 —— 线性问题里这是精确的。"""
    items = [SpanLoad(UNIFORM, (0.0, 0.0, W)),
             SpanLoad(POINT, (0.0, 0.0, P), a=1.5),
             SpanLoad(TRAPEZOID, (0.0, 0.0, 0.0), (0.0, 0.0, 0.5 * W))]
    grid = np.linspace(0.0, L, 41)

    def sampled(frame):
        # 有集中力时采样点会多出跳跃两侧的点，插到同一网格上才好比。
        # 弯矩在集中力处是连续的，插值不会引入误差
        d = member_diagram(frame, solve(frame), 1, stations=41)
        return np.interp(grid, d.x, d.Mz)

    together = pin_pin(bar(1))
    together.member_spans[1] = list(items)
    total = sampled(together)

    parts = np.zeros_like(grid)
    for item in items:
        one = pin_pin(bar(1))
        one.member_spans[1] = [item]
        parts = parts + sampled(one)
    assert total == pytest.approx(parts, rel=1e-9)


@pytest.mark.parametrize("item", [
    SpanLoad(POINT, (0.0, 0.0, P), a=2.0),
    SpanLoad(TRAPEZOID, (0.0, 0.0, 0.0), (0.0, 0.0, W)),
    SpanLoad(TRAPEZOID, (1e3, 0.0, W), (-1e3, 0.0, 0.5 * W)),
])
def test_global_equilibrium_holds_for_every_load_kind(item):
    """支座反力必须与外荷载合力、合力矩都平衡。

    只验合力不够 —— 荷载作用位置写错时合力照样对得上，合力矩才会露馅。
    """
    f = clamp_clamp(bar(1))
    f.member_spans[1] = [item]
    report = check_equilibrium(f, solve(f))
    assert report["ok"], report


def test_a_point_load_beyond_the_member_is_refused():
    """a 超出杆长时公式里 b = L − a 会变号，算出来的数看着正常其实没有意义。"""
    f = pin_pin(bar(1))
    f.member_spans[1] = [SpanLoad(POINT, (0.0, 0.0, P), a=L + 1.0)]
    issues = check_model(f)
    assert any("超出杆长" in m for m in issues), issues


def test_a_load_on_a_missing_member_is_refused():
    f = pin_pin(bar(1))
    f.member_spans[99] = [SpanLoad(POINT, (0.0, 0.0, P), a=1.0)]
    assert any("不存在的杆件" in m for m in check_model(f))


def test_an_unknown_load_kind_is_refused_at_construction():
    with pytest.raises(ValueError, match="杆间荷载类型"):
        SpanLoad("triangle", (0.0, 0.0, W))


def test_combos_scale_span_loads():
    """组合按系数合成，梯形与集中力也要跟着缩放。"""
    f = pin_pin(bar(1))
    f.case("DL").member_spans[1] = [SpanLoad(POINT, (0.0, 0.0, P), a=2.0)]
    f.load_cases.pop("default", None)
    f.combos["1.3DL"] = {"DL": 1.3}
    sol = solve(f)
    base = member_diagram(f, sol, 1, case="DL", stations=41).Mz
    combo = member_diagram(f, sol, 1, case="1.3DL", stations=41).Mz
    assert combo == pytest.approx(1.3 * base, rel=1e-9)


def test_round_trip_through_a_dict():
    for item in (SpanLoad(UNIFORM, (0.0, 0.0, W)),
                 SpanLoad(TRAPEZOID, (0.0, 0.0, W), (0.0, 0.0, 2 * W)),
                 SpanLoad(POINT, (0.0, 0.0, P), a=2.5)):
        assert SpanLoad.from_dict(item.to_dict()) == item
