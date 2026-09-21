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
from model_io import from_dict
from span_loads import (KINDS, PARTIAL, POINT, TRAPEZOID, UNIFORM, SpanLoad,
                        fixed_end, span_force)

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


# --------------------------------------------------- 部分跨均布 partial

def test_a_partial_load_over_the_whole_span_is_the_uniform_load():
    """a=0、b=L 时必须**逐项**退回满跨均布。

    这是 partial 最强的一条自检：它的固端力是把集中力公式从 a 积到 b 得来的，
    积满全跨就该还原成 wL/2 与 wL²/12。差一点都说明积分写错了。
    """
    rot = np.eye(3)
    full = fixed_end(SpanLoad(UNIFORM, (0.0, 0.0, W)), L, rot)
    part = fixed_end(SpanLoad(PARTIAL, (0.0, 0.0, W), a=0.0, b=L), L, rot)
    assert np.allclose(full, part, atol=1e-9), float(np.max(np.abs(full - part)))


def test_a_half_span_load_matches_the_textbook_fixed_end_forces():
    """半跨均布（0→L/2）的固端力有现成闭合解，四个数逐个对。

        V_i = 13wL/32    V_j = 3wL/32
        M_i = 11wL²/192  M_j = 5wL²/192

    这四个彼此独立、凑不出来——任一系数写错都会当场露馅。
    """
    p = fixed_end(SpanLoad(PARTIAL, (0.0, 0.0, W), a=0.0, b=L / 2), L, np.eye(3))
    vi, vj, mi, mj = p[2], p[8], p[4], p[10]
    assert vi == pytest.approx(13.0 * W * L / 32.0, rel=1e-12)
    assert vj == pytest.approx(3.0 * W * L / 32.0, rel=1e-12)
    # 局部 z 方向的弯矩项取反，与 uniform / trapezoid 是同一套符号来源
    assert mi == pytest.approx(-11.0 * W * L ** 2 / 192.0, rel=1e-12)
    assert mj == pytest.approx(5.0 * W * L ** 2 / 192.0, rel=1e-12)
    assert vi + vj == pytest.approx(W * L / 2.0, rel=1e-12)


def test_a_partial_load_reproduces_the_hand_calculation():
    """端到端：简支梁前半跨受载，跨内弯矩对手算。

    6 m 梁、前 3 m 上 20 kN/m：合力 60 kN 作用在 x=1.5，
    R₁ = 60×(1−1.5/6) = 45 kN，峰值出现在剪力过零处 x = R₁/w = 2.25 m，
    M = 45×2.25 − 20×2.25²/2 = 50.625 kN·m。
    """
    f = pin_pin(bar())
    f.case().member_spans[1] = [SpanLoad(PARTIAL, (0.0, 0.0, W), a=0.0, b=L / 2)]
    sol = solve(f)
    assert check_equilibrium(f, sol, "default")["ok"]
    r1 = -W * (L / 2) * (1.0 - (L / 4) / L)          # 近端反力（向上为正）
    x0 = r1 / -W                                      # 剪力过零
    exact = r1 * x0 + W * x0 ** 2 / 2.0
    x_peak, m_peak = peak(f)
    assert x_peak == pytest.approx(x0, abs=L / 200)
    assert m_peak == pytest.approx(exact, rel=1e-3)


def test_splitting_the_member_does_not_change_a_partial_load():
    """跨越剖分点时，两段合起来必须和不剖分给出同一个答案。

    重叠区间要落到每个单元**自己的**局部坐标上。算错只会让结果偏，
    不报错——所以拿"剖分与否同值"当判据。
    """
    plain = pin_pin(bar())
    plain.case().member_spans[1] = [
        SpanLoad(PARTIAL, (0.0, 0.0, W), a=L / 4, b=3 * L / 4)]
    cut = pin_pin(bar(n=2))                  # 在跨中切开
    cut.case().member_spans[1] = [
        SpanLoad(PARTIAL, (0.0, 0.0, W), a=L / 4, b=L / 2)]
    cut.case().member_spans[2] = [
        SpanLoad(PARTIAL, (0.0, 0.0, W), a=0.0, b=L / 4)]
    assert peak(plain)[1] == pytest.approx(peak(cut)[1], rel=1e-6)
    assert peak(plain)[0] == pytest.approx(peak(cut)[0], abs=L / 100)


@pytest.mark.parametrize("kind", KINDS)
def test_every_declared_span_load_kind_reaches_the_solver(kind):
    """**KINDS 里每一种都必须真的进得了求解器。**

    这条闸拦的是"加了类型却没改编译器"。model_compiler 里那段原本是个
    没有 else 的 if 链：POINT/UNIFORM/TRAPEZOID 各自 continue，其余一概
    掉出循环消失。加 partial 时实测后果是**反力全零、平衡残差 0.0、
    一句话都不报**——荷载凭空蒸发，而所有自检都说"没问题"。

    判据是反力合计等于荷载合力：荷载被丢掉时它会是零。
    """
    from agent import Session

    payload = {
        "units": "N-m-Pa",
        "materials": [{"name": "M", "E": E, "nu": 0.3}],
        "sections": [{"name": "S", "A": 0.01, "Iy": 4e-5, "Iz": 3e-4,
                      "J": 8e-7}],
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                  {"id": 2, "x": L, "y": 0, "z": 0}],
        "members": [{"id": 1, "i": 1, "j": 2, "material": "M", "section": "S"}],
        "supports": [{"node": 1, "fix": [1, 1, 1, 1, 1, 1]},
                     {"node": 2, "fix": [0, 1, 1, 1, 1, 1]}],
    }
    entry: dict = {"member": 1, "kind": kind, "w1": [0.0, 0.0, W]}
    if kind == TRAPEZOID:
        entry["w2"] = [0.0, 0.0, W]
    if kind == POINT:
        entry["a"] = L / 2
    if kind == PARTIAL:
        entry["a"], entry["b"] = 0.0, L / 2

    s = Session()
    assert s.set_model(payload).ok
    s.model["load_cases"] = [{"name": "C", "member_spans": [entry]}]
    assert s.solve_model().ok
    total = s.query_results("reactions", case="C").payload["vertical_total_kN"]
    assert abs(total) > 1e-6, (
        f"{kind} 类型的荷载没有产生任何反力——多半是被编译器静默丢掉了")


# ------------------------------------------------------- 弹性支座

def _spring_cantilever(k: float) -> Frame:
    """悬臂，端部加一个竖向弹簧。梁与弹簧**并联**。

    注意放开的是 ry 不是 rz：杆沿全局 X 时局部 y 指向全局 +Z，
    所以 Z 向荷载引起的转动是绕局部 z，对应全局 Y。
    """
    f = bar()
    f.supports[1] = (1, 1, 1, 1, 1, 1)
    f.supports[2] = (1, 1, 0, 1, 0, 1)
    if k:
        f.springs[2] = (0.0, 0.0, k, 0.0, 0.0, 0.0)
    f.case().nodal_loads[2] = (0.0, 0.0, P, 0.0, 0.0, 0.0)
    return f


@pytest.mark.parametrize("ratio", [0.0, 1.0, 10.0, 1e6])
def test_a_spring_support_acts_in_parallel_with_the_beam(ratio):
    """δ = P/(k + 3EI/L³)。弹簧与梁端刚度是并联关系。

    ratio=0 退回纯悬臂 PL³/(3EI)，ratio 很大时位移趋于零——两端都对上，
    中间那两档才有意义。四档实测误差 0.0000%。
    """
    beam_k = 3.0 * E * SEC.Iz / L ** 3
    k = ratio * beam_k
    f = _spring_cantilever(k)
    sol = solve(f)
    uz = sol["default"].U[f.node_dofs(2)[2]]
    assert uz == pytest.approx(P / (k + beam_k), rel=1e-9)


def test_the_spring_reaction_is_included_in_equilibrium():
    """弹簧支承的自由度是**自由**的，K@U−F 在那里恒为零。

    所以弹簧反力必须单独补 −k·u。漏掉的后果不是数字偏小，
    而是整体平衡直接失衡——这条拿 check_equilibrium 当判据。
    """
    beam_k = 3.0 * E * SEC.Iz / L ** 3
    f = _spring_cantilever(2.0 * beam_k)
    sol = solve(f)
    assert check_equilibrium(f, sol, "default")["ok"]
    total = sol["default"].R[[f.node_dofs(n)[2] for n in f.nodes]].sum()
    assert total == pytest.approx(-P, rel=1e-9)


def test_a_rigid_fix_and_a_spring_on_the_same_dof_is_rejected():
    """同一方向既 fix=1 又给弹簧刚度，必须报错而不是二选一。

    刚性约束会把自由度整个划掉，弹簧那一项永远用不上。静默忽略最糟：
    用户以为建了个弹性支座，算出来的却是刚接，而两者的内力分布完全不同。
    """
    from model_io import validate_payload

    payload = {
        "units": "N-m-Pa",
        "materials": [{"name": "M", "E": E, "nu": 0.3}],
        "sections": [{"name": "S", "A": 0.01, "Iy": 4e-5, "Iz": 3e-4,
                      "J": 8e-7}],
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                  {"id": 2, "x": L, "y": 0, "z": 0}],
        "members": [{"id": 1, "i": 1, "j": 2, "material": "M", "section": "S"}],
        "supports": [{"node": 1, "fix": [1, 1, 1, 1, 1, 1],
                      "spring": [0, 0, 1e7, 0, 0, 0]}],
    }
    errors = validate_payload(payload)
    assert any("既写了 fix=1 又给了" in e for e in errors), errors


def test_spring_stiffness_survives_a_unit_change():
    """平动弹簧是 力/长度、转动弹簧是 力·长度/弧度，**换算方向相反**。

    用同一个系数会让转动弹簧差 10⁶，而且只在换过单位的模型上才现形——
    和 cy/cz、静矩踩过的是同一个坑。判据是换算前后算出来的位移相同。
    """
    from units import MM, convert_model

    beam_k = 3.0 * E * SEC.Iz / L ** 3
    si = {
        "units": "N-m-Pa",
        "materials": [{"name": "M", "E": E, "nu": 0.3}],
        "sections": [{"name": "S", "A": 0.01, "Iy": 4e-5, "Iz": SEC.Iz,
                      "J": 8e-7}],
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                  {"id": 2, "x": L, "y": 0, "z": 0}],
        "members": [{"id": 1, "i": 1, "j": 2, "material": "M", "section": "S"}],
        "supports": [{"node": 1, "fix": [1, 1, 1, 1, 1, 1]},
                     {"node": 2, "fix": [1, 1, 0, 1, 0, 1],
                      "spring": [0, 0, beam_k, 0, 0, 1.0e6]}],
        "nodal_loads": [{"node": 2, "load": [0, 0, P, 0, 0, 0]}],
    }
    mm = convert_model(si, MM)
    a = solve(from_dict(si))["default"]
    b = solve(from_dict(mm))["default"]
    # 位移以 m 与 mm 计，差 1000 倍；物理量相同
    assert a.U[from_dict(si).node_dofs(2)[2]] * 1e3 == pytest.approx(
        b.U[from_dict(mm).node_dofs(2)[2]], rel=1e-9)
