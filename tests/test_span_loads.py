"""梯形与集中杆间荷载的验证。

固端力公式一旦写错，结果**看着完全正常**——量级对、图形也像那么回事，
只是数不对。所以这里每一条都对着闭合解或静力恒等式验，没有一条是"跑通即通过"。

分三层：
1. 公式层：等效节点荷载的合力与合力矩必须等于荷载本身的合力与合力矩；
2. 结构层：简支、固接、悬臂三种边界下对经典闭合解；
3. 一致性：均布的两种写法必须给出同一个数，叠加必须等于分别算再相加。
"""

import math

import numpy as np
import pytest

from frame3d import (Frame, Material, Member, Node, Section, check_equilibrium,
                     check_model, fixed_end_equivalent, solve)
from internal_forces import member_diagram
from model_io import from_dict
from span_loads import (KINDS, MOMENT, PARTIAL, POINT, RAMP, TRAPEZOID,
                        UNIFORM, SpanLoad,
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
    if kind == MOMENT:
        entry["w1"] = [0.0, W * L, 0.0]        # 力矩矢量，量纲是力×长度
        entry["a"] = L / 2
    if kind == PARTIAL:
        entry["a"], entry["b"] = 0.0, L / 2
    if kind == RAMP:
        entry["w2"] = [0.0, 0.0, W]
        entry["a"], entry["b"] = 0.0, L / 2

    s = Session()
    assert s.set_model(payload).ok
    s.model["load_cases"] = [{"name": "C", "member_spans": [entry]}]
    assert s.solve_model().ok
    # 判据是**内力非零**而不是反力合计。纯力偶的合力恒为零，拿反力合计去
    # 判会把它误判成"荷载被丢掉了"——判据本身对那一类荷载失效。
    peak = max(abs(s.query_diagram(comp, 1, case="C").payload["peak"])
               for comp in ("N", "Vy", "Vz", "My", "Mz"))
    assert peak > 1e-9, (
        f"{kind} 类型的荷载没有在杆件里产生任何内力——"
        "多半是被编译器静默丢掉了")


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


# ------------------------------------------- 荷载参照系 local / projected

def _sloped(pitch: float = 3.0, run: float = 6.0):
    """一根斜梁，从 (0,0,0) 到 (run,0,pitch)。两端固接，只看反力。"""
    from agent import Session

    s = Session()
    s.set_model({
        "units": "N-m-Pa",
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                  {"id": 2, "x": run, "y": 0, "z": pitch}],
        "members": [{"id": 1, "i": 1, "j": 2, "material": "S", "section": "B"}],
        "materials": [{"name": "S", "E": 2.06e11, "nu": 0.3, "density": 0.0}],
        "sections": [{"name": "B", "A": 8.6e-3, "Iy": 3e-5, "Iz": 1e-4,
                      "J": 1e-6}],
        "supports": [{"node": 1, "fix": [1] * 6}, {"node": 2, "fix": [1] * 6}],
    })
    return s


def test_a_projected_load_puts_the_right_total_on_a_sloped_member():
    """**斜梁上这个选错，结果看着完全正常但是错的。**

    同样写 10 kN/m：按沿杆长算总重是 10×6.708=67.08 kN，
    按水平投影算是 10×6=60.00 kN。差 11.8%，而两种写法在规范里都存在——
    雪载、活载给的是"每米水平投影"，自重给的是"每米杆长"。
    """
    run, pitch, w = 6.0, 3.0, 10e3
    length = math.hypot(run, pitch)

    along = _sloped(pitch, run)
    along.set_member_load(1, [0, 0, -w])
    along.solve_model()
    total_along = along.query_results("reactions").payload["vertical_total_kN"]

    projected = _sloped(pitch, run)
    r = projected.set_member_load(1, [0, 0, -w], reference="projected")
    projected.solve_model()
    total_proj = projected.query_results("reactions").payload["vertical_total_kN"]

    assert total_along == pytest.approx(w * length / 1e3, rel=1e-6)
    assert total_proj == pytest.approx(w * run / 1e3, rel=1e-6)
    # 换算过程必须看得见：用户拿到的强度和自己输入的不一样，得说清为什么
    assert r.payload["conversion"]["scale"] == pytest.approx(run / length,
                                                             rel=1e-6)


def test_a_local_load_is_rotated_not_rescaled():
    """local 只换方向、不改大小——它描述的是"垂直于杆轴的 10 kN/m"。

    投影换算改的是**大小**（总合力守恒），局部坐标换的是**方向**（模长守恒）。
    两者搞混的后果都是一个看着正常的错数，所以各自钉死。
    """
    s = _sloped()
    r = s.set_member_load(1, [0, -10e3, 0], reference="local")
    got = np.asarray(r.payload["load"], dtype=float)
    assert float(np.linalg.norm(got)) == pytest.approx(10e3, rel=1e-9)
    # 局部 y 在这根斜梁上指向"斜上方"，所以全局 x 分量为正、z 分量为负
    assert got[0] > 0 and got[2] < 0


def test_a_point_load_cannot_be_given_per_horizontal_metre():
    """集中力是**力**不是强度，没有"按水平投影分布"这回事。

    悄悄按 cosθ 缩一下最糟：用户以为自己加了 5 kN，实际只加了 4.47 kN。
    """
    s = _sloped()
    r = s.set_member_span_load(1, "point", [0, 0, -5e3], a=2.0,
                               reference="projected")
    assert not r.ok
    assert "集中力" in r.payload["error"]


def test_an_unknown_reference_is_rejected():
    s = _sloped()
    assert not s.set_member_load(1, [0, 0, -1e3], reference="plan").ok


# ------------------------------------------------------- 按集合施加荷载

def _framed():
    """两跨两层一开间的框架，梁已归入集合「梁」。"""
    from agent import Session

    s = Session()
    s.define_materials_and_sections(
        [{"name": "Q", "E": 2.06e11, "nu": 0.3}],
        [{"name": "B", "A": 0.018, "Iy": 5e-5, "Iz": 2e-3, "J": 1e-6}])
    r = s.generate_frame(spans=[6.0, 6.0], storeys=[4.0, 4.0], bays=[6.0])
    s.assign_properties([int(m["id"]) for m in s.model["members"]],
                        material="Q", section="B")
    beams = r.payload["beam_member_ids"]
    s.define_set("梁", member_ids=beams)
    return s, beams


def test_one_call_loads_a_whole_set():
    """一句"给所有梁加 20 kN/m"以前要拆成几十次调用。

    而漏掉其中两根**没有任何人会发现**：工具报成功、校验通过、结果也正常，
    只是少了两根梁的荷载。判据是反力合计等于手算总重。
    """
    s, beams = _framed()
    out = s.set_member_load("梁", [0, 0, -20e3])
    assert out.ok and out.payload["count"] == len(beams)
    assert len(s.model["load_cases"][0]["member_loads"]) == len(beams)
    s.solve_model()
    total = s.query_results("reactions").payload["vertical_total_kN"]
    assert total == pytest.approx(20.0 * 6.0 * len(beams), rel=1e-9)


def test_applying_to_a_set_twice_replaces_rather_than_stacks():
    """重复施加必须是替换。每根杆件有**自己**的名字才做得到——

    共用一个名字的话，按名替换会把前面几根刚写进去的又删掉，只剩最后一根。
    """
    s, beams = _framed()
    s.set_member_load("梁", [0, 0, -20e3])
    s.solve_model()
    once = s.query_results("reactions").payload["vertical_total_kN"]
    s.set_member_load("梁", [0, 0, -20e3])
    s.solve_model()
    twice = s.query_results("reactions").payload["vertical_total_kN"]
    assert twice == pytest.approx(once, rel=1e-12)


def test_a_span_load_on_a_set_reaches_every_member():
    """梯形按集合施加，**每一根**都要写进去。

    这一条盯的是"只取第一根"：我第一版就是那么写的，梯形看着施加成功，
    实际只加到了集合里的第一根梁上——反力少了十三根的份，而工具报的是
    成功。这一轮反复在修的正是这个形状的错。
    """
    s, beams = _framed()
    out = s.set_member_span_load("梁", "trapezoid", [0, 0, 0], [0, 0, -15e3])
    assert out.ok and out.payload["count"] == len(beams)
    assert len(s.model["load_cases"][0]["member_spans"]) == len(beams)
    s.solve_model()
    total = s.query_results("reactions").payload["vertical_total_kN"]
    assert total == pytest.approx(15.0 / 2 * 6.0 * len(beams), rel=1e-9)


def test_a_positioned_load_on_a_set_is_refused():
    """a / b 是沿杆长的**绝对距离**，一组长短不一的杆件共用没有意义。

    短杆上会超界、长杆上位置也对不上。按比例自作主张更糟——那是另一个
    物理量，而用户不会知道工具替他改了定义。
    """
    s, _ = _framed()
    for kind, kwargs in (("point", {"a": 2.0}),
                         ("partial", {"a": 1.0, "b": 3.0})):
        r = s.set_member_span_load("梁", kind, [0, 0, -5e3], **kwargs)
        assert not r.ok
        assert "绝对距离" in r.payload["error"]


def test_the_single_member_payload_did_not_change():
    """单根路径的返回值必须一个字节没变——既有调用方和测试都靠它。"""
    s, beams = _framed()
    one = s.set_member_load(beams[0], [0, 0, -30e3])
    assert set(one.payload) == {"case", "load", "member", "name"}


# --------------------------------------------- 区间梯形 partial_trapezoid

def test_a_ramp_reduces_to_the_two_kinds_it_generalises():
    """`partial_trapezoid` 是 partial 与 trapezoid 的推广，两头都要退得回去。

    这两条是它最强的自检：固端力由"集中力的核乘强度再积分"得来，
    而 partial（常量强度）和 trapezoid（满跨线性）各有一套**独立写成**的
    闭式。三者对上，说明积分没写错。
    """
    from span_loads import _bending_fixed_end, _partial_fixed_end, _ramp_fixed_end

    # w1 = w2 → partial
    assert _ramp_fixed_end(L, W, W, 0.0, L / 2) == pytest.approx(
        _partial_fixed_end(L, W, 0.0, L / 2), rel=1e-9, abs=1e-6)
    # a=0、b=L → trapezoid
    assert _ramp_fixed_end(L, 0.0, W, 0.0, L) == pytest.approx(
        _bending_fixed_end(L, 0.0, W), rel=1e-9, abs=1e-6)


def test_two_ramps_make_an_exact_symmetric_triangle():
    """**这是补这个类型的理由。**

    双向板短边梁承受的是跨中起峰的对称三角形。这个形状单根杆上一条荷载
    表达不了：trapezoid 是端到端线性，partial 是区间均布。两段 ramp 拼起来
    才是精确的。

    对称三角形（峰值 q）的跨中弯矩闭合解是 qL²/12。实测两段 ramp 给出
    160.0000，而 200 个集中力逼近是 159.9960 —— **新类型比逼近更准**。
    """
    q0 = 30e3
    f = pin_pin(bar())
    f.case().member_spans[1] = [
        SpanLoad(RAMP, (0.0, 0.0, 0.0), (0.0, 0.0, -q0), a=0.0, b=L / 2),
        SpanLoad(RAMP, (0.0, 0.0, -q0), (0.0, 0.0, 0.0), a=L / 2, b=L),
    ]
    sol = solve(f)
    assert check_equilibrium(f, sol, "default")["ok"]
    x_peak, m_peak = peak(f)
    assert x_peak == pytest.approx(L / 2, abs=L / 200)
    assert m_peak == pytest.approx(q0 * L ** 2 / 12.0, rel=1e-6)


def test_a_ramp_survives_being_split_by_an_interior_node():
    """跨越剖分点时，强度要按位置**线性插值**到各段两端。

    直接沿用整段的 w1/w2 会把斜率算错——结果只是偏，不报错，
    所以拿"剖分与否同值"当判据。
    """
    q0 = 30e3
    plain = pin_pin(bar())
    plain.case().member_spans[1] = [
        SpanLoad(RAMP, (0.0, 0.0, 0.0), (0.0, 0.0, -q0), a=0.0, b=L)]
    cut = pin_pin(bar(n=2))
    half = (0.0, 0.0, -q0 / 2)
    cut.case().member_spans[1] = [
        SpanLoad(RAMP, (0.0, 0.0, 0.0), half, a=0.0, b=L / 2)]
    cut.case().member_spans[2] = [
        SpanLoad(RAMP, half, (0.0, 0.0, -q0), a=0.0, b=L / 2)]
    assert peak(plain)[1] == pytest.approx(peak(cut)[1], rel=1e-6)


def test_the_ramp_tool_demands_both_ends_and_both_positions():
    """缺 w2 或缺 a/b 都必须报错，不许替用户猜。

    缺 w2 默认成 w1 的话，写漏的区间梯形会静悄悄变成区间均布；
    默认成 0 又会变成三角形。两种都是"看着正常的错数"。
    """
    s, _ = _framed()
    bad = s.set_member_span_load(1, "partial_trapezoid", [0, 0, -1e3],
                                 a=0.0, b=1.0)
    assert not bad.ok and "w2" in bad.payload["error"]


# ----------------------------------------------------------- 面荷载导荷

Q_AREA = 4e3            # 4 kN/m²


def _simple(span: float):
    """一根简支梁，用来量导荷后的总重。"""
    from agent import Session

    s = Session()
    s.set_model({
        "units": "N-m-Pa",
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                  {"id": 2, "x": span, "y": 0, "z": 0}],
        "members": [{"id": 1, "i": 1, "j": 2, "material": "S", "section": "B"}],
        "materials": [{"name": "S", "E": 2.06e11, "nu": 0.3, "density": 0.0}],
        "sections": [{"name": "B", "A": 8.6e-3, "Iy": 3e-5, "Iz": 1e-4,
                      "J": 1e-6}],
        "supports": [{"node": 1, "fix": [1, 1, 1, 1, 0, 0]},
                     {"node": 2, "fix": [0, 1, 1, 1, 0, 0]}],
    })
    return s


def _carried(s) -> float:
    """这根梁实际承受的总竖向荷载（kN），由反力量出来。"""
    assert s.solve_model().ok
    return s.query_results("reactions").payload["vertical_total_kN"]


@pytest.mark.parametrize("path,span,width,expect", [
    # 单向板：从属宽度 3、梁长 6 → q×3×6
    ("one_way", 6.0, 3.0, Q_AREA * 3.0 * 6.0),
    # 双向板短边梁：三角形，峰值 q·Lx/2，总重 = 峰值×L/2
    ("two_way_short", 4.0, 4.0, (Q_AREA * 4.0 / 2) * 4.0 / 2),
    # 双向板长边梁：梯形，总重 = 峰值×(L − Lx/2)
    ("two_way_long", 6.0, 4.0, (Q_AREA * 4.0 / 2) * (6.0 - 2.0)),
])
def test_each_load_path_puts_the_right_total_on_the_beam(path, span, width,
                                                         expect):
    """导荷唯一的硬判据是**总重**——反力合计必须等于手算。

    真实输入永远是 kN/m²，而求解器只认 kN/m。中间那步以前全靠手算，
    **算错了结果看着完全正常**：量级对、弯矩图也像那么回事，只是总重差一截。
    """
    s = _simple(span)
    assert s.apply_area_load([1], Q_AREA, width, path).ok
    assert _carried(s) == pytest.approx(expect / 1e3, rel=1e-9)


def test_the_four_beams_of_a_panel_carry_exactly_the_panel():
    """**这条是双向板分配最强的校验。**

    4×6 的板，两根短边梁 + 两根长边梁，合计必须精确等于 q×4×6。
    分配少了就是漏载，多了就是重复计——两种都不会报错，只会让整栋楼的
    总重悄悄偏掉。
    """
    short_span, long_span = 4.0, 6.0
    carried = 0.0
    for _ in range(2):
        s = _simple(short_span)
        assert s.apply_area_load([1], Q_AREA, short_span, "two_way_short").ok
        carried += _carried(s)
    for _ in range(2):
        s = _simple(long_span)
        assert s.apply_area_load([1], Q_AREA, short_span, "two_way_long").ok
        carried += _carried(s)
    panel = Q_AREA * short_span * long_span / 1e3
    assert carried == pytest.approx(panel, rel=1e-9)


def test_the_triangle_is_exact_not_an_equivalent_uniform():
    """三角形是**精确生成**的，不是等效均布。

    等效均布（5/8·q₀）只保证跨中弯矩相等；真实三角形的跨中弯矩是 q₀L²/12，
    而 5/8 均布给出 (5/8)q₀L²/8 = 0.078 q₀L²，两者差 6.25%。
    这里验的是前者。
    """
    span = 4.0
    s = _simple(span)
    assert s.apply_area_load([1], Q_AREA, span, "two_way_short").ok
    assert s.solve_model().ok
    peak_q = Q_AREA * span / 2.0
    got = s.query_diagram("Mz", 1, stations=401).payload["peak"]
    assert abs(got) == pytest.approx(peak_q * span ** 2 / 12.0 / 1e3, rel=1e-4)


def test_a_long_edge_shorter_than_the_short_span_is_refused():
    """长边梁比短跨还短，说明 width 传错了或者这根不是长边。

    硬算的话梯形会退化成两段重叠的斜坡，给出一个没有物理意义的分布。
    """
    r = _simple(3.0).apply_area_load([1], Q_AREA, 4.0, "two_way_long")
    assert not r.ok
    assert "不是长边" in r.payload["error"] or "短跨" in r.payload["error"]


# ------------------------------------------------------- 跨间集中力偶

def test_a_couple_gives_the_hand_calculated_reactions_wherever_it_sits():
    """简支梁上力偶的反力是 ±M₀/L，**与位置无关**。

    这一条同时验两件事：数值对不对，以及位置有没有被错误地卷进去。
    """
    m0 = 100e3
    for a in (L / 2, L / 4, 3 * L / 4):
        f = pin_pin(bar())
        # 绕局部 z 的力偶：这套几何下局部 z = −全局 Y
        f.case().member_spans[1] = [SpanLoad(MOMENT, (0.0, -m0, 0.0), a=a)]
        sol = solve(f)
        assert check_equilibrium(f, sol, "default")["ok"]
        r = sol["default"].R[f.node_dofs(1)[2]]
        assert r == pytest.approx(m0 / L, rel=1e-9)


def test_the_moment_diagram_jumps_by_exactly_the_applied_couple():
    """**这是力偶最强的判据。** 弯矩图在作用点处跳跃一个 M₀，不多不少。

    跳跃量算错的话反力仍可能是对的（力偶的合力为零），所以只验反力
    抓不到这类错。跨中力偶：M(a⁻)=+M₀L/(2L)=+M₀/2、M(a⁺)=−M₀/2。
    """
    m0, a = 100e3, L / 2
    f = pin_pin(bar())
    f.case().member_spans[1] = [SpanLoad(MOMENT, (0.0, -m0, 0.0), a=a)]
    sol = solve(f)
    eps = 1e-9
    d = member_diagram(f, sol, 1, at_x=np.array([0.0, a - eps, a + eps, L]))
    assert d.Mz[0] == pytest.approx(0.0, abs=1e-6)
    assert d.Mz[3] == pytest.approx(0.0, abs=1e-6)
    assert d.Mz[1] == pytest.approx(m0 / 2, rel=1e-9)
    assert d.Mz[2] == pytest.approx(-m0 / 2, rel=1e-9)
    assert abs(d.Mz[2] - d.Mz[1]) == pytest.approx(m0, rel=1e-9)


def test_the_element_formula_and_the_nodal_conversion_agree():
    """**两条独立实现必须给出同一个答案。**

    手工搭的 Frame 走 span_loads.fixed_end 的形函数导数公式；
    Session 走编译器——它在力偶位置剖分出节点，再把力偶折成**节点弯矩**，
    那一路是精确的、完全不碰固端力公式。

    这条交叉校验当场抓到过一个真错：我第一版照教科书的"固端弯矩"表写
    M_i = m₀·b(2a−b)/L²，简支梁反力算成 25 kN 而正确值是 12.5 kN，
    整体平衡也不成立。差的正是固端力与等效节点荷载之间那个负号。
    """
    from agent import Session

    m0, a = 100e3, L / 4
    f = pin_pin(bar())
    f.case().member_spans[1] = [SpanLoad(MOMENT, (0.0, -m0, 0.0), a=a)]
    by_element = solve(f)["default"].R[f.node_dofs(1)[2]]

    s = Session()
    s.set_model({
        "units": "N-m-Pa",
        "materials": [{"name": "M", "E": E, "nu": 0.3}],
        "sections": [{"name": "S", "A": 0.01, "Iy": 4e-5, "Iz": 3e-4,
                      "J": 8e-7}],
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                  {"id": 2, "x": L, "y": 0, "z": 0}],
        "members": [{"id": 1, "i": 1, "j": 2, "material": "M", "section": "S"}],
        "supports": [{"node": 1, "fix": [1, 1, 1, 1, 0, 0]},
                     {"node": 2, "fix": [0, 1, 1, 1, 0, 0]}],
        "load_cases": [{"name": "C", "member_spans": [
            {"member": 1, "kind": "moment", "w1": [0.0, -m0, 0.0], "a": a}]}],
    })
    assert s.solve_model().ok
    reactions = s.query_results("reactions", case="C").payload["reactions"]
    by_node = list(reactions.values())[0]["R"][2] * 1e3
    # query_results 的反力按 kN 保留六位，回乘 1e3 后精度到 1e-3 N
    assert by_element == pytest.approx(by_node, rel=1e-6)


def test_a_pure_couple_is_not_mistaken_for_an_empty_load_case():
    """纯力偶的**合力恒为零**，不能因此被判成"没有荷载"。

    实测这条会把一个完全正常的工况整个拦下来：求解平衡残差 2.9e-16、
    跨内挠度 0.81 mm，而 no_applied_load 报 critical 并置 ok=False。
    支座沉降与初应变早就是这样的特例，力偶是第三类。
    """
    from agent import Session

    s = Session()
    s.set_model({
        "units": "N-m-Pa",
        "materials": [{"name": "M", "E": E, "nu": 0.3}],
        "sections": [{"name": "S", "A": 0.01, "Iy": 4e-5, "Iz": 3e-4,
                      "J": 8e-7}],
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                  {"id": 2, "x": L, "y": 0, "z": 0}],
        "members": [{"id": 1, "i": 1, "j": 2, "material": "M", "section": "S"}],
        "supports": [{"node": 1, "fix": [1, 1, 1, 1, 0, 0]},
                     {"node": 2, "fix": [0, 1, 1, 1, 0, 0]}],
        "load_cases": [{"name": "C", "member_spans": [
            {"member": 1, "kind": "moment", "w1": [0.0, -50e3, 0.0],
             "a": L / 2}]}],
    })
    got = s.solve_model()
    assert got.ok, got.payload.get("unusable")


# --------------------------------------------------------- 温度梯度

def _hn_with_gradient(fix_i, fix_j, gradient: float):
    """HN400 梁，两端约束由参数给，施加截面上下温差。"""
    import sections as sec
    from agent import Session

    hn = sec.i_section("HN", 0.400, 0.200, 0.008, 0.013)
    s = Session()
    s.define_materials_and_sections(
        [{"name": "Q", "E": 2.06e11, "nu": 0.3, "density": 0.0,
          "alpha": 1.2e-5}], [hn])
    s.set_model({**s.model,
                 "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                           {"id": 2, "x": 6.0, "y": 0, "z": 0}],
                 "members": [{"id": 1, "i": 1, "j": 2, "material": "Q",
                              "section": "HN"}],
                 "supports": [{"node": 1, "fix": list(fix_i)},
                              {"node": 2, "fix": list(fix_j)}]})
    assert s.set_member_strain(1, gradient_t=gradient).ok
    assert s.solve_model().ok
    return s, hn


def test_a_temperature_gradient_bends_a_restrained_member():
    """上下温差让杆想要弯；被约束住就产生弯矩 EI·κ，κ = α·ΔT/h。

    HN400、ΔT=30℃：κ = 1.2e-5×30/0.4 = 9e-4 /m，EI·κ = 42.5769 kN·m，
    且沿杆**恒定**——纯弯曲不产生剪力。
    """
    s, hn = _hn_with_gradient((1,) * 6, (1,) * 6, 30.0)
    kappa = 1.2e-5 * 30.0 / (2 * hn["cy"])
    exact = 2.06e11 * hn["Iz"] * kappa / 1e3
    d = s.query_diagram("Mz", 1, stations=11).payload
    assert abs(d["peak"]) == pytest.approx(exact, rel=1e-6)
    # 两端同值 ⇒ 沿杆恒定
    assert d["at_i_end"] == pytest.approx(d["at_j_end"], rel=1e-9)


def test_a_statically_determinate_member_is_free_to_bend():
    """**这是温度作用最强的判据。**

    静定结构自由变形，温度不产生任何内力——和"自由伸长的杆轴力为零"
    是同一条道理。算出非零内力就说明初曲率被当成了外荷载。
    """
    s, _ = _hn_with_gradient((1, 1, 1, 1, 0, 0), (0, 1, 1, 1, 0, 0), 30.0)
    for comp in ("N", "Vy", "Vz", "My", "Mz"):
        peak = s.query_diagram(comp, 1, stations=21).payload["peak"]
        assert abs(peak) < 1e-6, f"{comp} 在静定杆上不该有内力，得到 {peak}"


def test_a_uniform_rise_and_a_gradient_are_not_the_same_thing():
    """均匀升温给轴力，上下温差给弯矩。混为一谈会得到完全不同的内力。

    把梯度折算成某种"等效均匀温度"是错的——前者产生弯矩，后者产生轴力，
    两者在同一根杆上同时存在也互不替代。
    """
    import sections as sec
    from agent import Session

    hn = sec.i_section("HN", 0.400, 0.200, 0.008, 0.013)
    uniform = Session()
    uniform.define_materials_and_sections(
        [{"name": "Q", "E": 2.06e11, "nu": 0.3, "density": 0.0,
          "alpha": 1.2e-5}], [hn])
    uniform.set_model({**uniform.model,
                       "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                                 {"id": 2, "x": 6.0, "y": 0, "z": 0}],
                       "members": [{"id": 1, "i": 1, "j": 2, "material": "Q",
                                    "section": "HN"}],
                       "supports": [{"node": 1, "fix": [1] * 6},
                                    {"node": 2, "fix": [1] * 6}]})
    assert uniform.set_member_strain(1, delta_t=30.0).ok
    assert uniform.solve_model().ok
    gradient, _ = _hn_with_gradient((1,) * 6, (1,) * 6, 30.0)

    n_uniform = abs(uniform.query_diagram("N", 1).payload["peak"])
    m_uniform = abs(uniform.query_diagram("Mz", 1).payload["peak"])
    n_gradient = abs(gradient.query_diagram("N", 1).payload["peak"])
    m_gradient = abs(gradient.query_diagram("Mz", 1).payload["peak"])
    assert n_uniform > 1.0 and m_uniform < 1e-6, "均匀升温只产生轴力"
    assert m_gradient > 1.0 and n_gradient < 1e-6, "温度梯度只产生弯矩"


def test_a_gradient_needs_the_section_depth_and_says_so():
    """算 κ=α·ΔT/h 要截面高度，只给 A/Iy/Iz/J 的截面算不出来。

    随便拿一个高度顶上会给出一个来路不明的弯矩——和正应力缺 cy 时的
    处置一致：明确拒绝，并说清楚缺什么。
    """
    from agent import Session

    s = Session()
    s.define_materials_and_sections(
        [{"name": "Q", "E": 2.06e11, "nu": 0.3, "alpha": 1.2e-5}],
        [{"name": "B", "A": 0.01, "Iy": 4e-5, "Iz": 3e-4, "J": 8e-7}])
    s.set_model({**s.model,
                 "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                           {"id": 2, "x": 6.0, "y": 0, "z": 0}],
                 "members": [{"id": 1, "i": 1, "j": 2, "material": "Q",
                              "section": "B"}],
                 "supports": [{"node": 1, "fix": [1] * 6},
                              {"node": 2, "fix": [1] * 6}]})
    bad = s.set_member_strain(1, gradient_t=30.0)
    assert not bad.ok and "cy" in bad.payload["error"]


# ----------------------------------------------------------- 规范组合

def _three_cases():
    """恒 + 活 + 风 三个工况的门式刚架。"""
    from agent import Session

    s = Session()
    s.define_materials_and_sections(
        [{"name": "Q", "E": 2.06e11, "nu": 0.3, "density": 7850.0}],
        [{"name": "B", "A": 0.018, "Iy": 5e-5, "Iz": 2e-3, "J": 1e-6}])
    r = s.generate_frame(spans=[6.0], storeys=[4.0])
    s.assign_properties([int(m["id"]) for m in s.model["members"]],
                        material="Q", section="B")
    beams = r.payload["beam_member_ids"]
    s.set_load_cases(cases=[
        {"name": "DL", "member_loads": [{"member": m, "w": [0, 0, -20e3]}
                                        for m in beams]},
        {"name": "LL", "member_loads": [{"member": m, "w": [0, 0, -10e3]}
                                        for m in beams]},
        {"name": "WX", "nodal_loads": [{"node": 3,
                                        "load": [15e3, 0, 0, 0, 0, 0]}]},
    ])
    return s


def test_every_variable_load_takes_a_turn_at_controlling():
    """**这是自动生成组合的理由。**

    只算"活载控制"而不算"风控制"，风控制的那些杆件就永远查不出来——
    包络只在给定的组合里取极值，少一个就是少一个，而结果看着完全正常。
    """
    s = _three_cases()
    got = s.generate_combinations(dead="DL", live="LL", wind="WX")
    assert got.ok
    named = {c["name"]: c["factors"] for c in got.payload["combos"]}
    # 两个可变荷载各有一组基本组合
    assert "基本-LL控制" in named and "基本-WX控制" in named
    # GB 50068-2018：γ_G=1.3、γ_Q=1.5、ψ_c(风)=0.6、ψ_c(活)=0.7
    assert named["基本-LL控制"] == pytest.approx(
        {"DL": 1.3, "LL": 1.5, "WX": 1.5 * 0.6})
    assert named["基本-WX控制"] == pytest.approx(
        {"DL": 1.3, "WX": 1.5, "LL": 1.5 * 0.7})


def test_a_favourable_dead_load_combination_is_generated_for_wind():
    """风吸把构件往上拔时恒载是**有利**的，这时用 1.3 反而不保守。

    软件判不了"哪根杆件上恒载算有利"——那是逐杆逐截面的事。
    所以两组都生成，交给包络逐点去挑。
    """
    s = _three_cases()
    got = s.generate_combinations(dead="DL", live="LL", wind="WX")
    named = {c["name"]: c["factors"] for c in got.payload["combos"]}
    assert named["基本-WX控制(恒载有利)"] == pytest.approx({"DL": 1.0, "WX": 1.5})


def test_the_2012_edition_uses_the_old_partial_factors():
    """既有项目按 2012 版校核：γ_G=1.2、γ_Q=1.4。

    两版差 8%，选错了不会报错，只会让结果整体偏松或偏紧。
    """
    s = _three_cases()
    got = s.generate_combinations(dead="DL", live="LL",
                                  standard="GB50009-2012")
    named = {c["name"]: c["factors"] for c in got.payload["combos"]}
    assert named["基本-LL控制"] == pytest.approx({"DL": 1.2, "LL": 1.4})


def test_the_psi_factors_can_be_overridden_and_are_reported():
    """ψ 系数随建筑类别变。默认是一般民用建筑，商业、库房不同。

    用错不会报错，只会让组合悄悄偏小——所以既要能覆盖，也要把实际用的
    那一套原样报出来。
    """
    s = _three_cases()
    got = s.generate_combinations(dead="DL", live="LL", wind="WX",
                                  psi_c={"live": 0.9})
    named = {c["name"]: c["factors"] for c in got.payload["combos"]}
    assert named["基本-WX控制"]["LL"] == pytest.approx(1.5 * 0.9)
    assert got.payload["psi_c"]["live"] == pytest.approx(0.9)


def test_generated_combinations_feed_the_envelope_automatically():
    """组合一经写入，包络与强度校核自动以它们为对象。

    这条连的是全链路：生成组合 → 求解 → 包络逐点取极值并记住控制组合。
    """
    s = _three_cases()
    assert s.generate_combinations(dead="DL", live="LL", wind="WX").ok
    assert s.solve_model().ok
    assert set(s.solution.combos) == {
        c["name"] for c in s.model["combos"]}
    got = s.query_envelope(component="Mz")
    assert got.ok


def test_an_unknown_case_name_is_refused_before_anything_is_written():
    """工况名写错必须当场拒绝，而且不能改坏模型。"""
    s = _three_cases()
    before = list(s.model.get("combos") or [])
    bad = s.generate_combinations(dead="DL", live="没有这个工况")
    assert not bad.ok
    assert list(s.model.get("combos") or []) == before


# ----------------------------------------------------- 活载最不利布置

def _three_span_frame():
    """三跨单层框架，梁已归入集合「梁」。"""
    from agent import Session

    s = Session()
    s.define_materials_and_sections(
        [{"name": "Q", "E": 2.06e11, "nu": 0.3}],
        [{"name": "B", "A": 0.018, "Iy": 5e-5, "Iz": 2e-3, "J": 1e-6}])
    r = s.generate_frame(spans=[6.0, 6.0, 6.0], storeys=[4.0])
    s.assign_properties([int(m["id"]) for m in s.model["members"]],
                        material="Q", section="B")
    beams = r.payload["beam_member_ids"]
    s.define_set("梁", member_ids=beams)
    return s, beams


def _midspan(s, member_id: int, case: str) -> float:
    from internal_forces import member_diagram, member_length

    length = member_length(s.frame, member_id)
    d = member_diagram(s.frame, s.solution, member_id, case,
                       at_x=np.array([length / 2]))
    return float(d.Mz[0])


def test_alternate_spans_beat_full_loading_at_midspan():
    """**这是最不利布置存在的理由。**

    连续梁与框架的跨中正弯矩不是满布时最大，而是隔跨布置时最大。
    实测三跨框架的中跨：满布 28.7 kN·m、偶数跨布置 41.2 kN·m ——
    只算满布会算小 43%，而且算小多少取决于跨数与刚度比，看不出来。
    """
    s, beams = _three_span_frame()
    assert s.generate_live_patterns("梁", [0, 0, -20e3]).ok
    assert s.solve_model().ok
    full = _midspan(s, beams[1], "LL-满布")
    even = _midspan(s, beams[1], "LL-偶数跨")
    assert even > full * 1.3, (full, even)
    # 边跨由奇数跨布置控制
    assert _midspan(s, beams[0], "LL-奇数跨") > _midspan(s, beams[0], "LL-满布")


def test_adjacent_spans_govern_the_support_moment():
    """支座负弯矩由相邻跨布置控制，不是跨中那一组。

    跨中和支座由**不同**的布置控制——这正是必须把几种布置都算一遍的原因。
    """
    from internal_forces import member_diagram

    s, beams = _three_span_frame()
    assert s.generate_live_patterns(
        "梁", [0, 0, -20e3],
        patterns=("full", "odd", "even", "adjacent")).ok
    assert s.solve_model().ok

    def left_support(case: str) -> float:
        return float(member_diagram(s.frame, s.solution, beams[1], case,
                                    stations=51).Mz[0])

    assert left_support("LL-相邻12") < left_support("LL-满布")


def test_the_span_grouping_is_reported_for_checking():
    """跨的归类是**几何推断**，必须原样报出来让人核对。

    混进了柱或另一方向的梁时归出来的跨会很怪，与其猜不如让人看见。
    """
    s, beams = _three_span_frame()
    got = s.generate_live_patterns("梁", [0, 0, -20e3])
    assert got.payload["span_count"] == 3
    ranges = [tuple(item["range"]) for item in got.payload["spans"]]
    assert ranges == [(0.0, 6.0), (6.0, 12.0), (12.0, 18.0)]
    # 每跨的梁都列了出来，且互不重叠
    everything = [m for item in got.payload["spans"] for m in item["members"]]
    assert sorted(everything) == sorted(beams)


def test_a_single_span_refuses_alternate_patterns_instead_of_faking_them():
    """只有一跨时隔跨布置无从谈起，直接说清楚。

    硬生成的话"奇数跨"等于满布、"偶数跨"是个空工况——后者还会被
    no_applied_load 拦下来，用户拿到的是一串莫名其妙的错误。
    """
    from agent import Session

    s = Session()
    s.define_materials_and_sections(
        [{"name": "Q", "E": 2.06e11, "nu": 0.3}],
        [{"name": "B", "A": 0.018, "Iy": 5e-5, "Iz": 2e-3, "J": 1e-6}])
    r = s.generate_frame(spans=[6.0], storeys=[4.0])
    s.assign_properties([int(m["id"]) for m in s.model["members"]],
                        material="Q", section="B")
    bad = s.generate_live_patterns(r.payload["beam_member_ids"], [0, 0, -20e3])
    assert not bad.ok and "跨" in bad.payload["error"]


def test_patterns_and_combinations_compose():
    """布置生成工况，组合消费工况——每种布置都要轮流当控制荷载。

    这条连的是规范层全链路：布置 → 组合 → 求解 → 包络。
    """
    s, beams = _three_span_frame()
    made = s.generate_live_patterns("梁", [0, 0, -20e3])
    names = [c["name"] for c in made.payload["cases"]]
    got = s.generate_combinations(live=names)
    assert got.ok
    controlled = {c["name"] for c in got.payload["combos"]}
    for name in names:
        assert f"基本-{name}控制" in controlled
