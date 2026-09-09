"""单元内挠度的验证。

这个功能是被扫描工具逼出来的：8 m 梁 25 kN/m，只查节点报 0.16 mm，
单元内真值 7.86 mm——**差 48 倍**。拿前者去校核 L/400 会得出"绰绰有余"
的错误结论，而且看不出任何异常。

做法是对已经算好的精确弯矩两次积分，所以每种荷载都自动覆盖。

**一条教训写在最前面**：只验"峰值对上闭合解"是**不够的**。两端位移是
强加的，曲率取反时整条曲线翻过来、两端仍被拉回正确值、峰值大小还一样——
第一版符号就是反的，而那几条闭合解测试全绿。真正定住符号的是两条：
端部转角（不参与定常数，是独立的）和单元**内部**某一点的值。
"""

import numpy as np
import pytest

from frame3d import Frame, Material, Member, Node, Section, solve
from internal_forces import (max_centerline_displacement, max_deflection,
                             member_deflection, member_displacement)
from span_loads import POINT, SpanLoad

E, NU, RHO = 2.1e11, 0.3, 7850.0
A, IY, IZ, J = 0.01, 4e-5, 3e-4, 8e-7
L, W, P = 6.0, -20e3, -30e3


def bar(n: int, ends: str, udl: bool = True) -> Frame:
    """只留 x–z 平面弯曲的梁。全局 z 向位移绕局部 z 弯曲，用 Iz。"""
    f = Frame()
    for k in range(n + 1):
        f.nodes[k + 1] = Node(k + 1, L * k / n, 0.0, 0.0)
        f.supports[k + 1] = (1, 1, 0, 1, 0, 1)
    for k in range(n):
        f.members[k + 1] = Member(k + 1, k + 1, k + 2, "S", "M")
    f.sections["S"] = Section("S", A, IY, IZ, J)
    f.materials["M"] = Material("M", E, NU, RHO)

    def fix(nid, *dofs):
        mask = list(f.supports[nid])
        for d in dofs:
            mask[d] = 1
        f.supports[nid] = tuple(mask)

    if ends == "simple":
        fix(1, 2); fix(n + 1, 2)
    elif ends == "cantilever":
        fix(1, 2, 4)
    else:
        fix(1, 2, 4); fix(n + 1, 2, 4)
    if udl:
        for k in range(1, n + 1):
            f.member_loads[k] = (0.0, 0.0, W)
    return f


def peak(f: Frame) -> float:
    return abs(max_deflection(f, solve(f), stations=201)["value"])


# ------------------------------------------------- 闭合解

EXACT = {
    "simple": 5 * abs(W) * L ** 4 / (384 * E * IZ),      # 5wL⁴/384EI
    "clamped": abs(W) * L ** 4 / (384 * E * IZ),         # wL⁴/384EI
    "cantilever": abs(W) * L ** 4 / (8 * E * IZ),        # wL⁴/8EI
}


@pytest.mark.parametrize("ends", list(EXACT))
@pytest.mark.gold
def test_uniform_load_deflection_matches_the_closed_form(ends):
    n = 2 if ends == "clamped" else 1
    assert peak(bar(n if ends != "simple" else 2, ends)) == pytest.approx(
        EXACT[ends], rel=1e-5)


def test_a_midspan_point_load_matches_PL3_over_48EI():
    f = bar(2, "simple", udl=False)
    f.member_spans[1] = [SpanLoad(POINT, (0.0, 0.0, P), a=L / 2)]
    # 集中力落在 1 号杆的中点（该杆长 L/2），换算到整跨就是跨中
    f = bar(1, "simple", udl=False)
    f.member_spans[1] = [SpanLoad(POINT, (0.0, 0.0, P), a=L / 2)]
    assert peak(f) == pytest.approx(abs(P) * L ** 3 / (48 * E * IZ), rel=1e-4)


def test_a_cantilever_tip_load_matches_PL3_over_3EI():
    f = bar(1, "cantilever", udl=False)
    f.nodal_loads[2] = (0.0, 0.0, P, 0.0, 0.0, 0.0)
    assert peak(f) == pytest.approx(abs(P) * L ** 3 / (3 * E * IZ), rel=1e-6)


def test_the_deflection_points_the_same_way_as_the_load():
    """向下的荷载必须给出向下的挠度。符号推反的话数值一样、方向朝上，
    只比大小的测试完全抓不住。

    注意挠度出现在**局部 y**（返回值的 v）：全局 z 向位移在这套局部坐标下
    绕局部 z 弯曲，用的是 Iz。和内力、模态那两处是同一个坐标关系。
    """
    f = bar(2, "simple")
    x, v, w = member_deflection(f, solve(f), 1, stations=51)
    assert v.min() < 0, "全局 z 向下的荷载应当产生负的局部 y 位移"
    assert np.allclose(w, 0.0), "局部 z 方向没有荷载，不该有位移"


# ------------------------------------------------- 与网格无关

@pytest.mark.parametrize("ends", list(EXACT))
def test_the_answer_is_independent_of_the_mesh(ends):
    """这是整套做法的意义所在：单跨一个单元也拿得到跨中挠度。

    只看节点的话，1 个单元报 0、8 个单元才慢慢逼近——那才是真正的问题。
    """
    counts = (2, 4, 8) if ends == "clamped" else (1, 2, 4, 8)
    got = [peak(bar(n, ends)) for n in counts]
    for value in got:
        assert value == pytest.approx(EXACT[ends], rel=2e-4), got


def test_nodal_query_misses_the_midspan_entirely():
    """把"只看节点"的失效摆出来当回归测试。

    单跨一个单元的简支梁，所有节点位移都是零（支座 + 端点），
    而真实跨中挠度是 5.36 mm。这条测试是那个 48 倍差距的存档。
    """
    f = bar(1, "simple")
    sol = solve(f)
    nodal = max(abs(float(sol["default"].U[d]))
                for nid in f.nodes for d in f.node_dofs(nid)[:3])
    assert nodal < 1e-12, "单元一个时节点上根本量不到挠度"
    assert peak(f) == pytest.approx(EXACT["simple"], rel=1e-4)


# ------------------------------------------------- 独立校验

def test_the_recovered_end_slope_matches_the_nodal_rotation():
    """两个积分常数只用端点**位移**定，端部**转角**因此是独立的校验。

    **这是唯一抓住符号写反的那条测试。** 曲率取反时闭合解的峰值检验照样通过
    （两端被强加、峰值大小不变），只有转角会露馅：悬臂根部的 v′(0) 本该为零，
    符号反了会变成 −0.0171。
    """
    f = bar(1, "cantilever")
    sol = solve(f)
    x, v, w = member_deflection(f, sol, 1, stations=4001)
    span = abs(W) * L ** 3 / (6 * E * IZ)          # 自由端转角量级
    root_slope = (v[1] - v[0]) / (x[1] - x[0])
    tip_slope = (v[-1] - v[-2]) / (x[-1] - x[-2])
    node_rot = float(sol["default"].U[f.node_dofs(2)[4]])   # 绕全局 y
    assert abs(root_slope) < 1e-3 * span, "固定端斜率应当为零"
    # 斜率与转角**差一个负号**，这是坐标约定不是错误：局部 y ∥ 全局 z，
    # 而绕全局 +y 的转动把 +x 转向 −z，所以 dv/dx = −ry
    assert tip_slope == pytest.approx(-node_rot, rel=1e-4)
    assert tip_slope == pytest.approx(W * L ** 3 / (6 * E * IZ), rel=1e-4)


def test_an_interior_point_matches_the_closed_form():
    """悬臂均布梁 v(x) = w·x²(6L² − 4Lx + x²) / (24EI)。

    取跨中——那里既不是节点、也不是峰值，所以两端强加的位移帮不上忙。
    这是第二条对符号敏感的判据。
    """
    f = bar(1, "cantilever")
    x, v, w = member_deflection(f, solve(f), 1, stations=4001)
    k = int(np.argmin(np.abs(x - L / 2)))
    a = L / 2
    exact = W * a ** 2 * (6 * L ** 2 - 4 * L * a + a ** 2) / (24 * E * IZ)
    assert v[k] == pytest.approx(exact, rel=1e-5)


def test_a_flipped_curvature_sign_would_be_caught():
    """把符号写反的那一版是什么样子，在这里存个档。

    悬臂梁曲率取反时：峰值（自由端）仍然精确，因为它就是强加的节点位移；
    但根部斜率会从 0 变成 −wL³/(2·6EI) 量级。以后谁再改这个符号，
    先看这条测试。
    """
    f = bar(1, "cantilever")
    x, v, w = member_deflection(f, solve(f), 1, stations=4001)
    tip = abs(v[-1])
    assert tip == pytest.approx(abs(W) * L ** 4 / (8 * E * IZ), rel=1e-9), \
        "端点值来自节点解，符号写反也是对的——所以它证明不了什么"
    root_slope = abs((v[1] - v[0]) / (x[1] - x[0]))
    assert root_slope < 1e-3 * abs(W) * L ** 3 / (6 * E * IZ)


def test_deflection_scales_inversely_with_stiffness():
    a = peak(bar(2, "simple"))
    stiff = bar(2, "simple")
    stiff.sections["S"] = Section("S", A, IY, 2 * IZ, J)
    assert peak(stiff) == pytest.approx(a / 2, rel=1e-6)


def test_deflection_scales_with_the_load():
    a = peak(bar(2, "simple"))
    heavy = bar(2, "simple")
    for k in heavy.members:
        heavy.member_loads[k] = (0.0, 0.0, 3 * W)
    assert peak(heavy) == pytest.approx(3 * a, rel=1e-9)


def test_the_summary_reports_where_the_peak_is():
    f = bar(4, "simple")
    got = max_deflection(f, solve(f), stations=201)
    # 跨中在第 2 或第 3 根杆的端部附近
    assert got["member"] in (2, 3)
    total_x = (got["member"] - 1) * (L / 4) + got["x"]
    assert total_x == pytest.approx(L / 2, abs=L / 100)


def test_full_centerline_displacement_has_exact_member_end_values():
    f = bar(1, "cantilever")
    f.nodal_loads[2] = (1000.0, 0.0, P, 0.0, 0.0, 0.0)
    sol = solve(f)
    x, displacement = member_displacement(f, sol, 1, stations=401)
    assert displacement[0] == pytest.approx(sol.U[f.node_dofs(1)[:3]])
    assert displacement[-1] == pytest.approx(sol.U[f.node_dofs(2)[:3]])
    got = max_centerline_displacement(f, sol, stations=401)
    assert got["member"] == 1
    assert got["value"] == pytest.approx(
        np.linalg.norm(displacement, axis=1).max())


# ------------------------------------------------- Agent 层

def test_the_agent_reports_the_ratio_when_given_a_span():
    """校核 L/400 要的就是这个比值，让工具直接给，别让模型自己去除——
    但跨度得由调用方给，程序判断不了。"""
    from agent import Session

    s = Session()
    s.define_materials_and_sections(
        [{"name": "M", "E": E, "nu": NU, "density": RHO}],
        [{"name": "S", "A": A, "Iy": IY, "Iz": IZ, "J": J}])
    s.set_model(model={
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                  {"id": 2, "x": L, "y": 0, "z": 0}],
        "members": [{"id": 1, "i": 1, "j": 2, "section": "S", "material": "M"}],
        "supports": [{"node": 1, "fix": [1, 1, 1, 1, 0, 0]},
                     {"node": 2, "fix": [0, 1, 1, 1, 0, 0]}]})
    s.set_load_cases(cases=[{"name": "DL",
                             "member_loads": [{"member": 1, "w": [0, 0, W]}]}])
    s.solve_model()

    nodal = s.query_results(what="max_displacement", case="DL").payload
    inner = s.query_results(what="max_deflection", case="DL", span=L).payload
    assert inner["magnitude_mm"] == pytest.approx(EXACT["simple"] * 1000, rel=1e-4)
    assert inner["magnitude_mm"] > 100 * nodal["magnitude_mm"], "两者差着量级"
    assert inner["span_over_deflection"] == pytest.approx(
        L / EXACT["simple"], rel=1e-3)
    # 这里单跨就是一根杆，两个比值恰好相同；划成多个单元时就不同了
    assert inner["member_length_over_deflection"] == pytest.approx(
        inner["span_over_deflection"], rel=1e-6)


def test_a_purely_axial_structure_reports_zero_not_an_error():
    """纯轴向受力时横向挠度处处为零。

    第一版的 max_deflection 用 {"value": 0.0} 起头，"大于 0"永远不成立，
    结果整个查询报"没有杆件"——既错又难懂。桁架类结构都会撞上。
    """
    from agent import Session

    s = Session()
    s.define_materials_and_sections(
        [{"name": "M", "E": E, "nu": NU, "density": RHO}],
        [{"name": "S", "A": A, "Iy": IY, "Iz": IZ, "J": J}])
    s.set_model(model={
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                  {"id": 2, "x": L, "y": 0, "z": 0}],
        "members": [{"id": 1, "i": 1, "j": 2, "section": "S", "material": "M"}],
        "supports": [{"node": 1, "fix": [1] * 6},
                     {"node": 2, "fix": [0, 1, 1, 1, 1, 1]}]})
    s.set_load_cases(cases=[{"name": "T", "nodal_loads":
                             [{"node": 2, "load": [50e3, 0, 0, 0, 0, 0]}]}])
    s.solve_model()
    r = s.query_results(what="max_deflection", case="T")
    assert r.ok, r.payload
    assert r.payload["magnitude_mm"] == pytest.approx(0.0, abs=1e-9)
    assert r.payload["at_member"] == 1
    assert r.payload["member_length_over_deflection"] is None, \
        "挠度为零时比值没有意义"


def test_the_length_ratio_is_named_after_what_it_actually_divides_by():
    """第一版把"杆长/挠度"叫成挠跨比，还说可以和 L/400 直接比。

    门式刚架的斜梁是**两根**杆，每根半跨；梁划成 4 个单元时差 4 倍。
    方向上是偏严（报"不满足"而实际可能满足），但仍然是错的——
    会让人去加大一个本来够用的截面。
    程序无法自己判断设计跨度（斜梁的常规跨度是水平距离，不是杆长），
    所以如实报杆长比值，跨度要由调用方给。
    """
    from agent import Session

    s = Session()
    s.define_materials_and_sections(
        [{"name": "M", "E": E, "nu": NU, "density": RHO}],
        [{"name": "S", "A": A, "Iy": IY, "Iz": IZ, "J": J}])
    # 把 L 长的梁划成两个单元：每根杆只有 L/2
    s.set_model(model={
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                  {"id": 2, "x": L / 2, "y": 0, "z": 0},
                  {"id": 3, "x": L, "y": 0, "z": 0}],
        "members": [{"id": 1, "i": 1, "j": 2, "section": "S", "material": "M"},
                    {"id": 2, "i": 2, "j": 3, "section": "S", "material": "M"}],
        "supports": [{"node": 1, "fix": [1, 1, 1, 1, 0, 0]},
                     {"node": 3, "fix": [0, 1, 1, 1, 0, 0]}]})
    s.set_load_cases(cases=[{"name": "DL", "member_loads":
                             [{"member": 1, "w": [0, 0, W]},
                              {"member": 2, "w": [0, 0, W]}]}])
    s.solve_model()

    plain = s.query_results(what="max_deflection", case="DL").payload
    assert "span_over_deflection" not in plain, "没给跨度就别叫它跨度比"
    assert plain["member_length_m"] == pytest.approx(L / 2, rel=1e-9)
    # 杆长比值确实是跨度比的一半——正是当初报错的那个倍数
    delta = plain["magnitude_mm"] / 1000.0
    assert plain["member_length_over_deflection"] == pytest.approx(
        (L / 2) / delta, rel=1e-3)

    given = s.query_results(what="max_deflection", case="DL", span=L).payload
    assert given["span_over_deflection"] == pytest.approx(L / delta, rel=1e-3)
    assert given["span_over_deflection"] == pytest.approx(
        2 * plain["member_length_over_deflection"], rel=1e-3)
