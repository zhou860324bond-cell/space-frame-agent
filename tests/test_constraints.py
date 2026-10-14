"""运动约束（Abaqus 的 MPC BEAM / TIE / PIN 与运动耦合）对拍闭合解。

约束是主从自由度消元，不是罚刚度：结果要和经典解析解对到数值精度，
罚刚度做不到这一点（取大病态、取小不刚），这些测试就是防它悄悄换成罚函数。
"""
import numpy as np
import pytest

from buckling import buckling
from frame3d import (Constraint, Frame, Material, Member, Node, Section,
                     check_equilibrium, check_model, dof_basis, solve)
from modal import modal
from nonlinear import solve_pdelta

E, NU = 210e9, 0.3
G = E / (2 * (1 + NU))
A, I, J = 0.01, 8e-6, 1.2e-5
SQ = Section("SQ", A, I, I, J)           # Iy = Iz，免得把强弱轴混进约束的验证里
MAT = Material("S", E, NU)
FIX = (1, 1, 1, 1, 1, 1)
L = 3.0


def build(nodes, members, constraints=(), supports=None):
    f = Frame()
    for n in nodes:
        f.nodes[n.id] = n
    for m in members:
        f.members[m.id] = m
    f.sections[SQ.name] = SQ
    f.materials[MAT.name] = MAT
    f.constraints = list(constraints)
    f.supports.update(supports or {})
    return f


def bar(mid, i, j):
    return Member(mid, i, j, SQ.name, MAT.name)


def offset_cantilever(e=0.4, P=-5e3, units_scale=1.0):
    """悬臂端 2 用刚臂连到上方 e 处的 3，力 P 沿 y 作用在 3 上。"""
    s = units_scale
    f = build([Node(1, 0, 0, 0), Node(2, L * s, 0, 0), Node(3, L * s, 0, e * s)],
              [bar(1, 1, 2)], [Constraint("arm", 2, (3,))], {1: FIX})
    f.nodal_loads[3] = (0, P, 0, 0, 0, 0)
    return f


@pytest.mark.gold
def test_rigid_arm_turns_offset_force_into_torsion():
    """偏心力经刚臂：端点挠度 PL³/3EI，扭转角 T·L/GJ，加载点再多 e·θ。

    u3y = PL³/(3EI) + e²·P·L/(GJ)，两项都是闭合解。少了 θ × d 那一项，
    第二项就没了——偏心荷载的扭转被静默吞掉。
    """
    e, P = 0.4, -5e3
    f = offset_cantilever(e, P)
    s = solve(f)
    u2 = s.U[f.node_dofs(2)]
    u3 = s.U[f.node_dofs(3)]
    assert u2[1] == pytest.approx(P * L**3 / (3 * E * I), rel=1e-10)
    assert u2[3] == pytest.approx(-e * P * L / (G * J), rel=1e-10)
    assert u3[1] == pytest.approx(P * L**3 / (3 * E * I) + e**2 * P * L / (G * J), rel=1e-10)
    np.testing.assert_allclose(u3[3:], u2[3:], rtol=0, atol=1e-15)
    # 支座反力：力 −P，扭矩 e·P（经约束传到支座的部分也要算进去）
    r1 = s.R[f.node_dofs(1)]
    assert r1[1] == pytest.approx(-P, rel=1e-10)
    assert r1[3] == pytest.approx(e * P, rel=1e-10)
    assert check_equilibrium(f, s)["ok"]


def test_rigid_arm_is_unit_invariant():
    """同一模型存成毫米制，位移按 1000 倍、转角不变——刚臂系数带长度量纲。"""
    si = solve(offset_cantilever())
    f_mm = offset_cantilever(units_scale=1000.0)
    f_mm.units = "N-mm-MPa"
    f_mm.sections = {"SQ": Section("SQ", A * 1e6, I * 1e12, I * 1e12, J * 1e12)}
    f_mm.materials = {"S": Material("S", E / 1e6, NU)}
    mm = solve(f_mm)
    for k in range(3):
        assert mm.U[k::6] == pytest.approx(1000.0 * si.U[k::6], rel=1e-9, abs=1e-12)
    for k in range(3, 6):
        assert mm.U[k::6] == pytest.approx(si.U[k::6], rel=1e-9, abs=1e-15)


def test_chained_constraints_compose_rigidly():
    """3 跟随 2、2 跟随 1 的链式约束，与 3 直接跟随 1 等价（刚体运动可以复合）。"""
    def model(chained):
        nodes = [Node(1, 0, 0, 0), Node(2, L, 0, 0), Node(3, L, 0.5, 0.2), Node(4, L, 0.9, 0.7)]
        cons = ([Constraint("a", 2, (3,)), Constraint("b", 3, (4,))] if chained
                else [Constraint("a", 2, (3, 4))])
        f = build(nodes, [bar(1, 1, 2)], cons, {1: FIX})
        f.nodal_loads[4] = (1e3, -2e3, 3e3, 100.0, 0, -50.0)
        return f
    a, b = solve(model(True)), solve(model(False))
    np.testing.assert_allclose(a.U, b.U, rtol=1e-12, atol=1e-18)


@pytest.mark.gold
def test_tie_joins_coincident_nodes_like_one_node():
    """两端固定梁在跨中断开成两个重合节点，TIE 起来就是连续梁：δ = P(2L)³/192EI。"""
    P = -10e3
    f = build([Node(1, 0, 0, 0), Node(2, L, 0, 0), Node(3, L, 0, 0), Node(4, 2 * L, 0, 0)],
              [bar(1, 1, 2), bar(2, 3, 4)],
              [Constraint("tie", 2, (3,), rigid=False)], {1: FIX, 4: FIX})
    assert check_model(f) == []
    f.nodal_loads[2] = (0, 0, P, 0, 0, 0)
    s = solve(f)
    assert s.U[f.node_dofs(3)[2]] == pytest.approx(P * (2 * L)**3 / (192 * E * I), rel=1e-10)


@pytest.mark.gold
def test_pin_makes_a_hinge():
    """PIN 只连平动：两端固定、跨中铰接，各半跨是承担 P/2 的悬臂，δ = (P/2)L³/3EI。"""
    P = -10e3
    f = build([Node(1, 0, 0, 0), Node(2, L, 0, 0), Node(3, L, 0, 0), Node(4, 2 * L, 0, 0)],
              [bar(1, 1, 2), bar(2, 3, 4)],
              [Constraint("pin", 2, (3,), dofs=(0, 1, 2), rigid=False)], {1: FIX, 4: FIX})
    f.nodal_loads[2] = (0, 0, P, 0, 0, 0)
    s = solve(f)
    assert s.U[f.node_dofs(2)[2]] == pytest.approx(0.5 * P * L**3 / (3 * E * I), rel=1e-10)
    # 铰两侧转角反号，说明转动确实没被连上
    assert s.U[f.node_dofs(2)[4]] == pytest.approx(-s.U[f.node_dofs(3)[4]], rel=1e-10)


def diaphragm(a=2.0, h=L, masses=False):
    """两根悬臂柱（x=±a），柱顶用平面刚性楼板（dofs 0,1,5）连到中心参考点 9。

    参考点只受约束连接，平面外的 uz、rx、ry 没有刚度——和 Abaqus 的参考点
    一样要用户补边界条件，这里给它加上。
    """
    nodes = [Node(1, -a, 0, 0), Node(2, -a, 0, h), Node(3, a, 0, 0), Node(4, a, 0, h),
             Node(9, 0, 0, h)]
    f = build(nodes, [bar(1, 1, 2), bar(2, 3, 4)],
              [Constraint("floor", 9, (2, 4), dofs=(0, 1, 5))],
              {1: FIX, 3: FIX, 9: (0, 0, 1, 1, 1, 0)})
    if masses:
        f.masses = {2: (1000.0, 0, 0, 0), 4: (1000.0, 0, 0, 0)}
    return f


@pytest.mark.gold
def test_rigid_diaphragm_lateral_and_torsional_stiffness():
    """侧移刚度 2k，扭转刚度 2(k·a² + GJ/h)，k = 3EI/h³——平面刚性楼板的经典结果。"""
    a, h = 2.0, L
    k = 3 * E * I / h**3
    f = diaphragm(a, h)
    f.nodal_loads[9] = (10e3, 0, 0, 0, 0, 4e3)
    s = solve(f)
    u9 = s.U[f.node_dofs(9)]
    assert u9[0] == pytest.approx(10e3 / (2 * k), rel=1e-10)
    assert u9[5] == pytest.approx(4e3 / (2 * (k * a**2 + G * J / h)), rel=1e-10)
    # 柱顶 y 向位移 = ±a·θz（楼板平面内刚体转动）
    assert s.U[f.node_dofs(4)[1]] == pytest.approx(a * u9[5], rel=1e-10)
    assert s.U[f.node_dofs(2)[1]] == pytest.approx(-a * u9[5], rel=1e-10)
    # 楼板不约束平面外：柱顶竖向位移各自独立（这里无竖向力，为零）
    assert check_equilibrium(f, s)["ok"]


@pytest.mark.gold
def test_rigid_diaphragm_modes():
    """柱无质量、柱顶各一个集中质量：平动 ω² = k/m，扭转 ω² = (k·a² + GJ/h)/(m·a²)。

    从节点上的质量经 Pᵀ·M·P 带着刚臂到参考点，自动成为转动惯量 2m·a²。
    """
    a, h, m = 2.0, L, 1000.0
    k = 3 * E * I / h**3
    res = modal(diaphragm(a, h, masses=True), num_modes=4)
    omega2 = np.sort(res.omega**2)
    assert omega2[0] == pytest.approx(k / m, rel=1e-9)
    assert omega2[1] == pytest.approx(k / m, rel=1e-9)
    assert omega2[2] == pytest.approx((k * a**2 + G * J / h) / (m * a**2), rel=1e-9)


def test_buckling_of_columns_tied_by_diaphragm():
    """两根等柱等载、柱顶平动连在一起：侧移屈曲与单根悬臂柱相同，π²EI/(4h²)，从上方逼近。"""
    h, n, P = L, 8, -1e3
    nodes = [Node(100 + k, -1.0, 0, h * k / n) for k in range(n + 1)]
    nodes += [Node(200 + k, 1.0, 0, h * k / n) for k in range(n + 1)]
    members = [bar(100 + k, 100 + k, 101 + k) for k in range(n)]
    members += [bar(200 + k, 200 + k, 201 + k) for k in range(n)]
    f = build(nodes, members, [Constraint("tie-top", 100 + n, (200 + n,), dofs=(0, 1))],
              {100: FIX, 200: FIX})
    f.nodal_loads[100 + n] = (0, 0, P, 0, 0, 0)
    f.nodal_loads[200 + n] = (0, 0, P, 0, 0, 0)
    lam = buckling(f, num_modes=2).factors[0]
    exact = np.pi**2 * E * I / (4 * h**2) / -P
    assert exact < lam < exact * (1 + 1e-4)


def test_settlement_of_master_support_moves_slaves():
    """主节点是支座且有沉降时，从节点按刚体运动跟着走（经 Q 矩阵）。"""
    f = build([Node(1, 0, 0, 0), Node(2, L, 0, 0), Node(3, 0, 0, 0.5)],
              [bar(1, 1, 2)], [Constraint("arm", 1, (3,))], {1: FIX, 2: (0, 0, 1, 0, 0, 0)})
    f.settlements[1] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.002)      # 绕 z 转 0.002 rad
    s = solve(f)
    u3 = s.U[f.node_dofs(3)]
    # (θ × d)：θ = (0,0,0.002)，d = (0,0,0.5) → 平行，平动为零；转角跟随
    assert u3[5] == pytest.approx(0.002)
    f.settlements[1] = (0.0, 0.0, 0.0, 0.003, 0.0, 0.0)      # 绕 x 转
    s = solve(f)
    u3 = s.U[f.node_dofs(3)]
    assert u3[1] == pytest.approx(-0.003 * 0.5)               # θx·dz 进 −y


def test_pdelta_respects_constraints():
    """P-Δ 迭代也经同一个自由度基：轴力很小时退化为线性解。"""
    f = offset_cantilever()
    lin = solve(f).U
    nl = solve_pdelta(f, increments=2).U
    np.testing.assert_allclose(nl, lin, rtol=1e-9, atol=1e-15)


def test_no_constraints_basis_is_plain_selection():
    """没有约束时自由度基就是原来的选取，不经稀疏乘积——老模型结果逐位不变。"""
    f = build([Node(1, 0, 0, 0), Node(2, L, 0, 0)], [bar(1, 1, 2)], (), {1: FIX})
    basis = dof_basis(f)
    assert basis.P is None and basis.Q is None
    assert list(basis.free) == list(range(6, 12))


@pytest.mark.parametrize("constraints, supports, words", [
    ([Constraint("a", 2, (3,)), Constraint("b", 1, (3,))], {1: FIX}, "同时被约束"),
    ([Constraint("a", 2, (1,))], {1: FIX}, "既有支座"),
    ([Constraint("a", 2, (2,))], {1: FIX}, "同时出现在从节点"),
    ([Constraint("a", 2, (3,)), Constraint("b", 3, (2,))], {1: FIX}, "成环"),
    ([Constraint("a", 7, (3,))], {1: FIX}, "主节点 7 不存在"),
])
def test_over_constraint_is_rejected_by_name(constraints, supports, words):
    """Abaqus 里是 over-constraint 错误的情况，这里在校验阶段点名拒绝，不静默取舍。"""
    f = build([Node(1, 0, 0, 0), Node(2, L, 0, 0), Node(3, L, 0, 0.5)],
              [bar(1, 1, 2)], constraints, supports)
    issues = check_model(f)
    assert any(words in text for text in issues), issues
    with pytest.raises(ValueError, match=words):
        dof_basis(f)


def test_untied_coincident_nodes_still_rejected():
    """没有约束连起来的重合节点仍然拒绝——那是刚度矩阵里断开的“假连接”。"""
    f = build([Node(1, 0, 0, 0), Node(2, L, 0, 0), Node(3, L, 0, 0), Node(4, 2 * L, 0, 0)],
              [bar(1, 1, 2), bar(2, 3, 4)], (), {1: FIX, 4: FIX})
    assert any("坐标重合" in text for text in check_model(f))
