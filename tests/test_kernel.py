"""内核验证：每一项都与解析解或静力平衡对拍。

测试名字说明它在防什么坑——改代码时哪一条红了，就知道破坏了什么。
运行： pytest -q     （pytest.ini 已把 src 加进 pythonpath）
"""
import numpy as np
import pytest

from frame3d import (Frame, Material, Member, Node, Section, check_equilibrium,
                     check_model, diagnose_singularity, solve)

E, NU = 210e9, 0.3
G = E / (2 * (1 + NU))
B, H = 0.2, 0.3                      # 矩形截面 b x h
SEC = Section("R200x300", B * H, H * B**3 / 12, B * H**3 / 12, 0.196 * B**3 * H)
MAT = Material("Q355", E, NU)
FIX = (1, 1, 1, 1, 1, 1)
L = 4.0
W = -20e3                            # 均布线荷载，全局 -Z


def build(nodes, members):
    f = Frame()
    for n in nodes:
        f.nodes[n.id] = n
    for m in members:
        f.members[m.id] = m
    f.sections[SEC.name] = SEC
    f.materials[MAT.name] = MAT
    return f


def two_node(**member_kw):
    return build([Node(1, 0, 0, 0), Node(2, L, 0, 0)],
                 [Member(1, 1, 2, SEC.name, MAT.name, **member_kw)])


def cantilever(**member_kw):
    f = two_node(**member_kw)
    f.supports[1] = FIX
    return f


# --------------------------------------------------------------- 基本单元行为

@pytest.mark.gold
def test_tip_deflection_matches_pl3_over_3ei():
    """水平悬臂梁端部竖向力：局部 y 朝上，故用强轴 Iz。"""
    f = cantilever()
    P = -10e3
    f.nodal_loads[2] = (0, 0, P, 0, 0, 0)
    s = solve(f)
    assert s.U[f.node_dofs(2)[2]] == pytest.approx(P * L**3 / (3 * E * SEC.Iz), rel=1e-10)


def test_fixed_end_moment():
    f = cantilever()
    f.nodal_loads[2] = (0, 0, -10e3, 0, 0, 0)
    s = solve(f)
    assert abs(s.R[f.node_dofs(1)[4]]) == pytest.approx(10e3 * L, rel=1e-10)


@pytest.mark.gold
def test_weak_axis_uses_iy():
    """侧向力落在局部 z 方向，应当用弱轴 Iy——守的是局部坐标系没搞反。"""
    f = cantilever()
    f.nodal_loads[2] = (0, -10e3, 0, 0, 0, 0)
    s = solve(f)
    assert s.U[f.node_dofs(2)[1]] == pytest.approx(-10e3 * L**3 / (3 * E * SEC.Iy), rel=1e-10)


@pytest.mark.gold
def test_torsion_angle():
    f = cantilever()
    T = 5e3
    f.nodal_loads[2] = (0, 0, 0, T, 0, 0)
    s = solve(f)
    assert s.U[f.node_dofs(2)[3]] == pytest.approx(T * L / (G * SEC.J), rel=1e-10)


def test_solver_scales_mixed_dof_stiffness_before_factorisation():
    """轴向/弯曲刚度差很多仍是合法结构，不应因原量纲主元比被误报成机构。"""
    f = Frame()
    f.nodes[1] = Node(1, 0, 0, 0)
    f.nodes[2] = Node(2, 1.0, 0, 0)
    f.materials["M"] = Material("M", E, NU)
    # 故意让 EA/L 与 EI/L³ 相差约 1e18；旧的原矩阵主元比检查会拒绝它。
    f.sections["S"] = Section("S", 1e6, 1e-12, 1e-12, 1e-12)
    f.members[1] = Member(1, 1, 2, "S", "M")
    f.supports[1] = FIX
    f.nodal_loads[2] = (100.0, 0.0, -1.0, 0.0, 0.0, 0.0)

    s = solve(f)
    dofs = f.node_dofs(2)
    assert s.U[dofs[0]] == pytest.approx(100.0 / (E * 1e6), rel=1e-9)
    assert s.U[dofs[2]] == pytest.approx(-1.0 / (3.0 * E * 1e-12), rel=1e-9)
    assert check_equilibrium(f, s)["ok"]


def test_vertical_column_reference_vector_fallback():
    """竖直杆件参考向量退化，走的是 local_axes 里的特判分支。"""
    height = 3.6
    f = build([Node(1, 0, 0, 0), Node(2, 0, 0, height)],
              [Member(1, 1, 2, SEC.name, MAT.name)])
    f.supports[1] = FIX
    f.nodal_loads[2] = (12e3, 0, 0, 0, 0, 0)
    s = solve(f)
    assert s.U[f.node_dofs(2)[0]] == pytest.approx(
        12e3 * height**3 / (3 * E * SEC.Iz), rel=1e-10)


# --------------------------------------------------------------- 杆间荷载

def simply_supported(n=8, w=W):
    nodes = [Node(k, k * L / n, 0, 0) for k in range(n + 1)]
    members = [Member(k, k, k + 1, SEC.name, MAT.name) for k in range(n)]
    f = build(nodes, members)
    f.supports[0] = (1, 1, 1, 1, 0, 0)
    f.supports[n] = (0, 1, 1, 1, 0, 0)
    for k in range(n):
        f.member_loads[k] = (0, 0, w)
    return f, n, w


def test_simply_supported_midspan_deflection():
    f, n, w = simply_supported()
    s = solve(f)
    assert s.U[f.node_dofs(n // 2)[2]] == pytest.approx(
        5 * w * L**4 / (384 * E * SEC.Iz), rel=1e-4)


@pytest.mark.gold
def test_simply_supported_reaction():
    f, n, w = simply_supported()
    s = solve(f)
    assert s.R[f.node_dofs(0)[2]] == pytest.approx(-w * L / 2, rel=1e-9)


def test_simply_supported_midspan_moment():
    """守的是内力回算把固端力加回来了——漏掉这一步跨中弯矩会算成 0。"""
    f, n, w = simply_supported()
    s = solve(f)
    assert abs(s.member_forces[n // 2][5]) == pytest.approx(abs(w) * L**2 / 8, rel=2e-3)


# --------------------------------------------------------------- 杆端释放

def fixed_fixed(**member_kw):
    f = two_node(**member_kw)
    f.supports[1] = FIX
    f.supports[2] = FIX
    f.member_loads[1] = (0, 0, W)
    return f


@pytest.mark.gold
def test_fixed_fixed_end_moment_is_wl2_over_12():
    """无释放的对照组：两端固接梁承受均布荷载，固端弯矩 wL²/12。"""
    s = solve(fixed_fixed())
    f = fixed_fixed()
    assert abs(s.R[f.node_dofs(1)[4]]) == pytest.approx(abs(W) * L**2 / 12, rel=1e-9)


def test_release_turns_fixed_fixed_into_propped_cantilever():
    """j 端释放绕 z 弯矩后，两端固接梁变成一端固定一端简支，固端弯矩 wL²/8。

    节点 2 在整体上仍是固接的，是**杆件**不传弯矩——这正是释放与支座条件的区别。
    """
    f = fixed_fixed(releases_j=("rz",))
    s = solve(f)
    assert abs(s.R[f.node_dofs(1)[4]]) == pytest.approx(abs(W) * L**2 / 8, rel=1e-9)


@pytest.mark.gold
def test_release_reaction_split_is_five_eighths_three_eighths():
    f = fixed_fixed(releases_j=("rz",))
    s = solve(f)
    assert s.R[f.node_dofs(1)[2]] == pytest.approx(-5 * W * L / 8, rel=1e-9)
    assert s.R[f.node_dofs(2)[2]] == pytest.approx(-3 * W * L / 8, rel=1e-9)


def test_released_end_transmits_no_moment():
    f = fixed_fixed(releases_j=("rz",))
    s = solve(f)
    scale = abs(W) * L**2
    assert abs(s.member_forces[1][11]) < 1e-9 * scale
    assert abs(s.R[f.node_dofs(2)[4]]) < 1e-9 * scale


def test_release_mechanism_is_reported_before_solving():
    """两端同时释放扭转会让单元散架，check_model 就该拦下，不必等求解报错。"""
    f = cantilever(releases_i=("rx",), releases_j=("rx",))
    issues = check_model(f)
    assert any("机构" in msg for msg in issues)


def test_release_mechanism_raises_on_solve():
    f = cantilever(releases_i=("ux",), releases_j=("ux",))
    with pytest.raises(ValueError, match="机构"):
        solve(f)


def test_invalid_release_name_is_reported():
    f = cantilever(releases_j=("moment",))
    assert any("moment" in msg for msg in check_model(f))


# --------------------------------------------------------------- 多工况与组合

def two_case_cantilever():
    f = cantilever()
    f.case("DL").member_loads[1] = (0, 0, -10e3)
    f.case("LL").nodal_loads[2] = (0, 0, -5e3, 0, 0, 0)
    f.combos["C1"] = {"DL": 1.3, "LL": 1.5}
    return f


def test_each_case_matches_its_analytic_solution():
    f = two_case_cantilever()
    s = solve(f)
    d = f.node_dofs(2)[2]
    assert s["DL"].U[d] == pytest.approx(-10e3 * L**4 / (8 * E * SEC.Iz), rel=1e-10)
    assert s["LL"].U[d] == pytest.approx(-5e3 * L**3 / (3 * E * SEC.Iz), rel=1e-10)


def test_combo_is_exact_superposition():
    """线性体系下组合就是工况结果的线性叠加，不必重新求解。"""
    f = two_case_cantilever()
    s = solve(f)
    d = f.node_dofs(2)[2]
    assert s["C1"].U[d] == pytest.approx(1.3 * s["DL"].U[d] + 1.5 * s["LL"].U[d], rel=1e-12)


def test_combo_member_forces_superpose():
    f = two_case_cantilever()
    s = solve(f)
    expected = 1.3 * s["DL"].member_forces[1] + 1.5 * s["LL"].member_forces[1]
    assert s["C1"].member_forces[1] == pytest.approx(expected, rel=1e-12)


def test_combo_satisfies_equilibrium():
    f = two_case_cantilever()
    report = check_equilibrium(f, solve(f), "C1")
    assert report["ok"], report


def test_combo_referencing_unknown_case_is_reported():
    f = two_case_cantilever()
    f.combos["bad"] = {"SNOW": 1.0}
    assert any("SNOW" in msg for msg in check_model(f))


def test_single_case_accessors_still_work():
    """单工况模型仍可直接用 sol.U / sol.R，不必写工况名。"""
    f = cantilever()
    f.nodal_loads[2] = (0, 0, -10e3, 0, 0, 0)
    s = solve(f)
    assert s.U.shape == (f.num_dofs,)
    assert s.member_forces[1].shape == (12,)


# --------------------------------------------------------------- 自校验

def portal():
    f = build([Node(1, 0, 0, 0), Node(2, 0, 0, 3.6),
               Node(3, 6.0, 0, 3.6), Node(4, 6.0, 0, 0)],
              [Member(1, 1, 2, SEC.name, MAT.name),
               Member(2, 2, 3, SEC.name, MAT.name),
               Member(3, 4, 3, SEC.name, MAT.name)])
    f.supports[1] = FIX
    f.supports[4] = FIX
    f.member_loads[2] = (0, 0, -15e3)
    f.nodal_loads[2] = (8e3, 0, 0, 0, 0, 0)
    return f


def test_portal_frame_equilibrium():
    f = portal()
    report = check_equilibrium(f, solve(f))
    assert report["ok"], report


def test_portal_frame_model_is_clean():
    assert check_model(portal()) == []


def test_singularity_diagnosis_locates_missing_restraint():
    """故意漏掉 Y 向约束，诊断应当定位到 Y 向平动这个刚体模态。"""
    f = cantilever()
    f.supports[1] = (1, 0, 1, 1, 1, 1)
    f.nodal_loads[2] = (0, 0, -1e3, 0, 0, 0)
    modes = diagnose_singularity(f)
    assert len(modes) == 1
    assert {p["dof"] for p in modes[0]["participants"]} == {1}


def test_check_model_reports_dangling_node():
    f = cantilever()
    f.nodes[99] = Node(99, 9, 9, 9)
    assert any("99" in msg for msg in check_model(f))


def test_check_model_reports_load_on_missing_member():
    f = cantilever()
    f.member_loads[77] = (0, 0, -1e3)
    assert any("77" in msg for msg in check_model(f))


def test_singular_system_is_caught_even_when_load_avoids_the_null_direction():
    """奇异矩阵不一定让分解报错——荷载不激发那个刚体方向时它会安静地算出错误答案。

    这里抽掉 Y 向约束、只加竖向荷载，主元检查必须把它拦下来。
    """
    f = cantilever()
    f.supports[1] = (1, 0, 1, 1, 1, 1)
    f.nodal_loads[2] = (0, 0, -10e3, 0, 0, 0)
    with pytest.raises(np.linalg.LinAlgError, match="奇异"):
        solve(f)


@pytest.mark.parametrize("P, sign", [(+100e3, +1), (-100e3, -1)])
def test_axial_sign_convention_is_tension_positive(P, sign):
    """以受拉为正的轴力是 j 端分量 f[6]，不是 f[0]——两者符号相反。

    绘图时用错分量会把受压的柱子画成受拉，这个错误只有画出来才看得见。
    """
    f = cantilever()
    f.nodal_loads[2] = (P, 0, 0, 0, 0, 0)
    forces = solve(f).member_forces[1]
    assert forces[6] == pytest.approx(sign * abs(P), rel=1e-10)
    assert forces[0] == pytest.approx(-sign * abs(P), rel=1e-10)


def test_gravity_puts_columns_in_compression():
    """重力作用下柱子必须是受压（N < 0），这是符号约定最实际的检验。"""
    height, span = 3.6, 6.0
    f = build([Node(1, 0, 0, 0), Node(2, 0, 0, height),
               Node(3, span, 0, height), Node(4, span, 0, 0)],
              [Member(1, 1, 2, SEC.name, MAT.name),
               Member(2, 2, 3, SEC.name, MAT.name),
               Member(3, 4, 3, SEC.name, MAT.name)])
    f.supports[1] = FIX
    f.supports[4] = FIX
    f.member_loads[2] = (0, 0, -20e3)
    s = solve(f)
    assert s.member_forces[1][6] < 0, "左柱应受压"
    assert s.member_forces[3][6] < 0, "右柱应受压"
