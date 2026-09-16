"""高级梁柱能力的解析与不变量测试。"""

import numpy as np
import pytest

from frame3d import Frame, Material, Member, Node, Section, solve
from modal import member_mode_displacement
from nonlinear import solve_material_nonlinear, solve_pdelta


E = 210e9
NU = 0.3
G = E / (2 * (1 + NU))
FIX = (1, 1, 1, 1, 1, 1)


def cantilever(length=1.0, section=None, member_kwargs=None):
    f = Frame()
    f.nodes[1] = Node(1, 0, 0, 0)
    f.nodes[2] = Node(2, length, 0, 0)
    f.materials["M"] = Material("M", E, NU, 7850)
    f.sections["S"] = section or Section("S", 0.02, 2e-5, 4e-5, 1e-5)
    f.members[1] = Member(1, 1, 2, "S", "M", **(member_kwargs or {}))
    f.supports[1] = FIX
    return f


@pytest.mark.gold
def test_timoshenko_tip_deflection_includes_shear_term():
    length = 0.8
    area_y = 0.012
    sec = Section("S", 0.02, 2e-5, 4e-5, 1e-5, Ay=area_y)
    f = cantilever(length, sec)
    force = -40e3
    f.nodal_loads[2] = (0, 0, force, 0, 0, 0)
    result = solve(f)
    expected = force * length**3 / (3 * E * sec.Iz) + force * length / (G * area_y)
    assert result.U[f.node_dofs(2)[2]] == pytest.approx(expected, rel=1e-11)


def test_missing_shear_area_keeps_euler_bernoulli_result():
    f = cantilever()
    force = -10e3
    f.nodal_loads[2] = (0, 0, force, 0, 0, 0)
    result = solve(f)
    assert result.U[f.node_dofs(2)[2]] == pytest.approx(
        force / (3 * E * f.sections["S"].Iz), rel=1e-11)


def test_rigid_offset_shortens_flexible_span_and_transfers_moment():
    offset = 0.2
    f = cantilever(member_kwargs={"offset_i": (offset, 0, 0)})
    force = -10e3
    f.nodal_loads[2] = (0, 0, force, 0, 0, 0)
    result = solve(f)
    flexible = 1.0 - offset
    expected = force * flexible**3 / (3 * E * f.sections["S"].Iz)
    assert result.U[f.node_dofs(2)[2]] == pytest.approx(expected, rel=1e-10)
    assert abs(result.R[f.node_dofs(1)[4]]) == pytest.approx(abs(force), rel=1e-10)


def test_mode_recovery_uses_rotations_and_keeps_exact_endpoints():
    f = cantilever()
    shapes = np.zeros((f.num_dofs, 1))
    shapes[f.node_dofs(2)[4], 0] = -1.0
    points, displacement = member_mode_displacement(f, shapes, 0, 1, stations=5)
    assert np.allclose(displacement[0], 0.0)
    assert np.allclose(displacement[-1], 0.0)
    assert abs(displacement[2, 2]) > 0.0
    assert np.allclose(points[[0, -1]], [[0, 0, 0], [1, 0, 0]])


def test_pdelta_amplifies_lateral_displacement_under_compression():
    f = cantilever(length=3.0)
    # 局部轴向压力与横向扰动力同时施加。
    f.nodal_loads[2] = (-1.0e6, 0, -2.0e3, 0, 0, 0)
    linear = solve(f).U[f.node_dofs(2)[2]]
    second = solve_pdelta(f, increments=6).U[f.node_dofs(2)[2]]
    assert abs(second) > abs(linear)
    assert second < 0.0


@pytest.mark.gold
def test_bilinear_axial_material_uses_incremental_return_mapping():
    f = cantilever(length=2.0)
    base = f.materials["M"]
    fy = 200e6
    ratio = 0.02
    f.materials["M"] = Material("M", base.E, base.nu, base.density,
                                yield_stress=fy, hardening_ratio=ratio)
    force = 2.1e6  # A=0.02 -> 105 MPa, still elastic; lower A to cross yield
    f.sections["S"] = Section("S", 0.01, 2e-5, 4e-5, 1e-5)
    force = 2.1e6  # stress = 210 MPa
    f.nodal_loads[2] = (force, 0, 0, 0, 0, 0)
    result = solve_material_nonlinear(f, increments=12)
    expected_strain = fy / E + (force / 0.01 - fy) / (ratio * E)
    assert result.U[f.node_dofs(2)[0]] == pytest.approx(
        2.0 * expected_strain, rel=2e-7)
    assert result.analysis["type"] == "material_nonlinear_axial_bilinear"


# ------------------------------------- P-Δ 的精确二阶弹性解

def _meshed_cantilever(n: int, length: float) -> Frame:
    """把悬臂分成 n 个单元。几何刚度阵是近似的，验精确解必须能加密。"""
    f = Frame()
    for index in range(n + 1):
        f.nodes[index + 1] = Node(index + 1, length * index / n, 0, 0)
    f.materials["M"] = Material("M", E, NU, 7850)
    f.sections["S"] = Section("S", 0.02, 2e-5, 4e-5, 1e-5)
    for index in range(n):
        f.members[index + 1] = Member(index + 1, index + 1, index + 2, "S", "M")
    f.supports[1] = FIX
    return f


def _exact_pdelta_tip(axial: float, lateral: float, length: float,
                      ei: float) -> float:
    """悬臂柱顶同时受轴压 P 与横向力 H 时的**精确**二阶弹性挠度。

        δ = H/(P·k) · ( tan(kL) − kL ),   k = √(P/EI)

    P→0 时 tan(kL) − kL → (kL)³/3，退化为 HL³/(3EI)，与线性解一致。
    这个解来自二阶微分方程本身，不是"线性解乘一个放大系数"的近似，
    也不来自本项目的任何一段实现。
    """
    k = np.sqrt(axial / ei)
    return lateral / (axial * k) * (np.tan(k * length) - k * length)


@pytest.mark.gold
def test_pdelta_converges_to_the_exact_second_order_solution():
    """P-Δ 必须收敛到精确二阶解，而不只是"比线性解大"。

    取 P/P_cr ≈ 0.43，二阶效应把挠度放大到 1.757 倍——量级足够大，
    实现里少算一项也躲不过去。实测相对误差：
    1 单元 2.3e-3、2 单元 1.7e-4、4 单元 1.1e-5、8 单元 6.8e-7、16 单元 4.3e-8。
    """
    length, axial, lateral = 3.0, 1.0e6, 2.0e3
    ei = E * 4e-5
    exact = _exact_pdelta_tip(axial, lateral, length, ei)

    # 二阶效应确实显著，否则这条测试等于在验线性解
    assert exact / (lateral * length ** 3 / (3 * ei)) > 1.5

    errors = []
    for n in (2, 4, 8):
        f = _meshed_cantilever(n, length)
        f.nodal_loads[n + 1] = (-axial, 0, -lateral, 0, 0, 0)
        tip = solve_pdelta(f, increments=8).U[f.node_dofs(n + 1)[2]]
        errors.append(abs(abs(tip) - abs(exact)) / abs(exact))

    assert errors[-1] < 1e-5                       # 8 单元贴合精确解
    assert all(a > b for a, b in zip(errors, errors[1:], strict=False)), errors   # 单调收敛


def test_pdelta_reduces_to_the_linear_solution_without_axial_force():
    """轴力撤掉后二阶解必须退回线性解——守的是几何刚度没有被无条件叠加。"""
    length, lateral = 3.0, 2.0e3
    ei = E * 4e-5
    f = _meshed_cantilever(4, length)
    f.nodal_loads[5] = (0.0, 0, -lateral, 0, 0, 0)
    tip = solve_pdelta(f, increments=4).U[f.node_dofs(5)[2]]
    assert abs(tip) == pytest.approx(lateral * length ** 3 / (3 * ei), rel=1e-9)
