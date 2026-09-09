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
