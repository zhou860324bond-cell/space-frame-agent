"""自研 C3D10 内核的解析级验证。"""

from __future__ import annotations

import numpy as np
import pytest

import solid3d as s3


def _unit_tet() -> np.ndarray:
    corners = np.array([[0., 0., 0.], [1., 0., 0.],
                        [0., 1., 0.], [0., 0., 1.]])
    return np.vstack((corners,
                      [(corners[i] + corners[j]) / 2.
                       for i, j in s3.EDGE_PAIRS]))


def test_tet10_shape_functions_form_a_partition_of_unity():
    bary = np.array([0.1, 0.2, 0.3, 0.4])
    N, dN = s3.shape_tet10(bary)
    assert N.sum() == pytest.approx(1.0, abs=1e-14)
    assert dN.sum(axis=0) == pytest.approx([0., 0., 0.], abs=1e-14)


def test_tet10_shape_functions_are_kronecker_at_all_ten_nodes():
    locations = [
        [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
        [0.5, 0.5, 0, 0], [0, 0.5, 0.5, 0], [0.5, 0, 0.5, 0],
        [0.5, 0, 0, 0.5], [0, 0.5, 0, 0.5], [0, 0, 0.5, 0.5],
    ]
    for i, bary in enumerate(locations):
        N, _ = s3.shape_tet10(bary)
        assert N == pytest.approx(np.eye(10)[i], abs=1e-14)


def test_constant_strain_patch_is_recovered_at_every_gauss_point():
    """任意线性位移场在二次单元里必须逐积分点精确还原。"""
    xyz = _unit_tet()
    gradient = np.array([[0.01, 0.02, -0.03],
                         [0.04, -0.02, 0.01],
                         [0.03, 0.05, 0.06]])
    offset = np.array([2.0, -3.0, 4.0])
    ue = (xyz @ gradient.T + offset).ravel()
    expected = np.array([
        gradient[0, 0], gradient[1, 1], gradient[2, 2],
        gradient[0, 1] + gradient[1, 0],
        gradient[1, 2] + gradient[2, 1],
        gradient[0, 2] + gradient[2, 0],
    ])
    for bary in s3.GAUSS_BARYCENTRIC:
        B, det_j = s3._b_matrix(xyz, bary)
        assert det_j == pytest.approx(1.0, abs=1e-12)
        assert B @ ue == pytest.approx(expected, abs=2e-14)


def test_element_stiffness_has_six_rigid_body_modes():
    K = s3.element_stiffness(_unit_tet(), 210000.0, 0.3)
    eig = np.linalg.eigvalsh(K)
    scale = eig[-1]
    assert np.count_nonzero(np.abs(eig) < scale * 1e-10) == 6
    assert eig[6] > 0.0
    assert K == pytest.approx(K.T, abs=1e-10)


def test_distributed_cut_face_forces_reproduce_the_six_component_wrench():
    points = np.array([[2., -1., -1.], [2., 1., -1.],
                       [2., 1., 1.], [2., -1., 1.], [2., 0., 0.]])
    origin = np.array([0.5, 0.2, -0.1])
    force = np.array([10., -20., 30.])
    moment = np.array([40., 50., -60.])
    nodal = s3.equivalent_nodal_wrench(points, force, moment, origin)
    got_force = nodal.sum(axis=0)
    got_moment = np.sum(np.cross(points - origin, nodal), axis=0)
    assert got_force == pytest.approx(force, abs=1e-12)
    assert got_moment == pytest.approx(moment, abs=1e-12)


def test_invalid_material_and_inverted_element_are_refused():
    with pytest.raises(s3.Solid3DError):
        s3.elasticity_matrix(-1.0, 0.3)
    with pytest.raises(s3.Solid3DError):
        s3.elasticity_matrix(1.0, 0.5)
    xyz = _unit_tet().copy()
    xyz[:, 0] *= -1.0
    with pytest.raises(s3.Solid3DError, match="Jacobian"):
        s3.element_stiffness(xyz, 1.0, 0.3)


def test_global_sparse_solve_reproduces_a_manufactured_displacement():
    """用 K·u 构造荷载，约束消元后的解必须还原原位移。"""
    xyz = _unit_tet()
    mesh = s3.SolidMesh(xyz, np.arange(10).reshape((1, 10)))
    K = s3.assemble(mesh, 210000.0, 0.3)
    rng = np.random.default_rng(7)
    expected = rng.normal(scale=1e-4, size=(10, 3))
    expected[:3] = 0.0
    loads = (K @ expected.ravel()).reshape((-1, 3))
    result = s3.solve(mesh, 210000.0, 0.3, [0, 1, 2], loads)
    assert result.displacement == pytest.approx(expected, abs=1e-12)
    assert result.residual_norm < 1e-8


def test_linear_stress_field_extrapolates_exactly_to_all_ten_nodes():
    corner = np.arange(24, dtype=float).reshape((4, 6))
    gauss = s3.GAUSS_BARYCENTRIC @ corner
    nodal = s3.extrapolate_element_nodal_stress(gauss)
    assert nodal[:4] == pytest.approx(corner, abs=1e-12)
    for k, (i, j) in enumerate(s3.EDGE_PAIRS, start=4):
        assert nodal[k] == pytest.approx(0.5 * (corner[i] + corner[j]), abs=1e-12)


def test_one_tet_has_all_ten_nodes_on_its_boundary():
    mesh = s3.SolidMesh(_unit_tet(), np.arange(10).reshape((1, 10)))
    assert s3.boundary_nodes(mesh).tolist() == list(range(10))
