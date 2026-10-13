"""梁—实体位移驱动防止位置映射、截面锁死、刚体假应力与漏载。"""

import numpy as np
import pytest
from scipy.sparse import eye

import joint_transfer as jt
import sections
import solid3d
import solid_joint
from agent import Session
from test_solid3d import _unit_tet
from test_solid_joint import _l_joint


def _cantilever(*, units="N-m-Pa", shear=False, reverse=False):
    s = Session(model={"units": units})
    scale = 1000.0 if units == "N-mm-MPa" else 1.0
    sec = sections.solid_circle("R", 0.2 * scale)
    if shear:
        sec["Ay"] = sec["Az"] = 0.8 * sec["A"]
    assert s.add_nodes([[0, 0, 0], [2 * scale, 0, 0], [4 * scale, 0, 0]]).ok
    assert s.define_materials_and_sections(
        [{"name": "steel", "E": 210000.0 if scale > 1 else 2.1e11, "nu": 0.3}],
        [sec]).ok
    pairs = [[2, 1], [3, 2]] if reverse else [[1, 2], [2, 3]]
    assert s.add_members(pairs, section="R", material="steel").ok
    assert s.set_supports([1], fix=[1] * 6).ok
    assert s.set_load_cases([{"name": "LC"}]).ok
    assert s.set_nodal_load(3, [0, -1000, 0, 0, 0, 0], case_name="LC").ok
    assert s.solve_model().ok
    return s


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("shear", [False, True])
def test_cut_motion_matches_cantilever_bending_and_independent_rotation(reverse, shear):
    """防止用节点运动替代切面运动、反向杆符号错误及剪切转角混入截面转角。"""
    s = _cantilever(shear=shear, reverse=reverse)
    spec = solid_joint.prepare_joint_spec(s, 2, loading_method="displacement")
    inertia = np.pi * 0.2**4 / 64
    for arm in spec.arms:
        x = 2.0 + arm.direction[0] * arm.length_mm / 1000
        expected = -1000 * x*x * (12 - x) / (6 * 2.1e11 * inertia)
        if shear:
            expected -= 1000 * x / ((2.1e11 / 2.6) * 0.8 * np.pi * 0.2**2 / 4)
        theta = -1000 * x * (8 - x) / (2 * 2.1e11 * inertia)
        assert arm.translation_mm == pytest.approx([0, expected * 1000, 0], abs=1e-9)
        assert arm.rotation_rad == pytest.approx([0, 0, theta], abs=1e-12)
        assert arm.source_element in (1, 2)


def test_cut_motion_is_identical_in_metre_and_millimetre_models():
    """防止转角误乘1000或实体边界平移漏做单位转换。"""
    specs = [solid_joint.prepare_joint_spec(_cantilever(units=u), 2,
                                           loading_method="displacement")
             for u in ("N-m-Pa", "N-mm-MPa")]
    for a, b in zip(specs[0].arms, specs[1].arms, strict=True):
        assert a.translation_mm == pytest.approx(b.translation_mm, abs=1e-9)
        assert a.rotation_rad == pytest.approx(b.rotation_rad, abs=1e-12)


@pytest.mark.parametrize("reverse", [False, True])
def test_cut_motion_preserves_axial_torsion_and_both_bending_axes(reverse):
    """防止六分量映射只验证一个弯曲方向，漏掉轴向、扭转或另一弯曲符号。"""
    s = _cantilever(reverse=reverse)
    assert s.set_nodal_load(3, [1000, 0, 0, 100, 200, 300], case_name="LC").ok
    assert s.solve_model().ok
    spec = solid_joint.prepare_joint_spec(s, 2, loading_method="displacement")
    area, inertia = np.pi * 0.2**2 / 4, np.pi * 0.2**4 / 64
    for arm in spec.arms:
        x = 2 + arm.direction[0]*arm.length_mm/1000
        u = [1000*x/(2.1e11*area), 300*x*x/(2*2.1e11*inertia),
             -200*x*x/(2*2.1e11*inertia)]
        theta = [100*x/((2.1e11/2.6)*2*inertia),
                 200*x/(2.1e11*inertia), 300*x/(2.1e11*inertia)]
        assert arm.translation_mm == pytest.approx(np.asarray(u)*1000, abs=1e-9)
        assert arm.rotation_rad == pytest.approx(theta, abs=1e-12)


def test_cut_motion_uses_split_analysis_elements_and_load_combinations():
    """防止自动剖分后仍按物理杆端插值，或组合结果漏恢复基础工况运动。"""
    s = _cantilever()
    assert s.set_member_span_load(2, "point", [0, -200, 0], a=1.0, case_name="LC").ok
    assert s.set_load_cases(s.model["load_cases"],
                            combos=[{"name": "C", "factors": {"LC": 1.5}}]).ok
    assert s.solve_model().ok
    for arm in solid_joint.prepare_joint_spec(s, 2, case="LC",
                                              loading_method="displacement").arms:
        x = 2 + arm.direction[0]*arm.length_mm/1000
        expected = -(1000*x*x*(12-x) + 200*x*x*(9-x))/(6*2.1e11*np.pi*0.2**4/64)
        assert arm.translation_mm[1] == pytest.approx(expected*1000, abs=1e-9)
    base = solid_joint.prepare_joint_spec(s, 2, case="LC", loading_method="displacement")
    combo = solid_joint.prepare_joint_spec(s, 2, case="C", loading_method="displacement")
    for a, b in zip(base.arms, combo.arms, strict=True):
        assert b.translation_mm == pytest.approx(np.asarray(a.translation_mm)*1.5)
        assert b.rotation_rad == pytest.approx(np.asarray(a.rotation_rad)*1.5)


def test_distributing_operator_preserves_motion_and_does_not_lock_warping():
    """防止分布耦合退化为刚性平截面约束，或不均匀面积导致刚体运动失真。"""
    p = np.array([[0, -2, -1], [0, 3, -1], [0, 2, 4], [0, -1, 2]], dtype=float)
    op, radius = jt.distributing_operator(p, np.zeros(3), [1, 2, 3, 4])
    u, theta = np.array([1, 2, 3]), np.array([0.02, -0.01, 0.03])
    motion = u + np.cross(theta, p)
    assert op @ motion.ravel() == pytest.approx(np.r_[u, radius * theta])
    rng = np.random.default_rng(31)
    field = rng.normal(size=p.shape)
    rigid = op @ field.ravel()
    warp = field - rigid[:3] - np.cross(rigid[3:] / radius, p)
    assert np.linalg.norm(warp) > 0.1
    assert op @ warp.ravel() == pytest.approx(np.zeros(6), abs=1e-14)


def test_distributing_operator_rejects_collinear_cut_nodes():
    """防止不能传递扭转的共线切面被当作完整六分量边界。"""
    with pytest.raises(solid3d.Solid3DError, match="共线"):
        jt.distributing_operator([[0, 0, 0], [0, 1, 0], [0, 2, 0]], [0, 0, 0], [1]*3)


def test_nonzero_constraint_rhs_recovers_constant_strain_patch():
    """防止非零边界运动未进入方程右端，导致位移驱动无载时返回零应力。"""
    xyz = _unit_tet()
    mesh = solid3d.SolidMesh(xyz, np.arange(10).reshape(1, 10))
    strain = 0.002
    prescribed = xyz * strain
    r = solid3d.solve(mesh, 210000, 0.3, [], np.zeros_like(xyz),
                      linear_constraints=(eye(30), prescribed.ravel()))
    assert r.displacement == pytest.approx(prescribed, abs=1e-12)
    expected = 210000 / (1 - 2*0.3) * strain
    assert r.gauss_stress[:, :, :3] == pytest.approx(np.full((1, 4, 3), expected))
    assert r.gauss_stress[:, :, 3:] == pytest.approx(np.zeros((1, 4, 3)), abs=1e-9)
    assert r.reaction.sum(axis=0) == pytest.approx(np.zeros(3), abs=1e-9)
    assert r.residual_norm < 1e-9


def test_average_rigid_body_motion_produces_no_solid_stress():
    """防止切面平均转角和平移驱动产生虚假应力，且验证约束消除刚体自由度。"""
    xyz = _unit_tet()
    mesh = solid3d.SolidMesh(xyz, np.arange(10).reshape(1, 10))
    op, radius = jt.distributing_operator(xyz, [0, 0, 0], np.ones(10))
    u, theta = np.array([2, -3, 4]), np.array([0.01, -0.02, 0.03])
    r = solid3d.solve(mesh, 210000, 0.3, [], np.zeros_like(xyz),
                      linear_constraints=(op, np.r_[u, radius*theta]))
    assert r.displacement == pytest.approx(u + np.cross(theta, xyz), abs=1e-10)
    assert r.element_mises.max() < 1e-7


def test_area_weights_integrate_a_quadratic_tetrahedron_cut_face():
    """防止二次面中点漏权或节点密集区被赋予额外加载权重。"""
    mesh = solid3d.SolidMesh(_unit_tet(), np.arange(10).reshape(1, 10))
    ids = np.array([0, 1, 2, 4, 5, 6])
    w = jt.cut_area_weights(mesh, ids)
    assert w.sum() == pytest.approx(0.5)
    assert np.all(w > 0)
    assert w @ mesh.nodes[ids] / w.sum() == pytest.approx([1/3, 1/3, 0])


def test_direct_joint_load_cannot_be_silently_omitted_in_displacement_mode():
    """防止直接集中力仍被拒绝或漏载；未重建的内部支承仍应明确阻止。"""
    result = _l_joint(load_at_joint=True).analyze_joint_solid(2, dry_run=True)
    assert result.ok
    assert result.payload['local_loads']
    result = _l_joint(load_at_joint=True).analyze_joint_solid(
        2, dry_run=True, loading_method="force")
    assert result.ok and result.payload["loading_method"] == "force"
    supported = _cantilever()
    assert supported.set_supports([2], fix=[1]*6).ok
    assert supported.solve_model().ok
    result = supported.analyze_joint_solid(2, dry_run=True)
    assert not result.ok and "内部支承" in result.payload["error"]


def test_distributed_load_inside_local_domain_is_explicitly_rejected():
    """防止重建均布荷载后仍以存在外载为理由阻止本可求解的节点。"""
    s = _cantilever()
    s.frame.load_cases["LC"].member_loads[1] = (0, -100, 0)
    spec = solid_joint.prepare_joint_spec(s, 2)
    jt.validate_local_sources(s, 2, spec.arms, "LC")
    from joint_loading import collect_local_loads, local_load_resultant
    force, _moment = local_load_resultant(collect_local_loads(s, 2, spec.arms, 'LC'))
    assert force[1] == pytest.approx(-100*spec.arms[0].length_mm/1000)


def test_abaqus_displacement_script_uses_deformable_coupling_and_all_cut_motions():
    """防止Abaqus后端仍固定锚固臂，或把新位移驱动写成刚性耦合。"""
    spec = solid_joint.prepare_joint_spec(_cantilever(), 2, loading_method="displacement")
    script = solid_joint._script_text(spec, [80, 60, 40], "result.json")
    assert "couplingType=(DISTRIBUTING" in script
    assert "model.DisplacementBC(name='DRIVE_" in script
    assert "ur1=theta[0],ur2=theta[1],ur3=theta[2]" in script
    assert "'translation_mm'" in script
    assert "'rotation_rad'" in script
    script.encode("ascii")
    compile(script, "joint_transfer.py", "exec")


def test_native_displacement_result_is_independent_of_anchor_and_balances(tmp_path, monkeypatch):
    """真实C3D10求解防止位移模式仍依赖锚固臂，检查运动、力和矩平衡。"""
    pytest.importorskip("gmsh")
    import native_joint
    monkeypatch.setattr(native_joint, "_write_vtu_and_png", lambda *args: (None, None))
    s = _l_joint(section=sections.solid_circle("R219", 0.219))
    results = [native_joint.run_native_joint_analysis(
        s, 2, anchor_member=a, mesh_sizes_mm=[80], output_dir=tmp_path / str(a))
        for a in (1, 2)]
    for result in results:
        row = result["meshes"][0]
        assert result["loading_method"] == "displacement"
        assert row["cut_motion_error_mm"] < 1e-9
        force_scale = max(np.linalg.norm(a.force_n) for a in
                          solid_joint.prepare_joint_spec(s, 2).arms)
        assert np.linalg.norm(row["force_balance_n"]) < 1e-8*max(force_scale, 1)
        moment_scale = max(np.linalg.norm(a.moment_nmm) for a in
                           solid_joint.prepare_joint_spec(s, 2).arms)
        assert np.linalg.norm(row["moment_balance_nmm"]) < 1e-8*max(moment_scale, 1)
        assert row["free_residual_norm_n"] < 1e-8*max(force_scale, 1)
        assert set(row["cut_transfer_check"]["cuts"]) == {"1", "2"}
    assert results[0]["meshes"][0]["max_mises_mpa"] == pytest.approx(
        results[1]["meshes"][0]["max_mises_mpa"], rel=1e-10)


@pytest.mark.parametrize("shear", [False, True])
def test_native_transfer_check_exposes_global_local_shear_stiffness_mismatch(
        shear, tmp_path, monkeypatch):
    """防止切面运动和平衡通过就宣称应力可靠，短实体梁需与整体剪切刚度匹配。"""
    pytest.importorskip("gmsh")
    import native_joint
    monkeypatch.setattr(native_joint, "_write_vtu_and_png", lambda *args: (None, None))
    result = native_joint.run_native_joint_analysis(
        _cantilever(shear=shear), 2, mesh_sizes_mm=[60], output_dir=tmp_path,
        geometry_mode="intersection")
    check = result["meshes"][0]["cut_transfer_check"]
    assert check["within_tolerance"] is shear
    assert any("差异超过5%" in w for w in result["warnings"]) is (not shear)
