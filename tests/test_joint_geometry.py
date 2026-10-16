"""真实截面、连接补齐、网格和连续应力显示的回归。"""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

import joint_geometry
import joint_blend
import native_joint
import section_geometry as sg
import sections
import solid3d
import solid_joint
from test_solid3d import _unit_tet
from test_solid_joint import _l_joint


def _teaching_column_joint():
    from agent import Session
    path = Path(__file__).resolve().parents[1]/'examples/空间刚架_简单整体复杂实体.json'
    session = Session(model=json.loads(path.read_text(encoding='utf-8')))
    assert session.solve_model().ok
    return solid_joint.prepare_joint_spec(session,5,case='W_侧向',geometry_mode='transition')


@pytest.mark.parametrize('point, expected', [
    ([90.,80.,-40.],1), ([-90.,80.,-40.],1),
    ([50.,-40.,90.],0), ([150.,30.,30.],0)])
def test_column_top_preserves_flange_material_and_does_not_extend_beams_into_voids(point,expected):
    """防止教学柱顶被角平分面削掉柱翼缘，或把梁向后延拓填进柱截面空隙。"""
    gmsh=pytest.importorskip('gmsh')
    spec=_teaching_column_joint()
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber('General.Terminal',0)
        gmsh.model.add('column_top')
        record=joint_geometry.build_occ_joint(gmsh,spec)
        volume=gmsh.model.getEntities(3)
        assert len(volume)==1
        assert gmsh.model.isInside(3,volume[0][1],point)==expected
        assert record['algorithm']=='column-host+constrained-rolling-ball'
        assert record['template']=='monolithic-column-top'
        assert record['fillet_status']=='rounded'
        assert record['removed_material_mm3'] <= record['boolean_volume_tolerance_mm3']
        assert record['added_material_outside_transition_mm3'] <= record['boolean_volume_tolerance_mm3']
        assert record['host_member_id']==1
        assert record['beam_start_mm']=={'5':100.,'7':6.}
    finally:
        gmsh.finalize()


def test_column_host_requires_the_full_beam_face_to_fit_the_column_material():
    """防止仅按梁中心线判接触，实际宽翼缘悬空却被自动声明为完整梁柱连接。"""
    spec=_teaching_column_joint()
    beam=spec.arms[1]
    wide=sg.profile(sg.descriptor('i',h=200.,b=500.,tw=10.,tf=14.))
    spec=replace(spec,arms=(spec.arms[0],replace(beam,profile_mm=wide),spec.arms[2]))
    with pytest.raises(ValueError,match='端面超出柱材料面'):
        joint_blend.column_host_plan(spec)


def test_column_host_mesh_keeps_three_original_cut_sections_and_areas():
    """防止柱顶修正虽外观正常，却缩窄远端H型切面并改变位移载荷传递面积。"""
    pytest.importorskip('gmsh')
    from joint_transfer import cut_area_weights
    spec=_teaching_column_joint()
    mesh,cuts=native_joint.generate_joint_mesh(spec,60.,9.)
    assert mesh.geometry_record['algorithm']=='column-host+constrained-rolling-ball'
    for arm in spec.arms:
        assert cut_area_weights(mesh,cuts[arm.member_id]).sum()==pytest.approx(arm.profile_mm['area'],rel=1e-6)
        cut=mesh.nodes[cuts[arm.member_id]]
        yz=np.column_stack((cut@arm.axis_y,cut@arm.axis_z))
        outer=np.asarray(arm.profile_mm['outer'])
        assert yz.min(axis=0)==pytest.approx(outer.min(axis=0),abs=1e-5)
        assert yz.max(axis=0)==pytest.approx(outer.max(axis=0),abs=1e-5)


@pytest.mark.parametrize('change', ['remove_material', 'extend_to_cut', 'local_addition'])
def test_continuous_blend_rejects_material_loss_and_nonlocal_growth(change):
    """防止圆角虽成功却削掉柱材料，或沿边链传播到远端切面而改变载荷传递。"""
    gmsh=pytest.importorskip('gmsh')
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber('General.Terminal',0)
        gmsh.model.add('blend_validation')
        occ=gmsh.model.occ
        original=[(3,occ.addBox(0.,0.,0.,1.,1.,1.))]
        if change=='remove_material':
            rounded=[(3,occ.addBox(.1,0.,0.,.9,1.,1.))]
        elif change=='extend_to_cut':
            rounded=[(3,occ.addBox(0.,0.,0.,4.,1.,1.))]
        else:
            rounded=[(3,occ.addBox(0.,0.,0.,1.1,1.,1.))]
        if change=='local_addition':
            record=joint_blend._validate_additive_blend(occ,original,rounded,2.)
            assert record['removed_material_mm3']==pytest.approx(0.,abs=1e-8)
            assert record['added_material_outside_transition_mm3']==pytest.approx(0.,abs=1e-8)
        else:
            with pytest.raises(ValueError,match='原有材料或超出局部过渡区'):
                joint_blend._validate_additive_blend(occ,original,rounded,2.)
        # 验证步骤只删除临时差集，不破坏调用方保留的CAD。
        assert occ.getMass(*original[0])==pytest.approx(1.)
    finally:
        gmsh.finalize()


SHAPES = [sg.descriptor("rectangle", b=.12, h=.18), sg.descriptor("box", b=.12, h=.18, t=.02),
          sg.descriptor("circle", d=.16), sg.descriptor("tube", d=.16, t=.02),
          sg.descriptor("i", b=.12, h=.18, tw=.02, tf=.025),
          sg.descriptor("channel", b=.12, h=.18, tw=.02, tf=.025),
          sg.descriptor("angle", b=.12, t=.02), sg.descriptor("tee", b=.12, h=.18, tw=.02, tf=.025)]


def _section(geometry):
    p = sg.profile(geometry)
    return {"name": "G", "A": p["area"], "Iy": 2e-5, "Iz": 4e-5, "J": 3e-5,
            "geometry": geometry}


@pytest.mark.parametrize("geometry", SHAPES, ids=[g["type"] for g in SHAPES])
def test_all_desktop_profiles_preserve_geometry_and_material_centroid(geometry):
    """防止实体从A/I猜形状，或槽钢、角钢、T形心偏移未用于实体定位。"""
    p = sg.profile(geometry)
    area, center = sg._area_centroid(p["outer"])
    moment = area*center
    for hole in p["holes"]:
        a, c = sg._area_centroid(hole)
        area -= a
        moment -= a*c
    assert np.linalg.norm(moment) < 1e-12
    assert p["feature_size"] > 0
    probe = sg.material_probe(p)
    assert sg.contains(p["outer"], probe)
    assert not any(sg.contains(h, probe) for h in p["holes"])
    s = _l_joint(section=_section(geometry))
    spec = solid_joint.prepare_joint_spec(s, 2, geometry_mode="transition")
    assert all(a.profile_mm["type"] == geometry["type"] for a in spec.arms)


@pytest.mark.parametrize("geometry", SHAPES, ids=[g["type"] for g in SHAPES])
def test_all_profile_shapes_mesh_with_original_cut_material_area(geometry):
    """真实Gmsh网格防止非圆截面挤出错误，或连接包络扩张到切面、丢掉孔洞。"""
    pytest.importorskip("gmsh")
    from joint_transfer import cut_area_weights
    spec = solid_joint.prepare_joint_spec(_l_joint(section=_section(geometry)), 2,
                                          geometry_mode="transition")
    mesh, cuts = native_joint.generate_joint_mesh(spec, 45, 20)
    assert len(mesh.elements) > 0
    for arm in spec.arms:
        assert cut_area_weights(mesh, cuts[arm.member_id]).sum() == pytest.approx(
            sg.profile(geometry)["area"]*1e6, rel=.04)
        if geometry['type'] not in ('circle','tube'):
            coordinates = mesh.nodes[cuts[arm.member_id]]
            yz = np.column_stack((coordinates @ arm.axis_y,coordinates @ arm.axis_z))
            expected = np.asarray(arm.profile_mm['outer'])
            assert yz.min(axis=0) == pytest.approx(expected.min(axis=0),abs=1e-4)
            assert yz.max(axis=0) == pytest.approx(expected.max(axis=0),abs=1e-4)


def test_transition_fills_only_joint_and_adds_material_to_rectangular_corner():
    """防止补齐算法变成填满整根杆件，或只改显示而没有生成连接体积。"""
    gmsh = pytest.importorskip("gmsh")
    spec = solid_joint.prepare_joint_spec(_l_joint(section=sections.rectangle("R", .12, .18)), 2)
    volumes = []
    for mode in ("intersection", "miter"):
        gmsh.initialize(interruptible=False)
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add(mode)
            record = joint_geometry.build_occ_joint(gmsh, replace(spec, geometry_mode=mode))
            volumes.append(record["volume_mm3"])
            assert record["volumes"] == 1
            assert record.get("fillet_region_radius_mm", 0) < min(a.length_mm for a in spec.arms)
        finally:
            gmsh.finalize()
    assert volumes[1] > volumes[0]*1.01


def test_polygon_with_unequal_hole_vertex_counts_survives_unit_conversion():
    """防止自定义截面多个不同顶点数孔洞在米/毫米转换时损坏。"""
    g = {"type": "polygon", "dimensions": {"feature_size": .1}, "outer": [[0,0],[1,0],[1,1],[0,1]],
         "holes": [[[.1,.1],[.2,.1],[.1,.2]], [[.6,.6],[.7,.6],[.7,.7],[.6,.7]]]}
    p = sg.profile(g)
    other = sg.profile(sg.scale_geometry(g, 1000))
    assert other["area"] == pytest.approx(p["area"]*1e6)
    assert other["feature_size"] == pytest.approx(p["feature_size"]*1000)
    assert len(other["holes"][0]) == 3 and len(other["holes"][1]) == 4


def test_self_intersecting_polygon_is_refused_before_cad():
    """防止自交轮廓进入OpenCASCADE，产生错误体积或难以诊断的网格失败。"""
    with pytest.raises(ValueError, match="自交"):
        sg.profile({"type":"polygon", "outer":[[0,0],[1,1],[0,1],[1,0]]})


@pytest.mark.parametrize("holes", [
    [[[.1,.1],[.8,.1],[.8,.8],[.1,.8]], [[.2,.2],[.3,.2],[.2,.3]]],
    [[[.1,.1],[.6,.1],[.1,.6]], [[.4,.1],[.8,.1],[.4,.6]]],
    [[[0,.1],[.2,.1],[.2,.2],[0,.2]]],
])
def test_invalid_polygon_hole_relationships_are_refused(holes):
    """防止嵌套、重叠或接触外边界的孔洞被CAD布尔运算静默改成另一截面。"""
    with pytest.raises(ValueError, match="孔洞"):
        sg.profile({"type":"polygon", "outer":[[0,0],[1,0],[1,1],[0,1]], "holes": holes})


def test_nodal_tensor_average_is_display_only_and_recovers_constant_stress():
    """防止平滑色块改写原单元峰值，或先平均Mises而非应力张量。"""
    mesh = solid3d.SolidMesh(_unit_tet(), np.arange(10).reshape(1,10))
    stress = np.tile([100., 0., 0., 0., 0., 0.], (1,4,1))
    saved = stress.copy()
    vm, principal = solid3d.nodal_display_stress(mesh, stress)
    assert vm == pytest.approx(np.full(10,100))
    assert principal == pytest.approx(np.full(10,100))
    assert stress == pytest.approx(saved)


def test_step_geometry_script_keeps_true_profile_probe_and_refuses_fake_hotspot():
    """防止Abaqus新增几何仍使用圆截面加载点或旧相贯线热点路径。"""
    spec = solid_joint.prepare_joint_spec(_l_joint(section=sections.rectangle("R",.12,.18)), 2,
                                          geometry_mode="transition")
    script = solid_joint._script_text(spec, [18,14,10], "out.json")
    assert "mdb.openStep('joint_geometry.step'" in script
    assert "yz = arm['probe_yz_mm']" in script
    assert "def surface_samples(odb, frame, size):\n    return {}" in script
    script.encode("ascii")
    compile(script, "generic_joint.py", "exec")


@pytest.mark.parametrize("kind", ["i", "tube", "rectangle"])
def test_g2_sweep_keeps_straight_end_tangents_and_zero_endpoint_curvature(kind):
    """防止所谓平滑仅改变显示，真实CAD曲线在直杆接口仍存在折角或曲率跳变。"""
    geometry = next(g for g in SHAPES if g['type'] == kind)
    spec = solid_joint.prepare_joint_spec(_l_joint(section=_section(geometry)), 2)
    plan, reason = joint_blend.sweep_plan(spec)
    assert reason is None
    control = np.asarray(plan['control_points_mm'])
    assert np.linalg.norm(np.cross(control[1]-control[0], spec.arms[0].direction)) < 1e-9
    assert np.linalg.norm(np.cross(control[-1]-control[-2], spec.arms[1].direction)) < 1e-9
    assert joint_blend.curvature(control, [0., 1.]) == pytest.approx([0., 0.], abs=1e-12)
    radius = 1/joint_blend.curvature(control, np.linspace(0,1,10001)).max()
    assert radius >= .999*1.3*max(a.outer_diameter_mm for a in spec.arms)/2
    assert plan['transition_length_mm'] < .8*min(a.length_mm for a in spec.arms)


def test_g2_h_profile_keeps_flange_void_that_the_convex_hull_filled():
    """复现用户指出的H节点大块填充：新CAD须保留翼缘间凹槽，不能只修饰云图。"""
    gmsh = pytest.importorskip('gmsh')
    spec = solid_joint.prepare_joint_spec(_l_joint(section=sections.i_section('H',.18,.12,.02,.025)),2)
    occupancy = []
    for mode in ('envelope','transition'):
        gmsh.initialize(interruptible=False)
        try:
            gmsh.option.setNumber('General.Terminal',0)
            gmsh.model.add(mode)
            record = joint_geometry.build_occ_joint(gmsh, replace(spec,geometry_mode=mode))
            volume = gmsh.model.getEntities(3)[0][1]
            occupancy.append(gmsh.model.isInside(3,volume,[60.,35.,-60.]))
            if mode == 'transition':
                assert record['algorithm'] == 'g2-bezier-sweep'
                assert record['volumes'] == 1
        finally:
            gmsh.finalize()
    assert occupancy == [1,0]


def test_equal_area_tubes_with_different_diameters_do_not_use_one_profile_sweep():
    """防止两圆管仅面积相同就被认定为同截面扫掠，导致另一切面直径被替换。"""
    spec = solid_joint.prepare_joint_spec(_l_joint(section=sections.circular_tube('T',.16,.02)),2)
    p = sg.profile(sg.descriptor('tube',d=200.,t=100.-np.sqrt(7200.)))
    arm = replace(spec.arms[1],outer_diameter_mm=200.,inner_diameter_mm=2*np.sqrt(7200.),profile_mm=p)
    plan, reason = joint_blend.sweep_plan(replace(spec,arms=(spec.arms[0],arm)))
    assert plan is None and '直径' in reason


def test_rotated_h_profiles_select_miter_instead_of_silently_twisting_the_cut():
    """防止扫掠端部截面朝向不匹配，损坏原杆件局部轴与边界运动关系。"""
    spec = solid_joint.prepare_joint_spec(_l_joint(section=sections.i_section('H',.18,.12,.02,.025)),2)
    a = spec.arms[1]
    y,z = np.asarray(a.axis_y),np.asarray(a.axis_z)
    a = replace(a,axis_y=tuple((y+z)/np.sqrt(2)),axis_z=tuple((z-y)/np.sqrt(2)))
    plan, reason = joint_blend.sweep_plan(replace(spec,arms=(spec.arms[0],a)))
    assert plan is None and '朝向' in reason


def test_three_arm_joint_builds_one_miter_body_with_actual_algorithm_record():
    """防止新算法只处理示例两臂，多臂节点仍被自动填成凸包大块。"""
    gmsh = pytest.importorskip('gmsh')
    spec = solid_joint.prepare_joint_spec(_l_joint(section=sections.rectangle('R',.12,.18)),2)
    third = replace(spec.arms[1],member_id=3,direction=(-1.,0.,0.))
    spec = replace(spec,arms=(*spec.arms,third),geometry_mode='miter')
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber('General.Terminal',0)
        gmsh.model.add('three_arm')
        record = joint_geometry.build_occ_joint(gmsh,spec)
        assert record['algorithm'] == 'halfspace-miter+rolling-ball'
        assert record['volumes'] == 1
        assert record['sweep_unavailable_reason'] is None
    finally:
        gmsh.finalize()


def test_failed_cad_fillet_is_recorded_without_replacing_h_voids_by_a_hull(monkeypatch):
    """防止圆角失败后悄悄退回被用户否定的块状填充，并误称已经形成光顺曲面。"""
    gmsh = pytest.importorskip('gmsh')
    spec = solid_joint.prepare_joint_spec(_l_joint(section=sections.i_section('H',.18,.12,.02,.025)),2,geometry_mode='miter')
    def unavailable(*args,**kw):
        raise RuntimeError('CAD fillet unavailable')
    monkeypatch.setattr(gmsh.model.occ,'fillet',unavailable)
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber('General.Terminal',0)
        gmsh.model.add('unavailable')
        record = joint_geometry.build_occ_joint(gmsh,spec)
        assert record['fillet_status'] == 'kernel_unavailable'
        assert record['applied_fillet_radius_mm'] == 0
        assert gmsh.model.isInside(3,gmsh.model.getEntities(3)[0][1],[60.,35.,-60.]) == 0
    finally:
        gmsh.finalize()


def test_g2_step_round_trip_preserves_body_volume_and_mm_bounds(tmp_path):
    """防止曲面只可显示，导出的STEP却失去体积、孔洞或毫米尺度，无法进入Abaqus。"""
    gmsh = pytest.importorskip('gmsh')
    spec = solid_joint.prepare_joint_spec(_l_joint(section=sections.circular_tube('T',.16,.02)),2,geometry_mode='transition')
    path = tmp_path/'g2_joint.step'
    record = joint_geometry.export_step(spec,path)
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber('General.Terminal',0)
        gmsh.option.setString('Geometry.OCCTargetUnit','MM')
        gmsh.model.add('imported')
        volumes = gmsh.model.occ.importShapes(str(path))
        gmsh.model.occ.synchronize()
        assert len(volumes) == 1
        assert set(gmsh.model.getEntities(2)) == set(gmsh.model.getBoundary(volumes,combined=False,oriented=False))
        assert gmsh.model.occ.getMass(*volumes[0]) == pytest.approx(record['volume_mm3'],rel=1e-6)
        bounds = np.asarray(gmsh.model.getBoundingBox(*volumes[0]))
        assert np.max(abs(bounds)) > 300 and np.max(abs(bounds)) < 1000
    finally:
        gmsh.finalize()


def test_cad_error_becomes_actionable_tool_error_and_releases_gmsh(monkeypatch):
    """防止曲面CAD失败直接抛到工具外，且必须释放Gmsh以允许用户切换几何重试。"""
    gmsh = pytest.importorskip('gmsh')
    spec = solid_joint.prepare_joint_spec(_l_joint(section=sections.rectangle('R',.12,.18)),2,geometry_mode='transition')
    def invalid(*args,**kw):
        raise ValueError('invalid curve')
    monkeypatch.setattr(joint_geometry,'build_occ_joint',invalid)
    with pytest.raises(solid_joint.SolidJointError,match='切换节点几何'):
        native_joint.generate_joint_mesh(spec,30)
    assert gmsh.isInitialized() == 0


@pytest.mark.parametrize('radius, status, source', [
    (None, 'rounded', 'auto'), (0., 'not_requested', 'user'), (4., 'rounded', 'user')])
def test_fillet_radius_is_a_traceable_input(radius, status, source):
    """圆角半径直接改变热点应力与 Kt，原来只按截面特征尺寸自动取值，用户既看不到也改不了。

    留空仍自动取值但记下来源；0 按平接实体建立（明确声明不加圆角，不算失败后退回）；
    正数按用户给的半径建立。
    """
    gmsh = pytest.importorskip('gmsh')
    spec = replace(_teaching_column_joint(), fillet_radius_mm=radius)
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.model.add('fillet_input')
        record = joint_geometry.build_occ_joint(gmsh, spec)
        assert record['fillet_status'] == status
        assert record['fillet_radius_source'] == source
        if radius == 0.:
            assert record['applied_fillet_radius_mm'] == 0.
            assert len(gmsh.model.getEntities(3)) == 1
        elif radius:
            assert record['applied_fillet_radius_mm'] in (pytest.approx(4.), pytest.approx(2.))
    finally:
        gmsh.finalize()
    warning = native_joint._auto_fillet_warning(record)
    assert (warning is not None) == (source == 'auto' and status == 'rounded')
    if warning:
        assert 'fillet_radius_mm' in warning and 'Kt' in warning


def test_negative_fillet_radius_is_rejected_before_meshing():
    from agent import Session
    path = Path(__file__).resolve().parents[1]/'examples/空间刚架_简单整体复杂实体.json'
    session = Session(model=json.loads(path.read_text(encoding='utf-8')))
    assert session.solve_model().ok
    with pytest.raises(solid_joint.SolidJointError, match='非负有限数'):
        solid_joint.prepare_joint_spec(session, 5, case='W_侧向', geometry_mode='transition',
                                       fillet_radius_mm=-1.)


# ------------------------------------------------- 型钢根部圆角

def test_root_fillet_area_matches_the_closed_form_and_the_section_table():
    """防止型钢表截面（面积含根部圆角）被直角轮廓判为“面积与真实轮廓不一致”。

    真实问题：HN400×200×8×13 按表填 A=84.12 cm²，直角轮廓只有 81.92 cm²，节点实体
    每次都拒绝。四个内凹角各补 (1−π/4)r²，r=16 mm 时与表值相差不到 1e-4。
    """
    b, h, tw, tf, r = 0.2, 0.4, 0.008, 0.013, 0.016
    p = sg.profile({"type": "i", "dimensions": {"b": b, "h": h, "tw": tw, "tf": tf, "r": r}})
    exact = 2*b*tf + (h - 2*tf)*tw + 4*(1 - np.pi/4)*r**2
    assert p["area"] == pytest.approx(exact, rel=1e-12)
    assert abs(p["area"] - 8.412e-3) < 1e-4 * 8.412e-3
    assert p["concave_corners"] == 4
    # 双轴对称：四段圆弧折线必须互为镜像，形心落在几何中心
    assert p["centroid"] == pytest.approx([h/2, b/2], abs=1e-12)
    outer = np.asarray(p["outer"])
    assert np.allclose(np.sort(outer[:, 0]), np.sort(-outer[:, 0]), atol=1e-12)
    # 圆弧上的点到圆心距离恰为 r：取左下内凹角 (tf, (b−tw)/2) 的圆心
    center = np.array([tf + r, (b - tw)/2 - r]) - np.array([h/2, b/2])
    near = outer[np.linalg.norm(outer - center, axis=1) < r * 1.01]
    assert len(near) >= 3
    assert np.linalg.norm(near - center, axis=1) == pytest.approx(r, rel=1e-12)


@pytest.mark.parametrize("kind, dims, corners, sharp", [
    ("channel", {"h": .25, "b": .09, "tw": .008, "tf": .012}, 2, 2*.09*.012 + (.25 - .024)*.008),
    ("angle", {"b": .075, "t": .008}, 1, 2*.075*.008 - .008**2),
    ("tee", {"h": .2, "b": .15, "tw": .008, "tf": .012}, 2, .15*.012 + (.2 - .012)*.008),
])
def test_root_fillet_rounds_every_reentrant_corner_of_rolled_profiles(kind, dims, corners, sharp):
    """防止根部圆角只对工字形生效：槽钢、角钢、T 型钢按内凹角个数补同样的解析面积。"""
    r = .01
    p = sg.profile({"type": kind, "dimensions": {**dims, "r": r}})
    assert p["concave_corners"] == corners
    assert p["area"] == pytest.approx(sharp + corners*(1 - np.pi/4)*r**2, rel=1e-12)


def test_root_fillet_area_is_unit_invariant():
    """防止圆角面积按模型单位出现换算误差：米制与毫米制只差长度比例的平方。"""
    g = {"type": "i", "dimensions": {"b": .2, "h": .4, "tw": .008, "tf": .013, "r": .016}}
    in_m, in_mm = sg.profile(g), sg.profile(sg.scale_geometry(g, 1000.0))
    assert in_mm["area"] == pytest.approx(in_m["area"] * 1e6, rel=1e-12)
    assert np.asarray(in_mm["outer"]) == pytest.approx(np.asarray(in_m["outer"]) * 1000, abs=1e-9)


@pytest.mark.parametrize("geometry, message", [
    ({"type": "i", "dimensions": {"b": .2, "h": .4, "tw": .008, "tf": .013, "r": .2}}, "过大"),
    ({"type": "box", "dimensions": {"b": .2, "h": .3, "t": .01, "r": .01}}, "只适用于"),
])
def test_root_fillet_rejects_radius_that_does_not_fit(geometry, message):
    """防止 r 填成别的单位（如毫米制模型填 16 当成 0.016 的反面）时生成自交轮廓。"""
    with pytest.raises(ValueError, match=message):
        sg.profile(geometry)


def test_exact_section_properties_converge_from_the_discretised_fillet():
    """防止圆角截面特性只按折线近似：解析值应是折线多边形加密的极限，误差按 n⁻² 下降。"""
    g = {"type": "i", "dimensions": {"b": .2, "h": .4, "tw": .008, "tf": .013, "r": .016}}
    exact = sg.section_properties(g)
    errors = []
    original = sg.FILLET_SEGMENTS_PER_QUARTER
    try:
        for n in (32, 128):
            sg.FILLET_SEGMENTS_PER_QUARTER = n
            m = sg._polygon_moments(sg.profile(g)["outer"])
            iz = m[3] - m[1]**2 / m[0]
            errors.append(abs(iz - exact["Iz"]) / exact["Iz"])
    finally:
        sg.FILLET_SEGMENTS_PER_QUARTER = original
    assert errors[1] < errors[0] / 12          # 4 倍加密，误差约降 16 倍
    assert errors[1] < 1e-5


@pytest.mark.parametrize("geometry, closed", [
    ({"type": "rectangle", "dimensions": {"b": .12, "h": .3}},
     (.036, .3*.12**3/12, .12*.3**3/12)),
    ({"type": "box", "dimensions": {"b": .2, "h": .3, "t": .01}},
     (.2*.3 - .18*.28, (.3*.2**3 - .28*.18**3)/12, (.2*.3**3 - .18*.28**3)/12)),
])
def test_exact_section_properties_match_closed_forms_without_fillet(geometry, closed):
    """直角轮廓与带孔轮廓的格林公式结果必须与经典闭合解一致（Iz 为高度方向强轴）。"""
    p = sg.section_properties(geometry)
    assert (p["A"], p["Iy"], p["Iz"]) == pytest.approx(closed, rel=1e-12)
