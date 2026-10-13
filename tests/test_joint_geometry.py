"""真实截面、连接补齐、网格和连续应力显示的回归。"""

from dataclasses import replace

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
    spec = replace(spec,arms=(*spec.arms,third),geometry_mode='transition')
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber('General.Terminal',0)
        gmsh.model.add('three_arm')
        record = joint_geometry.build_occ_joint(gmsh,spec)
        assert record['algorithm'] == 'halfspace-miter+rolling-ball'
        assert record['volumes'] == 1
        assert '多臂' in record['sweep_unavailable_reason']
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
