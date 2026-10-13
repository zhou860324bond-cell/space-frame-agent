"""真实截面、连接补齐、网格和连续应力显示的回归。"""

from dataclasses import replace

import numpy as np
import pytest

import joint_geometry
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


def test_transition_fills_only_joint_and_adds_material_to_rectangular_corner():
    """防止补齐算法变成填满整根杆件，或只改显示而没有生成连接体积。"""
    gmsh = pytest.importorskip("gmsh")
    spec = solid_joint.prepare_joint_spec(_l_joint(section=sections.rectangle("R", .12, .18)), 2)
    volumes = []
    for mode in ("intersection", "transition"):
        gmsh.initialize(interruptible=False)
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add(mode)
            record = joint_geometry.build_occ_joint(gmsh, replace(spec, geometry_mode=mode))
            volumes.append(record["volume_mm3"])
            assert record["volumes"] == 1
            assert record["connection_depth_mm"] < min(a.length_mm for a in spec.arms)
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
