"""桌面端建网格层的验证。

这一层是纯函数、无 Qt，所以能像内核一样逐条验——**这正是当初把它和
Qt 分开的理由**。会出错的是"几何和标量对不对"，不是"控件摆在哪"。

重点验四件事：

1. 管子确实沿杆件走，端点落在节点上；
2. 云图标量是每根杆**逐点**的真实内力，不是一根杆一个色块；
3. 发散色标关于零对称——不对称的话全受压结构里会有杆被读成受拉；
4. 变形用的是**单元内解析挠度**，单跨一个单元也画得出跨中那个鼓。
"""

import numpy as np
import pytest

pv = pytest.importorskip("pyvista", reason="未安装 pyvista，跳过桌面端网格测试")
pv.OFF_SCREEN = True

from agent import Session                                   # noqa: E402
from desktop import scene                                   # noqa: E402

MATERIALS = [{"name": "Q355", "E": 2.06e11, "nu": 0.3, "density": 7850.0}]
SECTIONS = [{"name": "COLUMN", "A": 0.0147, "Iy": 4.2e-5, "Iz": 1.18e-3, "J": 9e-7},
            {"name": "RAFTER", "A": 0.0186, "Iy": 5.2e-5, "Iz": 2.47e-3, "J": 1.4e-6}]


def portal() -> Session:
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    g = s.generate_portal_frame(spans=[24.0], eave_height=7.5, ridge_rise=1.2,
                                column_section="COLUMN", rafter_section="RAFTER",
                                material="Q355", base="pinned")
    s.set_load_cases(cases=[{"name": "D", "member_loads":
                             [{"member": m, "w": [0, 0, -8e3]}
                              for m in g.payload["rafter_member_ids"]]}])
    s.solve_model()
    return s


def beam() -> Session:
    """单跨一个单元的简支梁——验"单元内挠度"必须用它。"""
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    s.set_model(model={
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                  {"id": 2, "x": 8.0, "y": 0, "z": 0}],
        "members": [{"id": 1, "i": 1, "j": 2,
                     "section": "RAFTER", "material": "Q355"}],
        "supports": [{"node": 1, "fix": [1, 1, 1, 1, 0, 0]},
                     {"node": 2, "fix": [0, 1, 1, 1, 0, 0]}]})
    s.set_load_cases(cases=[{"name": "D", "member_loads":
                             [{"member": 1, "w": [0, 0, -25e3]}]}])
    s.solve_model()
    return s


# ------------------------------------------------- 几何

def test_polylines_start_and_end_at_the_real_nodes():
    s = portal()
    f = s.frame
    line = scene.member_polylines(f)
    pts = np.asarray(line.points)
    for mid in f.members:
        m = f.members[mid]
        for target in (f.nodes[m.i].xyz, f.nodes[m.j].xyz):
            assert np.min(np.linalg.norm(pts - target, axis=1)) < 1e-9


def test_every_member_becomes_a_tube():
    s = portal()
    tubes = scene.member_tubes(s.frame)
    assert tubes.n_cells > 0
    # 管子把每根杆包起来，包围盒不该比模型小
    bounds = np.array(tubes.bounds).reshape(3, 2)
    coords = np.array([n.xyz for n in s.frame.nodes.values()])
    assert np.all(bounds[:, 0] <= coords.min(axis=0) + 1e-9)
    assert np.all(bounds[:, 1] >= coords.max(axis=0) - 1e-9)


def test_the_member_id_travels_with_the_mesh():
    """拾取要靠它——点中一根管子得知道是哪根杆。"""
    s = portal()
    line = scene.member_polylines(s.frame)
    assert "member" in line.array_names
    assert set(np.unique(line["member"])) == set(s.frame.members)


def test_member_end_result_points_are_not_merged_at_joints():
    """梁截面力是单元局部结果，公共节点处不能把两根杆的端值揉成一个。

    PyVista 的 merge 默认合并重合点。门架 4 根杆、每根 21 个结果点，旧实现
    只剩 81 个点；丢掉的 3 个恰好是连接节点，云图最关键的杆端值被覆盖了。
    """
    s = portal()
    line = scene.member_polylines(s.frame, s.solution, "D", scalars="Mz")
    assert line.n_points == len(s.frame.members) * scene.STATIONS


# ------------------------------------------------- 标量

def test_scalars_vary_along_a_member_not_just_between_members():
    """弯矩沿杆长是变的。每根杆只给一个值的话，云图就成了色块拼图，
    跨中和支座的差别全看不见。"""
    s = beam()
    line = scene.member_polylines(s.frame, s.solution, "D", scalars="Mz")
    values = np.asarray(line["Mz"])
    assert len(values) >= scene.STATIONS
    assert values.std() > 0, "同一根杆上弯矩必须有变化"
    # 简支梁：两端为零、跨中最大
    assert abs(values[0]) < 1e-6 * abs(values).max()
    assert abs(values[-1]) < 1e-6 * abs(values).max()
    assert abs(values).max() == pytest.approx(25e3 * 8.0 ** 2 / 8, rel=1e-3)


def test_scalars_match_the_solver_not_a_second_calculation():
    """网格上的标量必须来自 member_diagram，不能是另算一遍的。"""
    from internal_forces import member_diagram

    s = portal()
    line = scene.member_polylines(s.frame, s.solution, "D", scalars="N")
    mid = sorted(s.frame.members)[0]
    got = np.asarray(line["N"])[np.asarray(line["member"]) == mid]
    d = member_diagram(s.frame, s.solution, mid, "D", stations=scene.STATIONS)
    assert got == pytest.approx(d.N, rel=1e-9)


def test_point_load_jump_is_kept_in_the_contour_stations():
    """集中力处的剪力是跳跃，不应被等距重采样画成一段斜坡。"""
    from span_loads import POINT, SpanLoad

    s = beam()
    s.frame.case("D").member_loads.clear()
    s.frame.case("D").member_spans[1] = [
        SpanLoad(POINT, (0.0, 0.0, -25e3), a=3.7)
    ]
    s.solution = __import__("frame3d").solve(s.frame)
    line = scene.member_polylines(s.frame, s.solution, "D", scalars="Vy")
    # 基础等距点之外，荷载点左/中/右三个恢复站也必须进入显示网格。
    assert line.n_points > scene.STATIONS
    x = np.asarray(line.points)[:, 0]
    near = np.argsort(np.abs(x - 3.7))[:3]
    assert np.ptp(np.asarray(line["Vy"])[near]) > 0.9 * 25e3


# ------------------------------------------------- 色标

def test_the_diverging_range_is_symmetric_about_zero():
    """全受压的结构里，最大值（最轻的压力，≈0）会落在红色端——
    而红色在我们的约定里是受拉。看图的人会把轻微受压的杆读成受拉。
    这不是难看，是读错。"""
    s = portal()
    tubes = scene.member_tubes(s.frame, s.solution, "D", scalars="N")
    assert np.asarray(tubes["N"]).max() <= 1e-6, "这个算例应当全受压"
    lo, hi = scene.symmetric_clim(tubes, "N")
    assert lo == -hi and hi > 0


def test_an_all_zero_field_gets_a_usable_range():
    """全零时不能给 VTK 一个零跨度的 clim。"""
    s = beam()
    tubes = scene.member_tubes(s.frame, s.solution, "D", scalars="T")
    assert np.allclose(tubes["T"], 0.0)
    lo, hi = scene.symmetric_clim(tubes, "T")
    assert lo < hi


def test_spatial_bending_magnitude_is_axis_invariant_and_nonnegative():
    """空间云图默认比较合弯矩，不能让杆件局部 y/z 方向改变结论。"""
    s = portal()
    line = scene.member_polylines(s.frame, s.solution, "D", scalars="M")
    values = np.asarray(line["M"])
    assert np.all(values >= 0.0)
    lo, hi = scene.contour_clim(line, "M")
    assert lo == 0.0 and hi == pytest.approx(values.max())


def test_signed_components_still_use_a_symmetric_colour_range():
    s = portal()
    line = scene.member_polylines(s.frame, s.solution, "D", scalars="Mz")
    assert scene.contour_clim(line, "Mz") == scene.symmetric_clim(line, "Mz")


# ------------------------------------------------- 变形

def test_the_deformed_shape_uses_in_element_deflection():
    """**单跨一个单元**的简支梁：两端节点位移都是零。

    只按节点位移画的话，变形后还是一条直线，跨中那个鼓完全看不见。
    这条测试盯的就是这个——和内力图那边是同一个道理。
    """
    s = beam()
    plain = scene.member_polylines(s.frame)
    moved = scene.member_polylines(s.frame, s.solution, "D", scale=1.0)
    offset = np.asarray(moved.points) - np.asarray(plain.points)
    # 端点不动
    assert np.linalg.norm(offset[0]) < 1e-12
    assert np.linalg.norm(offset[-1]) < 1e-12
    # 跨中动了，而且等于闭合解 5wL⁴/384EI
    mid = len(offset) // 2
    exact = 5 * 25e3 * 8.0 ** 4 / (384 * 2.06e11 * 2.47e-3)
    assert np.linalg.norm(offset[mid]) == pytest.approx(exact, rel=1e-3)


def test_auto_scale_uses_the_in_element_peak_not_only_nodes():
    """单单元简支梁的节点位移为零，自动放大仍必须得到可见的变形。"""
    s = beam()
    scale = scene.auto_deformation_scale(s.frame, s.solution, "D")
    exact = 5 * 25e3 * 8.0 ** 4 / (384 * 2.06e11 * 2.47e-3)
    assert scale == pytest.approx(0.06 * scene.model_size(s.frame) / exact,
                                  rel=1e-3)
    assert scale > 1.0


def test_the_scale_factor_scales():
    s = beam()
    a = scene.member_polylines(s.frame, s.solution, "D", scale=1.0)
    b = scene.member_polylines(s.frame, s.solution, "D", scale=3.0)
    base = np.asarray(scene.member_polylines(s.frame).points)
    assert (np.asarray(b.points) - base) == pytest.approx(
        3 * (np.asarray(a.points) - base), rel=1e-9)


def test_deformed_member_ends_equal_nodal_displacements_once():
    """杆端中心线必须准确落在变形后的节点上，不能把横移叠加两次。"""
    s = portal()
    moved = scene.member_polylines(s.frame, s.solution, "D", scale=1.0)
    mids = np.asarray(moved["member"])
    points = np.asarray(moved.points)
    result = s.solution["D"]
    for mid, member in s.frame.members.items():
        own = points[mids == mid]
        ui = result.U[s.frame.node_dofs(member.i)[:3]]
        uj = result.U[s.frame.node_dofs(member.j)[:3]]
        assert own[0] == pytest.approx(s.frame.nodes[member.i].xyz + ui,
                                       abs=1e-10)
        assert own[-1] == pytest.approx(s.frame.nodes[member.j].xyz + uj,
                                        abs=1e-10)


# ------------------------------------------------- 符号

def test_support_glyphs_are_classified_and_placed():
    s = portal()
    glyphs = scene.support_glyphs(s.frame)
    assert set(glyphs) == {"铰接"}, "门式刚架柱脚是铰接"
    bounds = np.array(glyphs["铰接"].bounds).reshape(3, 2)
    assert bounds[2, 1] <= 1e-9, "符号画在节点下方"


def test_out_of_plane_restraints_do_not_get_a_glyph():
    """平面刚架的面外约束每个节点都有。全画出来会把真正的柱脚淹掉。"""
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    s.generate_frame(spans=[6.0, 6.0], storeys=[3.6], column_section="COLUMN",
                     beam_section="RAFTER", material="Q355")
    frame = s.preview_frame()
    drawn = sum(1 for mask in frame.supports.values() if sum(mask[:3]) >= 2)
    assert drawn == 3 < len(frame.supports)
    glyphs = scene.support_glyphs(frame)
    assert sum(m.n_cells for m in glyphs.values()) > 0


def test_load_arrows_point_the_way_the_load_does():
    s = portal()
    arrows = scene.load_arrows(s.frame, "D")
    assert "杆间荷载" in arrows
    bounds = np.array(arrows["杆间荷载"].bounds).reshape(3, 2)
    coords = np.array([n.xyz for n in s.frame.nodes.values()])
    # 向下的荷载：箭头尾端在杆件上方
    assert bounds[2, 1] > coords[:, 2].max() - 1e-9


def test_the_two_load_groups_are_scaled_independently():
    """量级差几个数量级时，共用一把尺子会让小的那类彻底消失——
    而"看不见"和"不存在"在图上分不出来。"""
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    g = s.generate_frame(spans=[6.0], storeys=[3.6], column_section="COLUMN",
                         beam_section="RAFTER", material="Q355")
    s.set_load_cases(cases=[{
        "name": "D",
        "member_loads": [{"member": m, "w": [0, 0, -20e3]}
                         for m in g.payload["beam_member_ids"]],
        "nodal_loads": [{"node": 3, "load": [0, 0, -1.0, 0, 0, 0]}]}])
    arrows = scene.load_arrows(s.preview_frame(), "D")
    assert set(arrows) == {"节点荷载", "杆间荷载"}
    assert arrows["节点荷载"].n_cells > 0


def test_pure_moments_and_prescribed_motion_are_not_invisible():
    s = beam()
    result = s.set_load_cases(cases=[{
        "name": "D",
        "nodal_loads": [{"node": 2, "load": [0, 0, 0, 1000, 0, 0]}],
        "settlements": [{"node": 1, "d": [0.001, 0, 0, 0.01, 0, 0]}],
    }])
    assert result.ok, result.payload
    arrows = scene.load_arrows(s.preview_frame(), "D")
    assert {"节点力矩", "给定位移", "给定转角"}.issubset(arrows)
    assert all(arrows[key].n_cells > 0
               for key in ("节点力矩", "给定位移", "给定转角"))


def test_a_trapezoid_crossing_zero_does_not_create_a_zero_direction_arrow():
    s = beam()
    assert s.set_load_cases(cases=[{
        "name": "D",
        "member_spans": [{"member": 1, "kind": "trapezoid",
                           "w1": [0, 0, 1000], "w2": [0, 0, -1000]}],
    }]).ok
    arrows = scene.load_arrows(s.preview_frame(), "D", per_member=3)
    assert "杆间荷载" in arrows
    assert np.isfinite(np.asarray(arrows["杆间荷载"].points)).all()


def test_mode_shape_tubes_move_the_structure():
    from modal import modal

    s = portal()
    r = modal(s.frame, 3)
    still = scene.member_tubes(s.frame)
    moved = scene.mode_shape_tubes(s.frame, r.shapes, 0,
                                   scale=0.1 * scene.model_size(s.frame))
    assert not np.allclose(np.array(still.bounds), np.array(moved.bounds))


def test_mode_scale_uses_recovered_translation_not_rotation_units():
    from modal import member_mode_displacement

    s = portal()
    frame = s.frame
    shapes = np.zeros((frame.num_dofs, 1))
    member = next(iter(frame.members.values()))
    shapes[frame.node_dofs(member.j)[4], 0] = -1000.0
    scale = scene.auto_mode_scale(frame, shapes, 0)
    shown = max(
        np.linalg.norm(scale * member_mode_displacement(
            frame, shapes, 0, mid, scene.DEFLECTION_STATIONS)[1], axis=1).max()
        for mid in frame.members)
    assert shown == pytest.approx(0.12 * scene.model_size(frame))


def test_analysis_mesh_geometry_shows_the_compiler_split():
    """分析网格画的是编译产物，不是把物理杆件原样再画一遍。"""
    s = Session()
    assert s.set_model(model={
        "schema_version": 1,
        "units": "N-m-Pa",
        "nodes": [
            {"id": 11, "x": 0.0, "y": 0.0, "z": 0.0},
            {"id": 22, "x": 3.0, "y": 0.0, "z": 0.0},
        ],
        "members": [{"id": 101, "i": 11, "j": 22,
                     "section": "S", "material": "Steel"}],
        "materials": [{"name": "Steel", "E": 2.0e11, "nu": 0.3}],
        "sections": [{"name": "S", "A": 0.01, "Iy": 1e-5,
                      "Iz": 1e-5, "J": 2e-5}],
        "supports": [{"node": 11, "fix": [1, 1, 1, 1, 1, 1]}],
        "load_cases": [{
            "name": "L",
            "member_spans": [{"member": 101, "kind": "point",
                              "w1": [0.0, 0.0, -1000.0], "a": 1.5}],
        }],
    }).ok
    preview = s.preview_analysis_mesh()
    frame = s.preview_frame()

    mesh = scene.analysis_mesh_polylines(frame, preview.payload)
    split = scene.analysis_split_points(preview.payload)

    assert mesh.n_cells == 2
    assert set(mesh.cell_data["analysis_element"]) == {101, 102}
    assert set(mesh.cell_data["physical_member"]) == {101}
    assert split.n_points == 1
    assert split.points[0] == pytest.approx([1.5, 0.0, 0.0])
