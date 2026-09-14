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

def test_auto_view_uses_perspective_iso_even_for_planar_models():
    assert scene.preferred_view(portal().frame) == "isometric"
    spatial = scene.Frame(nodes={
        1: scene.Node(1, 0.0, 0.0, 0.0),
        2: scene.Node(2, 2.0, 0.0, 0.0),
        3: scene.Node(3, 0.0, 3.0, 0.0),
        4: scene.Node(4, 0.0, 0.0, 4.0),
    })
    assert scene.preferred_view(spatial) == "isometric"


def test_display_sampling_is_dense_without_refining_the_analysis_mesh():
    """显示加密和分析剖分是两件事；这一档只改善曲线/色块，不改刚度方程。"""
    s = beam()
    assert len(s.frame.members) == 1
    line = scene.member_polylines(s.frame, s.solution, "D", scalars="Mz")
    assert line.n_points == scene.STATIONS
    assert scene.STATIONS >= 41


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
    lo, hi = scene.contour_clim(line, "M", percentile=None)
    assert lo == 0.0 and hi == pytest.approx(values.max())


def test_contour_clim_clips_to_a_percentile_by_default():
    """默认按分位裁剪上限，并且能报告自己裁过——不裁的话重尾分布下
    绝大多数构件会挤在色带最底端，云图等于没有信息。"""
    s = portal()
    line = scene.member_polylines(s.frame, s.solution, "D", scalars="M")
    values = np.asarray(line["M"])
    lo, hi = scene.contour_clim(line, "M")
    assert lo == 0.0
    assert hi == pytest.approx(np.percentile(values, scene.CONTOUR_PERCENTILE))
    assert hi < values.max()
    assert scene.clim_is_clipped(line, "M", (lo, hi)) is True


def test_full_range_is_not_reported_as_clipped():
    """满量程时不得挂"已裁剪"的标——那会在图上写下一句不成立的说明。"""
    s = portal()
    line = scene.member_polylines(s.frame, s.solution, "D", scalars="M")
    clim = scene.contour_clim(line, "M", percentile=None)
    assert scene.clim_is_clipped(line, "M", clim) is False


def test_signed_components_still_use_a_symmetric_colour_range():
    s = portal()
    line = scene.member_polylines(s.frame, s.solution, "D", scalars="Mz")
    assert scene.contour_clim(line, "Mz") == scene.symmetric_clim(line, "Mz")


def test_contour_caption_does_not_claim_cross_section_stress():
    signed = scene.contour_caption("Mz", "kN·m", clipped=True,
                                   sign_filter="negative")
    magnitude = scene.contour_caption("M", "kN·m")
    assert "BEAM CENTERLINE INTERNAL-FORCE RESULT" in signed
    assert "signed in member local axes" in signed
    assert "clipped" in signed
    assert "negative values only" in signed
    assert "magnitude (nonnegative)" in magnitude


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
    # 容差按符号尺寸取，不用绝对 1e-9：锥面是多边形离散的，顶点位置带
    # 与半径同量级的舍入，符号一改小这条就会以 6e-9 这种量级失败，
    # 而它想守的其实是"画在节点下方"，不是某个绝对精度。
    tolerance = 1e-6 * scene.model_size(s.frame)
    assert bounds[2, 1] <= tolerance, "符号画在节点下方"


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


def test_support_labels_state_the_exact_restrained_dofs():
    s = beam()
    s.frame.supports[1] = (1, 0, 1, 0, 1, 0)
    points, labels = scene.support_labels(s.frame)
    assert len(points) == len(labels) == 2
    assert "N1 BC: U1 U3 UR2" in labels
    assert "N2 BC: U2 U3 UR1" in labels


def test_selected_member_local_axes_match_the_solver_axes():
    from frame3d import local_axes

    s = portal()
    mid = sorted(s.frame.members)[0]
    member = s.frame.members[mid]
    pi, pj = scene.member_endpoints(s.frame, member)
    _, expected = local_axes(pi, pj, member.ref_vector)
    glyphs = scene.member_local_axis_glyphs(s.frame, mid)
    points, labels = scene.member_local_axis_labels(s.frame, mid)
    origin = 0.5 * (pi + pj)
    shown = np.asarray(points) - origin
    shown /= np.linalg.norm(shown, axis=1)[:, None]
    assert set(glyphs) == {"x", "y", "z"}
    assert all(mesh.n_cells > 0 for mesh in glyphs.values())
    assert shown == pytest.approx(expected)
    assert labels == ["local x (i->j)", "local y", "local z"]


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


def test_moment_is_a_circular_right_hand_glyph_not_a_straight_force_arrow():
    s = beam()
    assert s.set_load_cases(cases=[{
        "name": "D",
        "nodal_loads": [{"node": 2, "load": [0, 0, 0, 1000, 0, 0]}],
    }]).ok
    mesh = scene.load_arrows(s.preview_frame(), "D")["节点力矩"]
    extent = np.ptp(np.asarray(mesh.points), axis=0)
    # 绕全局 X 的力矩应在 YZ 平面形成圆弧，而不是沿 X 拉出一根直箭头。
    assert extent[1] > 3.0 * extent[0]
    assert extent[2] > 3.0 * extent[0]


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


def test_load_labels_keep_signed_global_components_instead_of_magnitude():
    s = beam()
    assert s.set_load_cases(cases=[{
        "name": "D",
        "nodal_loads": [{"node": 2,
                          "load": [6000, 0, -8000, 0, 2000, 0]}],
    }]).ok
    _, labels = scene.load_labels(s.preview_frame(), "D")
    assert "Fx=+6, Fz=-8 kN [global]" in labels
    assert "My=+2 kN·m [global]" in labels
    assert all("10 kN" not in label for label in labels)


def test_trapezoid_label_reports_both_signed_end_intensities():
    s = beam()
    assert s.set_load_cases(cases=[{
        "name": "D",
        "member_spans": [{"member": 1, "kind": "trapezoid",
                           "w1": [0, 0, -3000], "w2": [0, 0, -9000]}],
    }]).ok
    _, labels = scene.load_labels(s.preview_frame(), "D")
    assert labels == ["w1z=-3 -> w2z=-9 kN/m [global]"]


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


# ------------------------------------------------- 云图分级（色块分区）

def test_band_index_and_value_agree_with_the_colour_bar():
    clim = (-60.0, 60.0)
    idx = scene.band_index([-60.0, -0.001, 0.0, 59.9, 60.0, 1e9], clim, 12)
    assert list(idx) == [0, 5, 6, 11, 11, 11], "超量程要夹到两端，不能溢出"
    # 每一级的代表值必须落在这一级里，否则色块和色标差一格
    for k in range(12):
        value = float(scene.band_value(k, clim, 12))
        assert scene.band_index(value, clim, 12) == k


def test_every_banded_segment_carries_exactly_one_colour():
    """分级着色的关键：颜色挂在**单元**上，一段一个常量值。

    挂在点上渲染器会插值，插出来是渐变，色块边界就没了。
    """
    line = pv.PolyData()
    line.points = np.array([[0.0, 0, 0], [10.0, 0, 0]])
    line.lines = np.array([2, 0, 1])
    line["v"] = np.array([0.0, 60.0])
    pieces = scene.banded_segments(line, "v", (0.0, 60.0), 12)

    name = "v" + scene.BAND_SUFFIX
    assert name in pieces.cell_data
    assert name not in pieces.point_data, "颜色不许挂在点上"
    assert pieces.n_cells == 12, "0→60 跨满 12 级，就该切成 12 段"
    values = np.asarray(pieces.cell_data[name])
    assert len(set(np.round(values, 9))) == 12
    assert values == pytest.approx(scene.band_value(np.arange(12), (0.0, 60.0), 12))


def test_a_band_boundary_is_cut_exactly_where_the_value_crosses_it():
    """边界位置是**算出来的**：线性插值到该级的边界值，不是按段长凑的。"""
    line = pv.PolyData()
    line.points = np.array([[0.0, 0, 0], [4.0, 0, 0]])
    line.lines = np.array([2, 0, 1])
    line["v"] = np.array([0.0, 4.0])          # 值与 x 同步，便于手算
    pieces = scene.banded_segments(line, "v", (0.0, 4.0), 4)   # 边界在 1、2、3
    xs = np.sort(np.unique(np.round(pieces.points[:, 0], 9)))
    assert xs == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0])


def test_a_segment_inside_one_band_is_not_split():
    line = pv.PolyData()
    line.points = np.array([[0.0, 0, 0], [1.0, 0, 0]])
    line.lines = np.array([2, 0, 1])
    line["v"] = np.array([11.0, 12.0])
    assert scene.banded_segments(line, "v", (0.0, 120.0), 12).n_cells == 1


def test_contour_levels_are_clamped_to_something_usable():
    from desktop import theme

    low, high = theme.CONTOUR_LEVELS_RANGE
    assert scene.contour_levels(None) == theme.CONTOUR_LEVELS
    assert scene.contour_levels(1) == low
    assert scene.contour_levels(999) == high


def test_the_banded_tube_keeps_the_cell_colours():
    session = portal()
    line = scene.contour_line(session.frame, session.solution, None, "Mz",
                              value_scale=1e-3)
    clim = scene.contour_clim(line, "Mz", percentile=None)
    tube = scene.banded_tubes(line, "Mz", clim, 12, radius=0.02)
    name = "Mz" + scene.BAND_SUFFIX
    assert name in tube.cell_data and tube.n_cells > 0
    values = np.unique(np.round(tube.cell_data[name], 9))
    assert len(values) <= 12, "颜色只能取那 12 个代表值"


def test_viewport_text_avoids_glyphs_the_vtk_font_drops():
    """视口里的文字由 VTK 自己的字体引擎画。

    实测：σ 被整个吞掉（只剩下后面的单位），竖线渲染成一大片空白（一行字
    被撑成两截，看着像文字丢了）。所以画在视口里的说明一律避开这两个字符。
    """
    for component in ("Mz", "M", scene.STRESS):
        caption = scene.contour_caption(component, "kN.m", True, "positive", 12)
        assert "σ" not in caption
        assert "|" not in caption
        assert "12 bands" in caption
    assert "sigma" in scene.contour_caption(scene.STRESS, "MPa")


def test_the_viewport_diverging_colours_come_from_viz_theme():
    """屏幕和报告必须是同一套配色。

    这里原来直接用 matplotlib 的 "coolwarm"，而报告里的静态图用的是
    viz_theme 的锚点色——同一个结构两处颜色不一样，而 report.py 的说明里
    还写着"两者共用 viz_theme 配色，不会各说各话"。

    顺带钉住中点不能接近白色：深色视口上，零内力的杆件不该是全图最亮的。
    """
    import viz_theme as V
    from desktop import theme

    cmap = theme.diverging_cmap()
    assert theme.DIVERGING_ANCHORS == V.VIEWPORT_DIVERGING
    low = np.array(cmap(0.0)[:3])
    high = np.array(cmap(1.0)[:3])
    mid = np.array(cmap(0.5)[:3])
    assert low[2] > low[0], "低端应当偏蓝（受压）"
    assert high[0] > high[2], "高端应当偏红（受拉）"
    assert mid.mean() < 0.75, "中点不能接近白色"


def test_banding_a_colormap_gives_exactly_that_many_colours():
    from desktop import theme

    assert theme.banded(theme.diverging_cmap(), 8).N == 8
    assert theme.banded(theme.sequential_cmap(), 999).N == theme.CONTOUR_LEVELS_RANGE[1]


def test_the_out_of_range_part_is_a_mesh_of_its_own():
    """色标按分位裁剪时，超限的段要能单独拿出来上专门的颜色。

    **分级之后，饱和的那一级看起来和正常的一级一模一样**——峰值所在的位置
    就此消失在一片同色里。这是"裁剪"这件事在分级云图上的新副作用。
    """
    line = pv.PolyData()
    line.points = np.array([[0.0, 0, 0], [10.0, 0, 0]])
    line.lines = np.array([2, 0, 1])
    line["v"] = np.array([0.0, 100.0])
    over = scene.out_of_range_tubes(line, "v", (0.0, 50.0), radius=0.1)
    assert over.n_cells > 0
    # 只该包含 x>5 的那一半（值超过 50 的部分），边界精确落在 x=5
    assert over.bounds[0] == pytest.approx(5.0, abs=1e-9)
    assert over.bounds[1] == pytest.approx(10.0, abs=1e-9)

    inside = scene.out_of_range_tubes(line, "v", (0.0, 100.0), radius=0.1)
    assert inside.n_cells == 0, "没有超限就不该多出一层网格"


def test_the_band_cuts_and_the_range_cuts_come_from_one_place():
    """分级边界和"量程外"的边界必须是同一套切法。

    分开写两遍，迟早出现色块边界和橙色段错开半格——那种错看图的人查不出来。
    """
    line = pv.PolyData()
    line.points = np.array([[0.0, 0, 0], [4.0, 0, 0]])
    line.lines = np.array([2, 0, 1])
    line["v"] = np.array([0.0, 4.0])
    got = [(round(float(a[0]), 9), round(float(b[0]), 9), round(v, 9))
           for a, b, v in scene.cut_at(line, "v", (1.0, 3.0))]
    assert [g[0] for g in got] == [0.0, 1.0, 3.0]
    assert [g[1] for g in got] == [1.0, 3.0, 4.0]
    assert [g[2] for g in got] == [0.5, 2.0, 3.5]     # 段中点的值
