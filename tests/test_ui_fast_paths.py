"""界面提速路径的等价性。

中等规模框架（500 节点、1300 杆、满布均布荷载）上，切一次显示模式原来要
几秒到几十秒。提速的每一处都**只许改变快慢，不许改变画出来的东西**：

1. 折线一次拼成——点、连接关系、逐点标量与逐杆拼接 merge 的结果一致；
2. 荷载箭头由模板实例化——每个箭头的几何与 `pv.Arrow` 一致；
3. 云图分级切段向量化——切出的每一小段与原逐段写法一致，顺序也一致；
4. Session 的预览帧与校验按内容缓存——模型一改就失效；
5. 杆件挠度/内力按 (frame, solution) 缓存——换一次求解就失效，返回副本。
"""

import copy

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from agent import Session                                           # noqa: E402
from desktop import scene                                           # noqa: E402
from frame3d import Frame, Material, Member, Node, Section, check_model  # noqa: E402
import internal_forces                                              # noqa: E402

MATERIALS = [{"name": "Q355", "E": 2.06e11, "nu": 0.3, "density": 7850.0}]
SECTIONS = [{"name": "COL", "A": 0.0147, "Iy": 4.2e-5, "Iz": 1.18e-3, "J": 9e-7,
             "cy": 0.15, "cz": 0.1},
            {"name": "BEAM", "A": 0.0186, "Iy": 5.2e-5, "Iz": 2.47e-3, "J": 1.4e-6,
             "cy": 0.2, "cz": 0.1}]


def solved_session(nx=2, nz=2, ny=1) -> Session:
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    assert s.generate_frame(spans=[6.0] * nx, storeys=[3.5] * nz, bays=[5.0] * ny,
                            beam_load=2e4, column_section="COL",
                            beam_section="BEAM", material="Q355").ok
    assert s.solve_model().ok
    return s


# --------------------------------------------------------------------------- 折线

def _merged_reference(chunks):
    """原来的做法：每条一个 PolyData，再 merge(merge_points=False)。"""
    blocks = [scene._polyline(c) for c in chunks]
    return blocks[0].merge(blocks[1:], merge_points=False) if len(blocks) > 1 \
        else blocks[0]


def test_polylines_match_the_per_block_merge():
    rng = np.random.default_rng(1)
    chunks = [rng.standard_normal((n, 3)) for n in (2, 5, 21, 3)]
    fast, slow = scene._polylines(chunks), _merged_reference(chunks)
    assert np.array_equal(fast.points, slow.points)
    assert np.array_equal(np.asarray(fast.lines), np.asarray(slow.lines))


def test_member_polylines_keep_per_member_scalars_and_owner():
    s = solved_session()
    frame = s.frame
    line = scene.member_polylines(frame, s.solution, None, 0.0, scalars="Mz")
    assert line.n_points == len(frame.members) * scene.STATIONS or line.n_points > 0
    owners = np.asarray(line["member"])
    # 每根杆的点连续成段，杆端**不**与相邻杆合并——节点处允许跳变
    assert list(dict.fromkeys(owners.tolist())) == sorted(frame.members)
    assert len(line["Mz"]) == line.n_points
    assert line.n_lines == len(frame.members)


def test_an_empty_frame_still_gives_an_empty_mesh():
    assert scene._polylines([]).n_points == 0
    assert scene.member_polylines(Frame()).n_points == 0


# --------------------------------------------------------------------------- 箭头

@pytest.mark.parametrize("direction", [(0, 0, -1), (1, 0, 0), (0.3, -0.4, 0.866),
                                       (0, 0, 1)])
def test_an_instanced_arrow_is_the_same_shape_as_pv_arrow(direction):
    d = np.asarray(direction, dtype=float)
    d /= np.linalg.norm(d)
    tail, length = np.array([1.0, 2.0, 3.0]), 0.7
    fast = scene._instanced(scene._arrow_template(), tail[None, :], d[None, :],
                            np.array([length]))
    slow = pv.Arrow(start=tail, direction=d, scale=length, tip_length=0.26,
                    tip_radius=0.06, shaft_radius=0.014, tip_resolution=24,
                    shaft_resolution=16)
    assert fast.n_points == slow.n_points and fast.n_cells == slow.n_cells
    # 箭头绕自身轴对称，旋转角可以不同，但轮廓一致：尖端、面积、离轴距离。
    # 容差按 float32 取：pv.Arrow 的顶点是单精度，它自己在不同方向下算出的
    # 面积就相差 1e-6 量级；实例化版本用双精度，反而各方向完全一致。
    tip = tail + length * d
    assert np.min(np.linalg.norm(fast.points - tip, axis=1)) < 1e-6
    assert fast.area == pytest.approx(slow.area, rel=1e-5)
    radial = lambda m: np.linalg.norm(np.cross(m.points - tail, d), axis=1)  # noqa: E731
    assert np.sort(radial(fast)) == pytest.approx(np.sort(radial(slow)), abs=1e-6)


def test_load_arrows_cover_the_same_surface_as_one_pv_arrow_each():
    """整组箭头与"每个站点一个 pv.Arrow 再 merge"的旧画法面积、包围盒一致。

    不比点数：旧画法 merge 时默认合并重合点，点数本来就对不上。
    """
    s = solved_session()
    frame = s.frame
    case = next(iter(frame.load_cases))
    got = scene.load_arrows(frame, case, per_member=7)["杆间荷载"]
    span = scene.model_size(frame)
    old = []
    for mid, member in frame.members.items():
        w = frame.load_cases[case].member_loads.get(mid)
        if w is None:
            continue
        pi, pj = scene.member_endpoints(frame, member)
        v = np.asarray(w, dtype=float)
        length = span * 0.06                     # 满跨均布：各站同一量值
        direction = v / np.linalg.norm(v)
        for t in np.linspace(0.15, 0.85, 7):
            tail = pi + t * (pj - pi) - direction * length
            old.append(pv.Arrow(start=tail, direction=direction, scale=length,
                                tip_length=0.26, tip_radius=0.06, shaft_radius=0.014,
                                tip_resolution=24, shaft_resolution=16))
    assert old, "测试模型里得有杆间荷载"
    reference = old[0].merge(old[1:])
    # 新画法多出每根杆顶上那条连线；扣掉它之后就是箭头本身
    arrows = len(old) * scene._arrow_template().area * (span * 0.06) ** 2
    assert got.area >= arrows
    assert arrows == pytest.approx(reference.area, rel=1e-5)
    assert np.allclose(got.bounds, reference.bounds, atol=span * 2e-3)


# --------------------------------------------------------------------------- 云图切段

def _cut_reference(line, name, edges):
    """原来的逐段写法，逐字搬过来当参照。"""
    values = np.asarray(line[name], dtype=float)
    points = np.asarray(line.points, dtype=float)
    edges = list(edges)
    connectivity = np.asarray(line.lines, dtype=int)
    cursor = 0
    while cursor < connectivity.size:
        count = int(connectivity[cursor])
        ids = connectivity[cursor + 1: cursor + 1 + count]
        cursor += count + 1
        for a, b in zip(ids[:-1], ids[1:], strict=False):
            pa, pb = points[a], points[b]
            va, vb = float(values[a]), float(values[b])
            cuts = [0.0, 1.0]
            if va != vb:
                lower, upper = (va, vb) if va < vb else (vb, va)
                cuts += [(edge - va) / (vb - va)
                         for edge in edges if lower < edge < upper]
            cuts = sorted(set(round(c, 12) for c in cuts))
            for t0, t1 in zip(cuts[:-1], cuts[1:], strict=False):
                mid = 0.5 * (t0 + t1)
                yield (pa + t0 * (pb - pa), pa + t1 * (pb - pa),
                       va + mid * (vb - va))


def test_band_cutting_matches_the_segment_by_segment_version():
    s = solved_session()
    line = scene.contour_line(s.frame, s.solution, None, "Mz")
    clim = scene.contour_clim(line, "Mz")
    levels = 12
    step = (clim[1] - clim[0]) / levels
    edges = [clim[0] + k * step for k in range(1, levels)]
    expected = list(_cut_reference(line, "Mz", edges))
    starts, ends, values = scene.cut_arrays(line, "Mz", edges)
    assert len(values) == len(expected)
    for k, (pa, pb, value) in enumerate(expected):
        assert np.allclose(starts[k], pa, atol=1e-12)
        assert np.allclose(ends[k], pb, atol=1e-12)
        assert values[k] == pytest.approx(value, abs=1e-9 * max(1.0, abs(value)))


def test_band_cutting_handles_flat_segments_and_no_edges():
    line = scene._polylines([np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0.0]])])
    line["v"] = np.array([1.0, 1.0, 3.0])
    got = list(scene.cut_at(line, "v", [2.0]))
    expected = list(_cut_reference(line, "v", [2.0]))
    assert len(got) == len(expected) == 3
    assert list(scene.cut_at(line, "v", [])) and len(list(scene.cut_at(line, "v", []))) == 2


def test_out_of_range_tubes_only_cover_the_saturated_part():
    line = scene._polylines([np.array([[0, 0, 0], [1, 0, 0.0]])])
    line["v"] = np.array([0.0, 2.0])
    tube = scene.out_of_range_tubes(line, "v", (0.0, 1.0), radius=0.01)
    assert tube.n_points > 0
    assert tube.bounds[0] == pytest.approx(0.5, abs=1e-6)     # 从 v=1 处开始


# --------------------------------------------------------------------------- Session 缓存

def test_preview_frame_is_reused_until_the_model_changes():
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    assert s.generate_frame(spans=[6.0], storeys=[3.0], column_section="COL",
                            beam_section="BEAM", material="Q355").ok
    first = s.preview_frame()
    assert s.preview_frame() is first
    s.model["nodes"][0]["x"] += 0.5                  # 就地修改，没有任何"改过了"标记
    second = s.preview_frame()
    assert second is not first
    assert second.nodes[s.model["nodes"][0]["id"]].x == pytest.approx(
        s.model["nodes"][0]["x"])


def test_an_invalid_model_keeps_raising_from_the_cache():
    s = Session()
    s.model = {"nodes": [{"id": 1, "x": 0, "y": 0, "z": 0}], "members": [{"id": 1}]}
    for _ in range(2):
        with pytest.raises(ValueError):
            s.preview_frame()


def test_validation_errors_are_cached_but_handed_out_as_copies():
    s = Session()
    s.model = {"nodes": [{"id": 1, "x": 0, "y": 0, "z": 0}], "members": [{"id": 1}]}
    errors = s.validation_errors()
    assert errors
    errors.clear()
    assert s.validation_errors(), "调用方清空了副本，缓存不能跟着空"


# --------------------------------------------------------------------------- 结果缓存

def test_member_results_are_cached_per_solution_and_returned_as_copies():
    s = solved_session()
    frame, solution = s.frame, s.solution
    mid = sorted(frame.members)[0]
    x1, d1 = internal_forces.member_displacement(frame, solution, mid)
    d1[:] = 0.0                                          # 调用方乱改
    x2, d2 = internal_forces.member_displacement(frame, solution, mid)
    assert np.abs(d2).max() > 0, "缓存被调用方改坏了"
    g1 = internal_forces.member_diagram(frame, solution, mid)
    g1.Mz[:] = 0.0
    g2 = internal_forces.member_diagram(frame, solution, mid)
    assert np.abs(g2.Mz).max() > 0


def test_a_new_solution_does_not_reuse_the_old_results():
    s = solved_session()
    mid = sorted(s.frame.members)[0]
    before = internal_forces.member_diagram(s.frame, s.solution, mid).Mz.copy()
    for item in s.model["member_loads"]:
        item["w"] = [2 * v for v in item["w"]]
    assert s.solve_model().ok
    after = internal_forces.member_diagram(s.frame, s.solution, mid).Mz
    assert np.allclose(after, 2 * before, rtol=1e-9, atol=1e-9)


# --------------------------------------------------------------------------- 模型检查

def test_coincident_nodes_are_reported_in_insertion_order():
    f = Frame()
    f.materials["M"] = Material("M", 2e11, 0.3)
    f.sections["S"] = Section("S", 1e-2, 1e-4, 1e-4, 1e-5)
    for nid, xyz in ((5, (0, 0, 0)), (1, (1, 0, 0)), (9, (0, 0, 0)), (3, (1, 0, 0))):
        f.nodes[nid] = Node(nid, *xyz)
    f.members[1] = Member(1, 5, 1, "S", "M")
    f.members[2] = Member(2, 9, 3, "S", "M")
    issues = [i for i in check_model(copy.deepcopy(f)) if "坐标重合" in i]
    assert issues == ["节点 5 与节点 9 坐标重合；请合并为同一节点",
                      "节点 1 与节点 3 坐标重合；请合并为同一节点"]


# --------------------------------------------------------------------------- 三维内力图

def _simple_beam(load):
    f = Frame()
    f.sections["S"] = Section("S", 1e-2, 1e-4, 1e-4, 1e-5)
    f.materials["M"] = Material("M", 2e11, 0.3)
    f.nodes[1] = Node(1, 0, 0, 0)
    f.nodes[2] = Node(2, 6, 0, 0)
    f.members[1] = Member(1, 1, 2, "S", "M")
    f.supports[1] = (1, 1, 1, 1, 0, 0)
    f.supports[2] = (0, 1, 1, 0, 0, 0)
    f.member_loads[1] = load
    from frame3d import solve
    return f, solve(f)


def _tips(got):
    points = got["ribbon"].points
    return points[len(points) // 2:]


def test_the_moment_diagram_is_drawn_on_the_tension_side():
    """竖向荷载下简支梁下缘受拉：弯矩图画在梁的下方（-Z），峰值在跨中。"""
    frame, solution = _simple_beam((0, 0, -1e4))
    got = scene.force_diagram(frame, solution, None, "M")
    tips = _tips(got)
    assert tips[:, 2].max() <= 1e-12 and tips[:, 2].min() < 0.0
    assert got["peak"]["value"] == pytest.approx(45e3)
    assert got["peak"]["x"] == pytest.approx(3.0)
    # 峰值画出去的长度就是模型尺寸的 DIAGRAM_SIZE_RATIO
    assert abs(tips[:, 2].min()) == pytest.approx(
        scene.DIAGRAM_SIZE_RATIO * scene.model_size(frame))


def test_a_sideways_load_bends_the_diagram_toward_its_tension_side():
    """水平 +Y 荷载下梁向 +Y 弯曲，+Y 侧受拉：My 图画向 +Y。"""
    frame, solution = _simple_beam((0, 1e4, 0))
    got = scene.force_diagram(frame, solution, None, "M")
    tips = _tips(got)
    assert tips[:, 1].min() >= -1e-12 and tips[:, 1].max() > 0.0


def test_shear_changes_side_at_its_zero_without_twisting_the_ribbon():
    """剪力跨中过零：两半分画在杆的两侧，过零处插了零点，四边形不打结。"""
    frame, solution = _simple_beam((0, 0, -1e4))
    got = scene.force_diagram(frame, solution, None, "Vy")
    values = np.asarray(got["ribbon"]["value"])
    assert values.min() < 0.0 < values.max()
    tips = _tips(got)
    assert tips[:, 2].min() < 0.0 < tips[:, 2].max()
    x, v = scene._with_zero_crossings(np.array([0.0, 1.0, 2.0]),
                                      np.array([2.0, -2.0, -1.0]))
    assert list(x) == [0.0, 0.5, 1.0, 2.0] and list(v) == [2.0, 0.0, -2.0, -1.0]


def test_a_member_with_constant_force_gets_a_rectangle():
    """轴力、无跨中荷载时的剪力沿杆不变：画出来是等高的矩形，大小一眼可读。"""
    s = solved_session()
    got = scene.force_diagram(s.frame, s.solution, None, "N")
    assert got["ribbon"].n_cells > 0 and got["peak"]["member"] is not None
