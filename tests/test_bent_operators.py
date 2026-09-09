"""一榀刚架、纵向拉伸与建模算子。

**这是对"可实现的刚架种类太少"的正面回答。**

原来只有两个生成器（正交网格、坡屋面门式），中间到"逐个列节点坐标"之间
什么都没有。模板是加法，算子是乘法：一榀 → 拉伸 → 算子。

验的重点有三层：

1. **几何对不对**——各种屋面形状、错层、悬挑都要生成得出来；
2. **算得对不对**——同一个门式刚架用一榀生成器和用原门式生成器
   必须给出**完全相同**的结果，这是新代码唯一站得住的参照；
3. **算子改的是不是它该改的**——加撑要真的降低侧移，抽柱要真的变柔。
"""

from __future__ import annotations

import math

import pytest

import bent
from agent import Session
from generator import GeneratorError
from sections import i_section

MAT = [{"name": "Q355", "E": 2.06e11, "nu": 0.3, "density": 7850.0}]
SEC = [i_section("COLUMN", .4, .2, .008, .012),
       i_section("BEAM", .5, .2, .008, .014),
       i_section("BRACE", .15, .15, .006, .008)]
SPAN, EAVE, RISE = 24.0, 7.5, 1.2


def drift(model, node_filter, load_nodes) -> float:
    """在指定节点上加水平力，量另一个节点的 X 位移。"""
    s = Session()
    s.define_materials_and_sections(MAT, SEC)
    assert s.set_model(model=model).ok
    s.set_load_cases(cases=[{"name": "W", "nodal_loads":
                             [{"node": n, "load": [30e3, 0, 0, 0, 0, 0]}
                              for n in load_nodes]}])
    assert s.solve_model().ok
    watch = node_filter(model)
    return abs(float(s.solution["W"].U[s.frame.node_dofs(watch)[0]]) * 1000)


def top_nodes(model) -> list[int]:
    z = max(n["z"] for n in model["nodes"])
    return [n["id"] for n in model["nodes"] if abs(n["z"] - z) < 1e-9]


def corner(model) -> int:
    z = max(n["z"] for n in model["nodes"])
    return next(n["id"] for n in model["nodes"]
                if abs(n["x"]) < 1e-9 and abs(n["y"]) < 1e-9
                and abs(n["z"] - z) < 1e-9)


# ------------------------------------------------- 一榀：几何

def test_a_flat_roof_bent():
    m, ids = bent.generate_bent(profile=[(0, 7.5), (24, 7.5)], columns=[0, 24])
    assert len(m["nodes"]) == 4
    assert len(ids["column_member_ids"]) == 2
    assert len(ids["rafter_member_ids"]) == 1


def test_a_gable_roof_comes_from_the_profile_alone():
    """双坡不需要单独的生成器——**折线本身就是屋面形状**。"""
    m, ids = bent.generate_bent(
        profile=[(0, 7.5), (12, 8.7), (24, 7.5)], columns=[0, 24])
    ridge = [n for n in m["nodes"] if abs(n["x"] - 12) < 1e-9]
    assert len(ridge) == 1 and ridge[0]["z"] == pytest.approx(8.7)
    assert len(ids["rafter_member_ids"]) == 2


def test_a_mono_pitch_roof():
    m, _ = bent.generate_bent(profile=[(0, 6.0), (24, 9.0)], columns=[0, 24])
    tops = sorted(n["z"] for n in m["nodes"] if n["z"] > 1e-9)
    assert tops == pytest.approx([6.0, 9.0])


def test_a_cantilever_is_just_a_longer_profile():
    """悬挑 = 折线伸到柱子外面。不必为它再写一个生成器。"""
    m, ids = bent.generate_bent(
        profile=[(-2, 7.5), (0, 7.5), (24, 7.5), (26, 7.5)], columns=[0, 24])
    xs = sorted(n["x"] for n in m["nodes"])
    assert xs[0] == pytest.approx(-2.0) and xs[-1] == pytest.approx(26.0)
    assert len(ids["rafter_member_ids"]) == 3


def test_split_level_bases():
    """错层：柱脚标高逐根给。"""
    m, _ = bent.generate_bent(profile=[(0, 7.5), (12, 7.5)],
                              columns=[0, 6, 12], base_levels=[0, 1.5, 0])
    lows = sorted(n["z"] for n in m["nodes"] if n["z"] < 7.0)
    assert lows == pytest.approx([0.0, 0.0, 1.5])


def test_intermediate_levels_create_floor_beams():
    m, ids = bent.generate_bent(profile=[(0, 7.2), (12, 7.2)],
                                columns=[0, 6, 12], levels=[3.6])
    assert len(ids["beam_member_ids"]) == 2      # 两跨，一层楼面
    assert len(ids["column_member_ids"]) == 6    # 三柱各断成两段


def test_absolute_floor_levels_may_be_zero_or_negative():
    """绝对标高不是层高；地下层和以任意基准点建模时必须允许 0/负数。"""
    model, ids = bent.generate_bent(
        profile=[(0, 3.0), (12, 3.0)], columns=[0, 6, 12],
        levels=[-1.0, 0.0, 1.5], base_levels=[-3.0, -3.0, -3.0])
    assert len(ids["beam_member_ids"]) == 6
    assert {-1.0, 0.0, 1.5}.issubset({node["z"] for node in model["nodes"]})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"profile": [(0, 3), (float("nan"), 3)], "columns": [0, 6]},
        {"profile": [(0, 3), (6, 3)], "columns": [0, float("inf")]},
        {"profile": [(0, 3), (6, 3)], "columns": [0, 6],
         "base_levels": [0, float("nan")]},
    ],
)
def test_irregular_generator_rejects_nonfinite_geometry(kwargs):
    with pytest.raises(GeneratorError, match="有限"):
        bent.generate_bent(**kwargs)


def test_a_column_outside_the_profile_is_refused():
    """外推出来的柱顶标高看着是对的，**但它不在结构上**，很难发现。"""
    with pytest.raises(GeneratorError, match="之外"):
        bent.generate_bent(profile=[(0, 7.5), (12, 7.5)], columns=[0, 20])


def test_a_zero_length_column_is_refused():
    with pytest.raises(GeneratorError, match="柱长"):
        bent.generate_bent(profile=[(0, 3.0), (12, 3.0)], columns=[0, 12],
                           base_levels=[0, 3.0])


def test_coincident_nodes_are_merged():
    """折线拐点和柱顶常常重合。**各建一个就会出现两个位置相同的节点**——
    求解不报错，但那根杆等于没连上。"""
    m, _ = bent.generate_bent(profile=[(0, 7.5), (12, 8.7), (24, 7.5)],
                              columns=[0, 12, 24])
    seen = {(round(n["x"], 9), round(n["z"], 9)) for n in m["nodes"]}
    assert len(seen) == len(m["nodes"])


# ------------------------------------------------- 一榀：算得对不对

def test_the_bent_matches_the_existing_portal_generator():
    """**同一个门式刚架，两条独立路径必须给出完全相同的结果。**

    这是新生成器唯一站得住的验证方式——自己跟自己比说明不了任何问题。
    """
    def run(build):
        s = Session()
        s.define_materials_and_sections(
            MAT, [i_section("COLUMN", .7, .25, .010, .016),
                  i_section("RAFTER", .9, .25, .010, .020)])
        rafters = build(s)
        s.set_load_cases(cases=[{"name": "D", "member_loads":
                                 [{"member": m, "w": [0, 0, -8e3]}
                                  for m in rafters]}])
        assert s.solve_model().ok
        return (s.query_results(what="max_displacement").payload["magnitude_mm"],
                s.query_results(what="max_deflection").payload["magnitude_mm"])

    def by_portal(s):
        g = s.generate_portal_frame(
            spans=[SPAN], eave_height=EAVE, ridge_rise=RISE,
            column_section="COLUMN", rafter_section="RAFTER",
            material="Q355", base="pinned")
        return g.payload["rafter_member_ids"]

    def by_bent(s):
        m, ids = bent.generate_bent(
            profile=[(0, EAVE), (SPAN / 2, EAVE + RISE), (SPAN, EAVE)],
            columns=[0, SPAN], column_section="COLUMN",
            beam_section="RAFTER", material="Q355", base="pinned")
        assert s.set_model(model=m).ok
        return ids["rafter_member_ids"]

    a, b = run(by_portal), run(by_bent)
    assert a[0] == pytest.approx(b[0], rel=1e-9), (a, b)
    assert a[1] == pytest.approx(b[1], rel=1e-9), (a, b)


def test_the_ids_are_not_part_of_the_model():
    """编号清单不能塞进模型：schema 是 additionalProperties: False，
    塞进去会被直接拒掉——几何和约束全对，却卡在校验上。"""
    from model_io import validate_payload

    m, ids = bent.generate_bent(profile=[(0, 7.5), (24, 7.5)], columns=[0, 24])
    m["units"] = "N-m-Pa"
    m["materials"] = MAT
    m["sections"] = [SEC[0]]
    for member in m["members"]:
        member["section"] = "COLUMN"
        member["material"] = "Q355"
    assert validate_payload(m) == []
    assert "column_member_ids" not in m and "column_member_ids" in ids


# ------------------------------------------------- 拉伸

def test_extrusion_makes_one_bent_per_bay_line():
    one, _ = bent.generate_bent(profile=[(0, 7.2), (12, 7.2)],
                                columns=[0, 6, 12], levels=[3.6])
    many, info = bent.extrude_bents(one, bays=[6, 6], tie_section="BEAM")
    assert info["bent_count"] == 3
    assert len(many["nodes"]) == 3 * len(one["nodes"])
    ys = sorted({n["y"] for n in many["nodes"]})
    assert ys == pytest.approx([0.0, 6.0, 12.0])


def test_unequal_bays_are_supported():
    """山墙开间往往小一些。**只给一个间距和一个数量就做不出来。**"""
    one, _ = bent.generate_bent(profile=[(0, 7.2), (6, 7.2)], columns=[0, 6])
    many, _ = bent.extrude_bents(one, bays=[4.5, 6, 6, 4.5])
    ys = sorted({n["y"] for n in many["nodes"]})
    assert ys == pytest.approx([0.0, 4.5, 10.5, 16.5, 21.0])


def test_ties_connect_corresponding_nodes_only():
    """连系梁只连同标高的对应节点。按"所有节点两两相连"来，
    会连出一堆穿过室内空间的斜杆。"""
    one, _ = bent.generate_bent(profile=[(0, 7.2), (6, 7.2)], columns=[0, 6])
    many, info = bent.extrude_bents(one, bays=[6])
    by_id = {n["id"]: n for n in many["nodes"]}
    for mid in info["tie_member_ids"]:
        m = next(x for x in many["members"] if x["id"] == mid)
        a, b = by_id[m["i"]], by_id[m["j"]]
        assert a["x"] == pytest.approx(b["x"])
        assert a["z"] == pytest.approx(b["z"])
        assert a["y"] != pytest.approx(b["y"])


def test_ties_do_not_connect_column_bases():
    """柱脚在地上，连系梁连柱脚没有意义。"""
    one, _ = bent.generate_bent(profile=[(0, 7.2), (6, 7.2)], columns=[0, 6])
    many, info = bent.extrude_bents(one, bays=[6])
    by_id = {n["id"]: n for n in many["nodes"]}
    for mid in info["tie_member_ids"]:
        m = next(x for x in many["members"] if x["id"] == mid)
        assert by_id[m["i"]]["z"] > 1e-9


def test_out_of_plane_restraints_are_dropped_after_extrusion():
    """**拉伸之后不再是平面刚架，面外约束必须去掉。**

    留着的话整个结构在 Y 方向被钉死，横向刚度算出来是假的——
    而且不报错，只是结果偏刚。这是拉伸这一步最容易出的错。
    """
    one, _ = bent.generate_bent(profile=[(0, 7.2), (6, 7.2)], columns=[0, 6])
    many, _ = bent.extrude_bents(one, bays=[6])
    by_id = {n["id"]: n for n in many["nodes"]}
    for s in many["supports"]:
        assert by_id[s["node"]]["z"] < 1e-9, "非柱脚不该还留着约束"


def test_pinned_column_bases_remain_pinned_after_extrusion():
    """平面分析用的 rx/rz 人工约束不能混进空间模型的真实柱脚。"""
    one, _ = bent.generate_bent(
        profile=[(0, 7.2), (6, 7.2)], columns=[0, 6], base="pinned")
    many, _ = bent.extrude_bents(one, bays=[6])
    assert many["supports"]
    assert all(s["fix"] == [1, 1, 1, 0, 0, 0] for s in many["supports"])


def test_extrusion_does_not_assume_contiguous_source_node_ids():
    one, _ = bent.generate_bent(profile=[(0, 7.2), (6, 7.2)], columns=[0, 6])
    remap = {node["id"]: node["id"] * 10 for node in one["nodes"]}
    for node in one["nodes"]:
        node["id"] = remap[node["id"]]
    for member in one["members"]:
        member["i"], member["j"] = remap[member["i"]], remap[member["j"]]
    for support in one["supports"]:
        support["node"] = remap[support["node"]]

    many, _ = bent.extrude_bents(one, bays=[6])
    node_ids = {node["id"] for node in many["nodes"]}
    assert node_ids == set(range(1, len(node_ids) + 1))
    assert all(m["i"] in node_ids and m["j"] in node_ids for m in many["members"])


def test_repeated_extrusion_is_refused_instead_of_flattening_the_model():
    one, _ = bent.generate_bent(profile=[(0, 7.2), (6, 7.2)], columns=[0, 6])
    many, _ = bent.extrude_bents(one, bays=[6])
    with pytest.raises(bent.GeneratorError, match="多个 Y 平面"):
        bent.extrude_bents(many, bays=[6])


# ------------------------------------------------- 算子

def test_selecting_by_geometry_not_by_id():
    """**算子的输入靠几何条件，不是让用户报编号**——
    编号规则是生成器定的，用户并不知道 27 号在哪。"""
    one, _ = bent.generate_bent(profile=[(0, 7.2), (12, 7.2)],
                                columns=[0, 6, 12], levels=[3.6])
    columns = bent.select_members(one, orientation="vertical")
    beams = bent.select_members(one, orientation="horizontal")
    assert len(columns) == 6 and len(beams) == 4
    lower = bent.select_members(one, orientation="vertical", z_range=(0, 3.6))
    assert len(lower) == 3


def test_bracing_actually_stiffens_the_braced_frame():
    """加一道 X 撑对侧移的改善，往往比把所有柱加大一号还明显。"""
    one, _ = bent.generate_bent(profile=[(0, 7.2), (12, 7.2)],
                                columns=[0, 6, 12], levels=[3.6],
                                column_section="COLUMN", beam_section="BEAM",
                                material="Q355")
    many, _ = bent.extrude_bents(one, bays=[6, 6], tie_section="BEAM")
    braced, info = bent.add_bracing(many, kind="X", x_range=(0, 6),
                                    y_range=(-0.1, 0.1), section="BRACE")
    loads = top_nodes(many)
    before = drift(many, corner, loads)
    after = drift(braced, corner, loads)
    assert after < 0.3 * before, (before, after)
    assert len(info["brace_member_ids"]) == 4    # 两层各一对交叉撑


def test_a_local_measure_is_needed_not_the_global_maximum():
    """**局部加固时"全局最大位移"是个误导性指标。**

    只撑一榀，全结构最大位移会跑到没撑的那几榀去，看起来几乎没改善——
    我第一次验证就栽在这上面，差点以为支撑没生效。
    """
    one, _ = bent.generate_bent(profile=[(0, 7.2), (12, 7.2)],
                                columns=[0, 6, 12], levels=[3.6],
                                column_section="COLUMN", beam_section="BEAM",
                                material="Q355")
    many, _ = bent.extrude_bents(one, bays=[6, 6], tie_section="BEAM")
    braced, _ = bent.add_bracing(many, kind="X", x_range=(0, 6),
                                 y_range=(-0.1, 0.1), section="BRACE")

    def global_max(model):
        s = Session()
        s.define_materials_and_sections(MAT, SEC)
        s.set_model(model=model)
        s.set_load_cases(cases=[{"name": "W", "nodal_loads":
                                 [{"node": n, "load": [30e3, 0, 0, 0, 0, 0]}
                                  for n in top_nodes(many)]}])
        s.solve_model()
        return s.query_results(what="max_displacement",
                               case="W").payload["magnitude_mm"]

    # 被撑那一榀明显变刚
    assert drift(braced, corner, top_nodes(many)) < \
        0.3 * drift(many, corner, top_nodes(many))
    # 但全结构最大值几乎没动——这正是这条测试要说明的
    assert global_max(braced) > 0.8 * global_max(many)


def test_braces_carry_pure_axial_force():
    """支撑靠轴力工作。铰接建模之后轴力沿杆长应当是常数，
    **按刚接建会高估它对弯矩的贡献**。"""
    one, _ = bent.generate_bent(profile=[(0, 7.2), (12, 7.2)],
                                columns=[0, 6, 12], levels=[3.6],
                                column_section="COLUMN", beam_section="BEAM",
                                material="Q355")
    many, _ = bent.extrude_bents(one, bays=[6], tie_section="BEAM")
    braced, info = bent.add_bracing(many, kind="X", x_range=(0, 6),
                                    y_range=(-0.1, 0.1), section="BRACE")
    s = Session()
    s.define_materials_and_sections(MAT, SEC)
    s.set_model(model=braced)
    s.set_load_cases(cases=[{"name": "W", "nodal_loads":
                             [{"node": n, "load": [30e3, 0, 0, 0, 0, 0]}
                              for n in top_nodes(many)]}])
    assert s.solve_model().ok
    from internal_forces import member_diagram
    tension = compression = 0
    for mid in info["brace_member_ids"]:
        d = member_diagram(s.frame, s.solution, mid, "W", stations=11)
        assert max(d.N) - min(d.N) < 1e-6 * max(abs(max(d.N)), 1.0), "轴力不是常数"
        if d.N[0] > 0:
            tension += 1
        else:
            compression += 1
    assert tension and compression, "交叉撑应当一拉一压"


def test_removing_a_column_makes_the_frame_softer():
    one, _ = bent.generate_bent(profile=[(0, 7.2), (12, 7.2)],
                                columns=[0, 6, 12], levels=[3.6],
                                column_section="COLUMN", beam_section="BEAM",
                                material="Q355")
    many, _ = bent.extrude_bents(one, bays=[6, 6], tie_section="BEAM")
    mid_cols = bent.select_members(many, orientation="vertical",
                                   x_range=(5.9, 6.1), z_range=(0, 3.6))
    cut, info = bent.remove_members_by(many, mid_cols)
    assert len(info["removed_members"]) == 3
    loads = top_nodes(many)
    assert drift(cut, corner, loads) > drift(many, corner, loads)


def test_removing_members_cleans_up_orphan_nodes():
    """**留着孤立节点，刚度矩阵就有一整行零，求解直接奇异。**
    用户以为自己只是"删了一根柱"，却得到一句"矩阵奇异"，完全对不上。"""
    one, _ = bent.generate_bent(profile=[(0, 7.2), (6, 7.2)], columns=[0, 6])
    cut, info = bent.remove_members_by(one, [1])       # 删掉一根柱
    used = {m["i"] for m in cut["members"]} | {m["j"] for m in cut["members"]}
    assert all(n["id"] in used for n in cut["nodes"])
    assert info["removed_orphan_nodes"]


def test_raising_a_region_moves_only_that_region():
    """"把 X=10~20m 范围内的上弦提高 1m" 这类指令的落点。"""
    one, _ = bent.generate_bent(profile=[(0, 7.2), (6, 7.2), (12, 7.2)],
                                columns=[0, 6, 12])
    raised, info = bent.raise_nodes(one, dz=1.0, x_range=(5.9, 6.1),
                                    z_range=(7.0, 8.0))
    moved = {n["id"] for n in raised["nodes"]
             if n["z"] > 7.2 + 1e-9}
    assert moved == set(info["moved_nodes"])
    assert len(moved) == 1


def test_raising_nothing_is_an_error_not_a_silent_no_op():
    """框错了位置却什么都没发生，用户会以为算子坏了。"""
    one, _ = bent.generate_bent(profile=[(0, 7.2), (6, 7.2)], columns=[0, 6])
    with pytest.raises(GeneratorError, match="没有节点"):
        bent.raise_nodes(one, dz=1.0, x_range=(100, 200))


def test_retaper_changes_only_the_named_members():
    one, ids = bent.generate_bent(profile=[(0, 7.2), (6, 7.2)], columns=[0, 6],
                                  column_section="COLUMN", beam_section="BEAM")
    out, info = bent.retaper(one, ids["column_member_ids"], "BRACE")
    for m in out["members"]:
        expect = "BRACE" if m["id"] in ids["column_member_ids"] else "BEAM"
        assert m["section"] == expect


def test_operators_do_not_mutate_the_input():
    """算子要能串起来用，**改坏输入会让上一步的模型悄悄变样**。"""
    one, ids = bent.generate_bent(profile=[(0, 7.2), (6, 7.2)], columns=[0, 6])
    before = len(one["members"]), len(one["nodes"])
    bent.remove_members_by(one, [1])
    bent.raise_nodes(one, dz=1.0)
    bent.retaper(one, [1], "X")
    assert (len(one["members"]), len(one["nodes"])) == before


# ------------------------------------------------- 接进 Session 之后

def test_the_operators_are_recorded_in_the_build_history():
    """**漏登记的话撤销和时间线就看不见它们**——
    用户抽了一根柱，按 Ctrl+Z 却什么都没发生。"""
    s = Session()
    s.define_materials_and_sections(MAT, SEC)
    s.generate_bent(profile=[[0, 7.2], [12, 7.2]], columns=[0, 6, 12],
                    levels=[3.6], column_section="COLUMN",
                    beam_section="BEAM", material="Q355")
    s.extrude_bents(bays=[6, 6], tie_section="BEAM")
    s.add_bracing(kind="X", x_range=[-0.1, 6.1], y_range=[-0.1, 0.1],
                  section="BRACE")
    s.raise_nodes(dz=0.3, z_range=[7.0, 7.4])
    tools = [s.history[k].tool for k in range(len(s.history))]
    for name in ("generate_bent", "extrude_bents", "add_bracing", "raise_nodes"):
        assert name in tools, f"{name} 没进建模过程"
    assert s.history.can_undo


def test_every_history_summary_is_readable():
    """摘要取错键不会报错，只会一直显示 None——
    这类 bug 没人会专门去看，得靠测试盯住。`remove_members` 就栽过：
    参数名是 ids，摘要里写的是 members。"""
    s = Session()
    s.define_materials_and_sections(MAT, SEC)
    s.generate_bent(profile=[[0, 7.2], [12, 7.2]], columns=[0, 6, 12],
                    levels=[3.6], column_section="COLUMN",
                    beam_section="BEAM", material="Q355")
    s.extrude_bents(bays=[6], tie_section="BEAM")
    cols = s.select_members(orientation="vertical", x_range=[5.9, 6.1],
                            z_range=[0, 3.6]).payload["member_ids"]
    s.remove_members(ids=cols)
    s.add_bracing(kind="X", x_range=[-0.1, 6.1], y_range=[-0.1, 0.1],
                  section="BRACE")
    s.raise_nodes(dz=0.3, z_range=[7.0, 7.4])
    s.retaper(member_ids=[1], section="BRACE")
    for step in s.history.timeline():
        text = step["做了什么"]
        assert "None" not in text, f"第 {step['步']} 步摘要里有 None：{text}"
        assert len(text) > 4, text


def test_generate_bent_keeps_the_unit_system():
    """**换模型时 units 必须一起带过来。** 漏了之后模型立刻不合规，
    而报错要等到下一步操作才冒出来，用户看不出是哪一步的问题。"""
    s = Session()
    s.define_materials_and_sections(MAT, SEC)
    s.generate_bent(profile=[[0, 7.2], [12, 7.2]], columns=[0, 12],
                    column_section="COLUMN", beam_section="BEAM",
                    material="Q355")
    assert s.model.get("units"), "units 丢了"
    from model_io import validate_payload
    assert validate_payload(s.model) == []


def test_removing_members_through_the_session_cleans_orphans():
    s = Session()
    s.define_materials_and_sections(MAT, SEC)
    g = s.generate_frame(spans=[6.0, 6.0], storeys=[3.6],
                         column_section="COLUMN", beam_section="BEAM",
                         material="Q355")
    before = len(s.model["nodes"])
    cols = s.select_members(orientation="vertical",
                            x_range=[-0.1, 0.1]).payload["member_ids"]
    r = s.remove_members(ids=cols)
    assert r.ok, r.payload
    assert r.payload["removed_orphan_nodes"]
    assert len(s.model["nodes"]) < before


def test_a_full_chain_solves():
    """一榀 → 拉伸 → 抽柱 → 加撑 → 求解，全程走 Session。"""
    s = Session()
    s.define_materials_and_sections(MAT, SEC)
    s.generate_bent(profile=[[0, 7.2], [12, 7.2]], columns=[0, 6, 12],
                    levels=[3.6], column_section="COLUMN",
                    beam_section="BEAM", material="Q355")
    s.extrude_bents(bays=[6, 6], tie_section="BEAM")
    cols = s.select_members(orientation="vertical", x_range=[5.9, 6.1],
                            z_range=[0, 3.6]).payload["member_ids"]
    assert s.remove_members(ids=cols).ok
    assert s.add_bracing(kind="X", x_range=[-0.1, 6.1], y_range=[-0.1, 0.1],
                         section="BRACE").ok
    beams = s.select_members(orientation="horizontal").payload["member_ids"]
    s.set_load_cases(cases=[{"name": "D", "member_loads":
                             [{"member": m, "w": [0, 0, -15e3]} for m in beams]}])
    got = s.solve_model()
    assert got.ok, got.payload
