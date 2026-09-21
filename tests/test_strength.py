"""强度验算与逐杆稳定校核（讲义 §3-9 三）。

判据全是闭式解：

* 轴力杆 σ = N/A，应力比 = σ/[σ]
* 两端铰支压杆 ``Pcr = π²EI/L²``，μ=1
* μ 按两端约束取 1 / 0.7 / 0.5（讲义四种里能从杆端释放看出来的三种）

三条盯的是**不报错的失效方式**：

1. 拉压许用不同的材料，用 |σ| 除一个许用值会判错——错的那一侧还偏不安全。
2. 杆件被自动剖分后若按分段长度算 Pcr，两等分就是四倍，结果偏大且看着合理。
3. 短粗杆的欧拉 σcr 超过屈服应力，是个没有物理意义的大数，照着它判「稳定够」
   最危险。
"""

from __future__ import annotations

import math

import pytest

from frame3d import (Frame, LoadCase, Material, Member, Node, Section, solve)
from model_compiler import compile_model
from strength import (BUCKLING_NA, BUCKLING_OK, BUCKLING_SLENDER,
                      BUCKLING_UNKNOWN, MU_FIXED_FIXED, MU_FIXED_PINNED,
                      MU_PINNED_PINNED, StrengthUnavailable, check_member,
                      check_strength, effective_length_factor)

E = 2.0e11
AREA = 4.0e-3
IY = IZ = 8.0e-6
LENGTH = 5.0


def _sec(**kw):
    # cy/cz 必须给，否则 stress.py 直接拒绝算弯曲应力
    return Section("S", A=AREA, Iy=IY, Iz=IZ, J=1.6e-5, cy=0.05, cz=0.05, **kw)


def _bar(load, *, ft=100e6, fc=None, pinned=False, yield_stress=None) -> Frame:
    """沿 x 的单根杆，远端加轴向力 load（正为拉）。"""
    rel = ("ry", "rz") if pinned else ()
    frame = Frame(
        nodes={1: Node(1, 0.0, 0.0, 0.0), 2: Node(2, LENGTH, 0.0, 0.0)},
        members={1: Member(1, 1, 2, "S", "M", releases_i=rel, releases_j=rel)},
        sections={"S": _sec()},
        materials={"M": Material("M", E=E, nu=0.3, yield_stress=yield_stress,
                                 allow_tension=ft, allow_compression=fc)},
        supports={1: (1, 1, 1, 1, 1, 1), 2: (0, 1, 1, 1, 1, 1)},
        load_cases={"P": LoadCase("P", nodal_loads={2: [load, 0, 0, 0, 0, 0]})},
    )
    return frame


# ------------------------------------------------------------------ 强度

def test_axial_stress_ratio_is_exactly_sigma_over_allowable():
    load = 200e3
    frame = _bar(load)
    got = check_member(frame, solve(frame), 1)["strength"]
    sigma = load / AREA
    assert got["sigma_max"] == pytest.approx(sigma, rel=1e-12)
    assert got["ratio"] == pytest.approx(sigma / 100e6, rel=1e-12)
    assert got["governs"] == "受拉"
    assert got["ok"]


def test_tension_and_compression_allowables_are_used_separately():
    """抗拉抗压相差 3 倍的材料（铸铁那一类），同一个 |σ| 判出两个结果。

    如果实现里先取 |σ| 再除一个许用值，这两条必然有一条错——
    而且错的方向是「压杆判成通过」，正好是不安全的那一侧。
    """
    ft, fc = 40e6, 120e6
    sigma = 60e6
    load = sigma * AREA

    pulled = check_member(_bar(load, ft=ft, fc=fc),
                          solve(_bar(load, ft=ft, fc=fc)), 1)["strength"]
    pushed = check_member(_bar(-load, ft=ft, fc=fc),
                          solve(_bar(-load, ft=ft, fc=fc)), 1)["strength"]

    assert pulled["ratio"] == pytest.approx(sigma / ft, rel=1e-12)   # 1.5，超限
    assert not pulled["ok"] and pulled["governs"] == "受拉"
    assert pushed["ratio"] == pytest.approx(sigma / fc, rel=1e-12)   # 0.5，通过
    assert pushed["ok"] and pushed["governs"] == "受压"


def test_compression_allowable_defaults_to_tension_and_says_so():
    frame = _bar(-100e3, ft=100e6, fc=None)
    got = check_member(frame, solve(frame), 1)
    assert got["allow_compression"] == 100e6
    assert "取拉许用" in got["allow_source"]


def test_a_material_without_allowables_is_refused_not_guessed():
    frame = _bar(100e3)
    frame.materials["M"] = Material("M", E=E, nu=0.3)
    with pytest.raises(StrengthUnavailable, match="许用应力"):
        check_member(frame, solve(frame), 1)


# ------------------------------------------------------------------ 稳定

def test_pinned_column_matches_the_euler_formula_to_machine_precision():
    frame = _bar(-10e3, pinned=True)
    frame.members[1] = Member(1, 1, 2, "S", "M", releases_i=("ry", "rz"),
                              releases_j=("ry", "rz"), mu_y=1.0, mu_z=1.0)
    got = check_member(frame, solve(frame), 1)["buckling"]
    assert got["P_cr"] == pytest.approx(
        math.pi ** 2 * E * IY / LENGTH ** 2, rel=1e-12)
    assert got["P"] == pytest.approx(10e3, rel=1e-9)
    assert got["ratio"] == pytest.approx(got["P"] / got["P_cr"], rel=1e-12)
    assert got["slenderness"] == pytest.approx(
        LENGTH / math.sqrt(IY / AREA), rel=1e-12)


def test_the_four_classic_end_conditions():
    assert effective_length_factor(True, True) == MU_PINNED_PINNED
    assert effective_length_factor(True, False) == MU_FIXED_PINNED
    assert effective_length_factor(False, True) == MU_FIXED_PINNED
    assert effective_length_factor(False, False) == MU_FIXED_FIXED


def test_an_explicit_mu_wins_over_the_inferred_one_and_is_labelled():
    frame = _bar(-10e3)
    frame.members[1] = Member(1, 1, 2, "S", "M", mu_y=2.0, mu_z=2.0)
    got = check_member(frame, solve(frame), 1)["buckling"]
    assert got["axes"]["y"]["mu"] == 2.0
    assert got["axes"]["y"]["mu_source"] == "用户给定"
    assert got["P_cr"] == pytest.approx(
        math.pi ** 2 * E * IY / (2.0 * LENGTH) ** 2, rel=1e-12)

    frame.members[1] = Member(1, 1, 2, "S", "M")
    inferred = check_member(frame, solve(frame), 1)["buckling"]
    assert inferred["axes"]["y"]["mu"] == MU_FIXED_FIXED
    assert inferred["axes"]["y"]["mu_source"] == "由杆端释放推定"
    assert any("无侧移" in w for w in inferred["warnings"])


def test_a_stocky_column_is_flagged_as_outside_the_euler_range():
    """λ < λp 时欧拉 σcr 已超过屈服应力，不能拿它判「稳定够」。"""
    frame = _bar(-10e3, yield_stress=235e6)
    # 粗短：把回转半径放大到与杆长同量级，λ 掉到十几
    frame.sections["S"] = Section("S", A=AREA, Iy=1.0e-3, Iz=1.0e-3, J=2e-3,
                                  cy=0.05, cz=0.05)
    row = check_member(frame, solve(frame), 1)
    got = row["buckling"]
    assert got["lambda_p"] == pytest.approx(math.pi * math.sqrt(E / 235e6))
    assert got["euler_applicable"] is False
    # 三种结局要分清楚：通过、超限、**判不了**。这一档是判不了。
    assert got["status"] == BUCKLING_NA
    assert not got["conclusive"], "欧拉公式不适用时不能给出结论"
    assert not row["conclusive"] and row["ok"], "判不了不等于超限"
    assert got["ratio"] < 1.0, "本例的 Pcr 很大，正是「照着算就会误判通过」"
    assert any("不适用" in w for w in got["warnings"])


def test_without_a_yield_stress_the_applicability_is_unknown_not_assumed():
    frame = _bar(-10e3)
    got = check_member(frame, solve(frame), 1)["buckling"]
    assert got["euler_applicable"] is None
    assert got["status"] == BUCKLING_UNKNOWN and not got["conclusive"]
    assert any("无法判断" in w for w in got["warnings"])


def test_a_member_in_tension_gets_no_buckling_check():
    frame = _bar(200e3)
    assert check_member(frame, solve(frame), 1)["buckling"] is None


def test_a_slenderness_limit_can_fail_an_otherwise_safe_column():
    frame = _bar(-10.0, yield_stress=235e6)
    # 要一根真细长的杆，否则先被「欧拉不适用」判掉，就试不出限值本身
    frame.sections["S"] = Section("S", A=AREA, Iy=1e-6, Iz=1e-6, J=2e-6,
                                  cy=0.05, cz=0.05)
    loose = check_member(frame, solve(frame), 1)["buckling"]
    assert loose["euler_applicable"] is True
    assert loose["ok"] and loose["status"] == BUCKLING_OK
    tight = check_member(frame, solve(frame), 1,
                         slenderness_limit=loose["slenderness"] / 2)["buckling"]
    assert not tight["ok"] and tight["status"] == BUCKLING_SLENDER
    assert any("长细比" in w for w in tight["warnings"])


# ------------------------------------------------ 剖分：这一条是本组的重点

_SPLIT_COLUMN = {
    "units": "N-m-Pa",
    "materials": [{"name": "M", "E": E, "nu": 0.3, "allow_tension": 100e6}],
    "sections": [{"name": "S", "A": AREA, "Iy": IY, "Iz": IZ, "J": 1.6e-5,
                  "cy": 0.05, "cz": 0.05}],
    "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
              {"id": 2, "x": 0, "y": 0, "z": LENGTH}],
    "members": [{"id": 1, "i": 1, "j": 2, "section": "S", "material": "M",
                 "ref_vector": [1.0, 0.0, 0.0], "mu_y": 1.0, "mu_z": 1.0}],
    "supports": [{"node": 1, "fix": [1, 1, 1, 1, 1, 1]},
                 {"node": 2, "fix": [1, 1, 0, 1, 1, 1]}],
    "load_cases": [{"name": "P", "nodal_loads":
                    [{"node": 2, "load": [0, 0, -10e3, 0, 0, 0]}]}],
}


def _compiled(payload):
    c = compile_model(payload)
    return c, solve(c.analysis_model)


def test_splitting_a_column_must_not_change_its_critical_load():
    """同一根柱子，剖成两段后 Pcr 必须一模一样。

    按分段长度算就是 4 倍——**偏大、不报错、看着还合理**，
    这正是稳定校核最容易出的错。
    """
    whole, sol_w = _compiled(_SPLIT_COLUMN)
    a = check_strength(whole.analysis_model, sol_w,
                       mapping=whole.mapping)["members"][0]

    split = {k: v for k, v in _SPLIT_COLUMN.items()}
    split["nodes"] = [*_SPLIT_COLUMN["nodes"],
                      {"id": 3, "x": 0, "y": 0, "z": LENGTH / 2}]
    piece, sol_s = _compiled(split)
    assert len(piece.mapping.element_ids(1)) == 2, "这个算例本该被剖分"
    b = check_strength(piece.analysis_model, sol_s,
                       mapping=piece.mapping)["members"][0]

    assert b["length"] == pytest.approx(LENGTH, rel=1e-12)
    assert b["buckling"]["P_cr"] == pytest.approx(a["buckling"]["P_cr"], rel=1e-12)
    assert b["buckling"]["P_cr"] == pytest.approx(
        math.pi ** 2 * E * IY / LENGTH ** 2, rel=1e-12)


def test_dropping_the_mapping_says_so_instead_of_quietly_being_wrong():
    split = {k: v for k, v in _SPLIT_COLUMN.items()}
    split["nodes"] = [*_SPLIT_COLUMN["nodes"],
                      {"id": 3, "x": 0, "y": 0, "z": LENGTH / 2}]
    piece, sol = _compiled(split)
    got = check_strength(piece.analysis_model, sol)          # 故意不给 mapping
    assert any("偏不安全" in n for n in got["notes"])
    assert got["count"] == 2


def test_the_summary_names_the_worst_member_and_the_failures():
    whole, sol = _compiled(_SPLIT_COLUMN)
    got = check_strength(whole.analysis_model, sol, mapping=whole.mapping)
    assert got["worst_strength"]["member"] == 1
    assert got["worst_buckling"]["member"] == 1
    assert got["ok"] == (got["failed"] == [])
    # 判不了的杆件单独成一档，不混进 failed
    assert set(got["inconclusive"]) & set(got["failed"]) == set()


def test_allowable_stresses_follow_the_unit_system():
    """许用应力是应力。漏掉换算 Pa↔MPa 差 10⁶，只在换过单位的模型上现形。"""
    from units import MM, convert_model
    mm = convert_model(_SPLIT_COLUMN | {
        "materials": [{"name": "M", "E": E, "nu": 0.3,
                       "allow_tension": 100e6, "allow_compression": 90e6}]}, MM)
    assert mm["materials"][0]["allow_tension"] == pytest.approx(100.0)
    assert mm["materials"][0]["allow_compression"] == pytest.approx(90.0)

    # 判据是物理量不变：两套单位下的应力比必须逐位相同
    si, sol_si = _compiled(_SPLIT_COLUMN)
    r_si = check_strength(si.analysis_model, sol_si, mapping=si.mapping)
    other, sol_mm = _compiled(convert_model(_SPLIT_COLUMN, MM))
    r_mm = check_strength(other.analysis_model, sol_mm, mapping=other.mapping)
    assert (r_mm["members"][0]["strength"]["ratio"]
            == pytest.approx(r_si["members"][0]["strength"]["ratio"], rel=1e-12))
    assert (r_mm["members"][0]["buckling"]["ratio"]
            == pytest.approx(r_si["members"][0]["buckling"]["ratio"], rel=1e-12))


# --------------------------------------------- 折算应力（强度理论）

def _hn_beam(span: float, w: float):
    """工字梁简支受均布。改跨度就能在弯曲控制与剪切控制之间切换。"""
    import sections as sec
    from agent import Session

    hn = sec.i_section("HN400", 0.400, 0.200, 0.008, 0.013)
    s = Session()
    s.define_materials_and_sections(
        [{"name": "Q355", "E": 2.06e11, "nu": 0.3, "density": 7850.0,
          "yield_stress": 355e6, "allow_tension": 305e6}], [hn])
    s.set_model({**s.model,
                 "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                           {"id": 2, "x": span, "y": 0, "z": 0}],
                 "members": [{"id": 1, "i": 1, "j": 2,
                              "material": "Q355", "section": "HN400"}],
                 "supports": [{"node": 1, "fix": [1, 1, 1, 1, 0, 0]},
                              {"node": 2, "fix": [0, 1, 1, 1, 0, 0]}]})
    s.set_member_load(1, [0, 0, -w])
    s.solve_model()
    return s, hn


def test_a_slender_beam_is_governed_by_the_extreme_fibre():
    """细长梁上折算应力等于正应力：那里 τ=0，四个理论必然重合。

    这一条是**下界校验**——新加的折算应力不该凭空抬高一根本来就由弯曲
    控制的梁。若这里 σr4 > σ，说明 σ 与 τ 被凑到了截面上并不存在的同一点。
    """
    from strength import combined_stress_check

    span, w = 6.0, 30e3
    s, hn = _hn_beam(span, w)
    c = combined_stress_check(s.frame, s.solution, 1,
                              mapping=s.compilation.mapping, stations=101)
    sigma = w * span ** 2 / 8 * hn["cy"] / hn["Iz"]
    assert c["theories"]["4"]["point"] == "极端纤维"
    assert c["theories"]["4"]["sigma_r"] == pytest.approx(sigma, rel=1e-3)
    # τ=0 处四条理论同值
    values = [c["theories"][t]["sigma_r"] for t in "1234"]
    assert max(values) == pytest.approx(min(values), rel=1e-9)


def test_a_short_deep_beam_is_governed_by_shear_and_normal_stress_misses_it():
    """**这条是这个功能存在的理由。**

    L/h = 2.5 的短深梁：只看正应力是 19.6 MPa、应力比 0.064，看着安全得很；
    真实折算应力 54.6 MPa，差 2.78 倍。荷载再大一点就直接翻转结论。
    控制点在中性轴——那里弯曲 σ=0 而 τ 最大，正应力校核永远看不到。
    """
    from strength import combined_stress_check

    span, w = 1.0, 180e3
    s, hn = _hn_beam(span, w)
    c = combined_stress_check(s.frame, s.solution, 1,
                              mapping=s.compilation.mapping, stations=101)
    sigma = w * span ** 2 / 8 * hn["cy"] / hn["Iz"]
    assert c["theories"]["4"]["point"] == "中性轴"
    assert c["theories"]["4"]["sigma_r"] > 2.5 * sigma


def test_pure_shear_orders_the_four_theories_the_textbook_way():
    """纯剪下 σr1 < σr2 < σr4 < σr3，比值 1 : 1+ν : √3 : 2。

    顺序反了说明某一条套错了公式。这是四条理论差别最大的状态，
    第三理论比第四保守 15.5%——那 15.5% 正是选哪条理论的实际后果。
    """
    from strength import combined_stress_check

    s, _ = _hn_beam(1.0, 180e3)
    c = combined_stress_check(s.frame, s.solution, 1,
                              mapping=s.compilation.mapping, stations=101)
    r = {t: c["theories"][t]["sigma_r"] for t in "1234"}
    assert r["1"] < r["2"] < r["4"] < r["3"]
    tau = r["1"]                                   # 纯剪时 σ1 = τ
    assert r["2"] == pytest.approx(1.3 * tau, rel=1e-3)      # ν=0.3
    assert r["4"] == pytest.approx(math.sqrt(3.0) * tau, rel=1e-3)
    assert r["3"] == pytest.approx(2.0 * tau, rel=1e-3)


def test_the_tool_reports_both_conclusions_side_by_side():
    """正应力与折算应力是两条独立结论，工具层要同时给出。

    只留一条都不行：正应力那条按拉压分别比许用值（铸铁、木材必须），
    折算应力那条才看得见剪切控制。
    """
    s, _ = _hn_beam(1.0, 180e3)
    row = s.check_strength().payload["members"][0]
    assert row["stress_ratio"] < row["combined_ratio"], (
        "短深梁上折算应力必须比纯正应力更不利")
    assert row["combined_point"] == "中性轴"
    assert set(row["equivalent_stress_MPa"]) == {"σr1", "σr2", "σr3", "σr4"}
