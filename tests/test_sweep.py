"""参数扫描的验证。

扫描是"一次调用规划 N 次求解"，最要紧的两条：

1. **扫描不能改动当前模型。** 用户扫完还要接着用原模型算别的。
   只还原 model 而不还原装配好的 frame / solution 的话，之后的查询会拿
   最后一个探针的结果去回答原模型的问题——不报错，数全是错的。
2. **限值判断的方向不能写反。** 位移、弯矩是越小越好；屈曲因子、频率是越大越好。
   写反了会给出"截面越小越安全"这种荒唐结论，而每一行数字看着都对。
"""

import pytest

from agent import Session

E, RHO = 2.1e11, 7850.0
MATERIALS = [{"name": "STEEL", "E": E, "nu": 0.3, "density": RHO}]
SECTIONS = [{"name": "COLUMN", "A": 0.012, "Iy": 8e-5, "Iz": 2.4e-4, "J": 1e-6},
            {"name": "BEAM", "A": 0.010, "Iy": 4e-5, "Iz": 3e-4, "J": 8e-7}]
SPAN, W = 8.0, -25e3


def simple_beam() -> Session:
    """单跨简支梁。比例关系（挠度 ∝ 1/I）只在静定结构上严格成立——
    刚架里改梁的 I 会改变弯矩分配，比例就不准了。验比例得用这个。"""
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    s.set_model(model={
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                  {"id": 2, "x": SPAN, "y": 0, "z": 0}],
        "members": [{"id": 1, "i": 1, "j": 2,
                     "section": "BEAM", "material": "STEEL"}],
        "supports": [{"node": 1, "fix": [1, 1, 1, 1, 0, 0]},
                     {"node": 2, "fix": [0, 1, 1, 1, 0, 0]}]})
    s.set_load_cases(cases=[{"name": "DL",
                             "member_loads": [{"member": 1, "w": [0, 0, W]}]}])
    s.solve_model()
    return s


def solved() -> Session:
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    g = s.generate_frame(spans=[SPAN], storeys=[3.6], column_section="COLUMN",
                         beam_section="BEAM", material="STEEL")
    s.set_load_cases(cases=[{"name": "DL", "member_loads":
                             [{"member": m, "w": [0, 0, W]}
                              for m in g.payload["beam_member_ids"]]}])
    s.solve_model()
    return s


# ------------------------------------------------- 不改动模型

def test_a_sweep_leaves_the_model_untouched():
    s = solved()
    before = {sec["name"]: dict(sec) for sec in s.model["sections"]}
    s.sweep(what="section_property", target="BEAM", prop="Iz",
            values=[1e-4, 5e-4, 1e-3], metric="max_deflection", case="DL")
    after = {sec["name"]: dict(sec) for sec in s.model["sections"]}
    assert after == before


def test_results_after_a_sweep_still_describe_the_original_model():
    """这条比上一条更硬：模型对了不等于结果对了。

    只还原 model 而不重新求解的话，接下来的查询会拿最后一个探针的
    结果去回答原模型的问题——不报错，但数全是错的。
    """
    s = solved()
    before = s.query_results(what="max_deflection", case="DL").payload["magnitude_mm"]
    s.sweep(what="section_property", target="BEAM", prop="Iz",
            values=[1e-4, 5e-4, 1e-3], metric="max_deflection", case="DL")
    after = s.query_results(what="max_deflection", case="DL").payload["magnitude_mm"]
    assert after == pytest.approx(before, rel=1e-12)


def test_a_sweep_on_an_unsolved_session_still_restores_cleanly():
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    s.generate_frame(spans=[SPAN], storeys=[3.6], column_section="COLUMN",
                     beam_section="BEAM", material="STEEL")
    s.set_load_cases(cases=[{"name": "DL", "member_loads":
                             [{"member": 3, "w": [0, 0, W]}]}])
    assert s.solution is None
    s.sweep(what="section_property", target="BEAM", prop="Iz",
            values=[1e-4, 1e-3], metric="max_deflection", case="DL")
    assert s.solution is None, "扫描前没算过，扫描后也不该凭空多出结果"


# ------------------------------------------------- 数值正确

def test_deflection_is_inversely_proportional_to_inertia_on_a_simple_beam():
    """挠度 ∝ 1/I。这条同时验了每个探针确实用上了那个取值。

    必须用**静定**结构。刚架里改梁的 I 会改变梁柱之间的弯矩分配，
    比例关系就不成立了——第一版拿刚架来验这条，红了才想起来。
    """
    s = simple_beam()
    r = s.sweep(what="section_property", target="BEAM", prop="Iz",
                values=[3e-4, 6e-4, 1.2e-3], metric="max_deflection", case="DL")
    assert r.ok, r.payload
    got = [row["metric"] for row in r.payload["rows"]]
    assert got[1] == pytest.approx(got[0] / 2, rel=1e-4)
    assert got[2] == pytest.approx(got[0] / 4, rel=1e-4)


def test_deflection_still_decreases_monotonically_on_a_frame():
    """刚架里比例关系不成立，但单调性还在——这才是刚架上该验的。"""
    s = solved()
    r = s.sweep(what="section_property", target="BEAM", prop="Iz",
                values=[1e-4, 3e-4, 1e-3, 3e-3], metric="max_deflection",
                case="DL")
    got = [row["metric"] for row in r.payload["rows"]]
    assert all(a > b for a, b in zip(got, got[1:], strict=False)), got


def test_each_probe_matches_solving_that_model_by_hand():
    """扫描出来的每一行，都必须等于手工把那个取值写进模型再算一遍。"""
    s = solved()
    values = [2e-4, 7e-4]
    r = s.sweep(what="section_property", target="BEAM", prop="Iz",
                values=values, metric="max_deflection", case="DL")
    for value, row in zip(values, r.payload["rows"]):
        manual = solved()
        for sec in manual.model["sections"]:
            if sec["name"] == "BEAM":
                sec["Iz"] = value
        manual._invalidate()
        manual.solve_model()
        expect = manual.query_results(what="max_deflection",
                                      case="DL").payload["magnitude_mm"]
        assert row["metric"] == pytest.approx(expect, rel=1e-12)


def test_a_load_scale_sweep_is_linear():
    """线弹性下位移与荷载成正比。"""
    s = solved()
    r = s.sweep(what="load_scale", target="DL", values=[1.0, 2.0, 3.0],
                metric="max_deflection")
    got = [row["metric"] for row in r.payload["rows"]]
    assert got[1] == pytest.approx(2 * got[0], rel=1e-6)
    assert got[2] == pytest.approx(3 * got[0], rel=1e-6)


def test_a_modulus_sweep_is_inverse():
    s = solved()
    r = s.sweep(what="material_E", target="STEEL", values=[E, 2 * E],
                metric="max_deflection", case="DL")
    got = [row["metric"] for row in r.payload["rows"]]
    assert got[1] == pytest.approx(got[0] / 2, rel=1e-6)


# ------------------------------------------------- 限值方向

def test_smaller_is_better_for_deflection():
    """位移越小越好：满足限值的是**大**截面，最省取值是其中最小的那个。"""
    s = solved()
    limit = SPAN / 400 * 1000                 # L/400，单位 mm
    r = s.sweep(what="section_property", target="BEAM", prop="Iz",
                values=[1e-4, 3e-4, 1e-3, 3e-3], metric="max_deflection",
                case="DL", limit=limit)
    rows = {row["value"]: row["metric"] for row in r.payload["rows"]}
    assert all(rows[v] <= limit for v in r.payload["satisfying_values"])
    assert r.payload["smallest_sufficient"] == min(r.payload["satisfying_values"])
    # 不满足的那些确实超限
    for v in rows:
        if v not in r.payload["satisfying_values"]:
            assert rows[v] > limit


def test_bigger_is_better_for_the_buckling_factor():
    """屈曲因子越大越好——方向和位移相反。写反了会得出"截面越小越安全"。"""
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    s.set_model(model={
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                  {"id": 2, "x": 3.0, "y": 0, "z": 0},
                  {"id": 3, "x": 6.0, "y": 0, "z": 0}],
        "members": [{"id": 1, "i": 1, "j": 2, "section": "COLUMN", "material": "STEEL"},
                    {"id": 2, "i": 2, "j": 3, "section": "COLUMN", "material": "STEEL"}],
        "supports": [{"node": 1, "fix": [1, 1, 1, 1, 0, 1]},
                     {"node": 2, "fix": [0, 1, 0, 1, 0, 1]},
                     {"node": 3, "fix": [0, 1, 1, 1, 0, 1]}]})
    s.set_load_cases(cases=[{"name": "P", "nodal_loads":
                             [{"node": 3, "load": [-1e3, 0, 0, 0, 0, 0]}]}])
    s.solve_model()
    r = s.sweep(what="section_property", target="COLUMN", prop="Iz",
                values=[1e-5, 1e-4, 1e-3], metric="buckling_factor",
                case="P", limit=1000.0)
    assert r.ok, r.payload
    got = [row["metric"] for row in r.payload["rows"]]
    assert got[0] < got[1] < got[2], "截面越大越不容易失稳"
    assert all(v >= 1000.0
               for value, v in zip(r.payload["satisfying_values"], got[-len(
                   r.payload["satisfying_values"]):]))
    assert r.payload["smallest_sufficient"] == min(r.payload["satisfying_values"])


def test_an_unreachable_limit_says_so():
    s = solved()
    r = s.sweep(what="section_property", target="BEAM", prop="Iz",
                values=[1e-6, 2e-6], metric="max_deflection",
                case="DL", limit=0.001)
    assert r.payload["satisfying_values"] == []
    assert r.payload["smallest_sufficient"] is None
    assert "hint" in r.payload


# ------------------------------------------------- 失败路径

def test_an_unknown_target_lists_what_exists():
    s = solved()
    r = s.sweep(what="section_property", target="NOPE", prop="Iz",
                values=[1e-4, 1e-3], metric="max_deflection")
    assert not r.ok
    assert "COLUMN" in r.payload["available"]


def test_a_section_sweep_needs_a_property():
    s = solved()
    assert not s.sweep(what="section_property", target="BEAM",
                       values=[1e-4, 1e-3], metric="max_deflection").ok


def test_one_value_is_not_a_sweep():
    s = solved()
    assert not s.sweep(what="section_property", target="BEAM", prop="Iz",
                       values=[1e-4], metric="max_deflection").ok


def test_a_sweep_without_a_model_is_refused():
    assert not Session().sweep(what="material_E", target="STEEL",
                               values=[1, 2], metric="max_deflection").ok


def test_a_probe_that_fails_is_recorded_not_fatal():
    """某个取值算不出来时，那一行标记失败，其余照常返回。"""
    s = solved()
    r = s.sweep(what="section_property", target="BEAM", prop="Iz",
                values=[3e-4, 6e-4], metric="first_frequency", case="DL")
    assert r.ok
    assert all(row["metric"] is not None for row in r.payload["rows"])
