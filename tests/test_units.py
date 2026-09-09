"""单位制。

判据只有一条，但它足够硬：**同一个结构，换算到另一套单位制再算，
报出来的 mm 和 kN 必须逐位相同。**

比逐个核对换算系数可靠得多——漏掉任何一类量（截面惯性矩、弯矩、线荷载、
集中力位置……）都会立刻露馅，而逐个核对很容易把漏掉的那一项也一起漏看。
"""

import pytest

from agent import Session
from model_io import validate_payload
from units import MM, SI, convert_model, system

E_SI, L, P = 2.1e11, 6.0, 30e3
MATERIALS = [{"name": "STEEL", "E": E_SI, "nu": 0.3, "density": 7850.0}]
SECTIONS = [{"name": "BEAM", "A": 0.01, "Iy": 4e-5, "Iz": 3e-4, "J": 8e-7}]


def rich_model() -> dict:
    """把每一类需要换算的量都用上，少一类这套测试就漏一类。"""
    return {
        "units": SI,
        "materials": [dict(m) for m in MATERIALS],
        "sections": [dict(s) for s in SECTIONS],
        "nodes": [{"id": 1, "x": 0.0, "y": 0.0, "z": 0.0},
                  {"id": 2, "x": L, "y": 0.0, "z": 0.0},
                  {"id": 3, "x": 2 * L, "y": 0.0, "z": 0.0}],
        "members": [{"id": 1, "i": 1, "j": 2, "section": "BEAM", "material": "STEEL"},
                    {"id": 2, "i": 2, "j": 3, "section": "BEAM", "material": "STEEL"}],
        "supports": [{"node": 1, "fix": [1] * 6}, {"node": 3, "fix": [1] * 6}],
        "load_cases": [{
            "name": "ALL",
            "nodal_loads": [{"node": 2, "load": [0, 0, -P, 0, 0, 5e3]}],
            "member_loads": [{"member": 1, "w": [0, 0, -8e3]}],
            "member_spans": [
                {"member": 2, "kind": "point", "w1": [0, 0, -P], "a": 2.0},
                {"member": 2, "kind": "trapezoid", "w1": [0, 0, 0], "w2": [0, 0, -4e3]},
            ],
            "settlements": [{"node": 3, "d": [0, 0, -0.005, 0, 0, 0]}],
        }],
    }


def solved(model: dict) -> Session:
    s = Session()
    assert s.set_model(model=model).ok, validate_payload(model)
    assert s.solve_model().ok
    return s


# ------------------------------------------------- 常数

def test_gravity_differs_between_the_two_systems():
    assert system(MM).gravity == pytest.approx(1000.0 * system(SI).gravity)


def test_an_unknown_system_is_refused():
    with pytest.raises(ValueError, match="未知单位制"):
        system("kg-cm-s")


def test_reported_units_are_the_same_in_both_systems():
    """报告一律用 mm / kN / kN·m，读的人不用先想模型里存的是 m 还是 mm。"""
    for name in (SI, MM):
        u = system(name)
        assert (u.disp_unit, u.force_unit, u.moment_unit) == ("mm", "kN", "kN·m")


# ------------------------------------------------- 换算不变性

@pytest.fixture(scope="module")
def both():
    si = rich_model()
    mm = convert_model(si, MM)
    assert not validate_payload(mm), validate_payload(mm)
    return solved(si), solved(mm)


def test_conversion_leaves_the_source_model_untouched():
    si = rich_model()
    before = si["sections"][0]["Iz"]
    convert_model(si, MM)
    assert si["sections"][0]["Iz"] == before
    assert si["units"] == SI


def test_the_converted_model_declares_the_new_system():
    assert convert_model(rich_model(), MM)["units"] == MM


def test_displacement_is_the_same_in_both_systems(both):
    a, b = both
    da = a.query_results(what="max_displacement", case="ALL").payload
    db = b.query_results(what="max_displacement", case="ALL").payload
    assert da["node"] == db["node"]
    assert db["magnitude_mm"] == pytest.approx(da["magnitude_mm"], rel=1e-9)
    for k in ("ux", "uy", "uz"):
        assert db["components_mm"][k] == pytest.approx(da["components_mm"][k],
                                                       rel=1e-9, abs=1e-9)


def test_reactions_are_the_same_in_both_systems(both):
    a, b = both
    ra = a.query_results(what="reactions", case="ALL").payload
    rb = b.query_results(what="reactions", case="ALL").payload
    assert rb["vertical_total_kN"] == pytest.approx(ra["vertical_total_kN"], rel=1e-9)
    for nid, va in ra["reactions"].items():
        for k in range(3):
            assert rb["reactions"][nid]["R"][k] == pytest.approx(
                va["R"][k], rel=1e-9, abs=1e-9)


def test_member_forces_are_the_same_in_both_systems(both):
    """轴力用 force_scale、弯矩用 moment_scale，两者在 mm 制下相差 1000 倍。
    只验位移的话，把弯矩系数写错完全发现不了。"""
    a, b = both
    fa = a.query_results(what="member_forces", case="ALL").payload["all_members"]
    fb = b.query_results(what="member_forces", case="ALL").payload["all_members"]
    assert set(fa) == set(fb)
    for mid, row in fa.items():
        for key, value in row.items():
            assert fb[mid][key] == pytest.approx(value, rel=1e-9, abs=1e-9), (mid, key)


@pytest.mark.parametrize("component", ["N", "Vy", "Mz", "My"])
def test_internal_force_diagrams_are_the_same_in_both_systems(both, component):
    a, b = both
    da = a.query_diagram(component=component, member=2, stations=9).payload
    db = b.query_diagram(component=component, member=2, stations=9).payload
    assert db["peak"] == pytest.approx(da["peak"], rel=1e-9, abs=1e-9)
    for x, y in zip(da["values"], db["values"]):
        assert y == pytest.approx(x, rel=1e-9, abs=1e-9)


def test_the_point_load_position_is_converted(both):
    """a 是长度。忘了换算的话，mm 制下 2.0 会被当成 2 mm 而不是 2 m ——
    集中力几乎跑到杆端，弯矩图完全变样。"""
    a, b = both
    da = a.query_diagram(component="Mz", member=2, stations=41).payload
    db = b.query_diagram(component="Mz", member=2, stations=41).payload
    assert db["at_x_m"] == pytest.approx(da["at_x_m"], rel=1e-6)


def test_all_reported_length_fields_are_metres_in_both_systems(both):
    a, b = both
    for s in (a, b):
        s.solve_model()
    qa = a.query_results(what="max_deflection", case="ALL", span=6.0).payload
    qb = b.query_results(what="max_deflection", case="ALL", span=6.0).payload
    assert qb["at_x_m"] == pytest.approx(qa["at_x_m"], rel=1e-6)
    assert qb["member_length_m"] == pytest.approx(qa["member_length_m"], rel=1e-6)
    assert qb["span_over_deflection"] == pytest.approx(qa["span_over_deflection"], rel=1e-6)

    da = a.query_diagram(component="Mz", member=2, stations=9).payload
    db = b.query_diagram(component="Mz", member=2, stations=9).payload
    assert db["length_m"] == pytest.approx(da["length_m"], rel=1e-6)
    assert db["x_m"] == pytest.approx(da["x_m"], rel=1e-6)

    na = a.query_results(what="max_displacement", case="ALL").payload
    nb = b.query_results(what="max_displacement", case="ALL").payload
    assert nb["coordinates_xyz"] == pytest.approx(na["coordinates_xyz"], rel=1e-6)


def test_self_weight_matches_across_systems():
    """密度和 g 各换一次、方向相反，乘起来必须抵消。"""
    def total(model):
        s = Session()
        s.set_model(model=model)
        r = s.add_self_weight(case="ALL")
        assert r.ok, r.payload
        s.solve_model()
        return s.query_results(what="reactions",
                               case="ALL").payload["vertical_total_kN"]

    si = rich_model()
    assert total(convert_model(si, MM)) == pytest.approx(total(si), rel=1e-9)


def test_a_round_trip_returns_the_original_numbers():
    si = rich_model()
    back = convert_model(convert_model(si, MM), SI)
    assert back["sections"][0]["Iz"] == pytest.approx(si["sections"][0]["Iz"], rel=1e-12)
    assert back["materials"][0]["E"] == pytest.approx(si["materials"][0]["E"], rel=1e-12)
    assert back["nodes"][1]["x"] == pytest.approx(si["nodes"][1]["x"], rel=1e-12)
    case, ref = back["load_cases"][0], si["load_cases"][0]
    assert case["member_loads"][0]["w"][2] == pytest.approx(
        ref["member_loads"][0]["w"][2], rel=1e-12)
    assert case["settlements"][0]["d"][2] == pytest.approx(
        ref["settlements"][0]["d"][2], rel=1e-12)


def test_parametric_generation_preserves_an_existing_mm_unit_system():
    """参数仍按公开接口的 m / N/m 输入，但落地数值必须服从当前模型单位。"""
    s = Session(model=rich_model())
    assert s.set_units(MM).ok
    result = s.generate_frame(
        spans=[6.0], storeys=[3.6], column_section="BEAM",
        beam_section="BEAM", material="STEEL", beam_load=8e3)
    assert result.ok, result.payload
    assert s.model["units"] == MM
    assert max(n["x"] for n in s.model["nodes"]) == pytest.approx(6000.0)
    assert s.model["materials"][0]["E"] == pytest.approx(2.1e5)
    assert s.model["sections"][0]["A"] == pytest.approx(1.0e4)
    assert s.model["member_loads"][0]["w"][2] == pytest.approx(-8.0)


def test_complex_bent_operators_convert_metres_in_an_mm_project():
    """折线刚架、拉伸、范围选择和抬高必须遵守同一套公开长度单位。"""
    s = Session(model=rich_model())
    assert s.set_units(MM).ok
    made = s.generate_bent(
        profile=[[0.0, 7.2], [12.0, 7.2]], columns=[0.0, 12.0],
        column_section="BEAM", beam_section="BEAM", material="STEEL")
    assert made.ok, made.payload
    assert s.model["units"] == MM
    assert max(n["x"] for n in s.model["nodes"]) == pytest.approx(12000.0)
    assert max(n["z"] for n in s.model["nodes"]) == pytest.approx(7200.0)

    extruded = s.extrude_bents(bays=[6.0], tie_section="BEAM")
    assert extruded.ok, extruded.payload
    assert max(n["y"] for n in s.model["nodes"]) == pytest.approx(6000.0)

    selected = s.select_members(orientation="vertical", x_range=[11.9, 12.1])
    assert selected.ok and selected.payload["member_ids"]
    before = {n["id"]: n["z"] for n in s.model["nodes"]}
    raised = s.raise_nodes(dz=1.0, x_range=[11.9, 12.1])
    assert raised.ok, raised.payload
    assert raised.payload["dz"] == pytest.approx(1.0)
    for nid in raised.payload["moved_nodes"]:
        assert s.model["nodes"][nid - 1]["z"] == pytest.approx(before[nid] + 1000.0)


def test_converting_to_the_same_system_changes_nothing():
    si = rich_model()
    assert convert_model(si, SI) == si


def test_rotational_settlement_is_not_scaled():
    """转角是弧度，无量纲。跟着长度一起换算是个很容易犯的错。"""
    si = rich_model()
    si["load_cases"][0]["settlements"] = [
        {"node": 3, "d": [0, 0, 0, 0, 0, 1e-3]}]
    mm = convert_model(si, MM)
    assert mm["load_cases"][0]["settlements"][0]["d"][5] == pytest.approx(1e-3)


# ------------------------------------------------- Agent 工具

def test_the_agent_can_switch_units_without_changing_the_answer():
    """set_units 换的是数值，结果必须一个字不变。"""
    a = solved(rich_model())
    before = a.query_results(what="max_displacement", case="ALL").payload
    forces = a.query_results(what="member_forces", case="ALL").payload["all_members"]

    r = a.set_units(units=MM)
    assert r.ok, r.payload
    assert r.payload["was"] == SI and r.payload["units"] == MM
    assert a.solve_model().ok
    after = a.query_results(what="max_displacement", case="ALL").payload
    assert after["magnitude_mm"] == pytest.approx(before["magnitude_mm"], rel=1e-9)
    now = a.query_results(what="member_forces", case="ALL").payload["all_members"]
    for mid, row in forces.items():
        for key, value in row.items():
            assert now[mid][key] == pytest.approx(value, rel=1e-9, abs=1e-9)


def test_set_units_refuses_an_unknown_system():
    a = solved(rich_model())
    r = a.set_units(units="kg-cm-s")
    assert not r.ok
    assert a.model["units"] == SI, "被拒绝时不能动原模型"


def test_unit_system_can_be_chosen_before_modeling():
    s = Session()
    result = s.set_units(units=MM)
    assert result.ok
    assert s.model["units"] == MM


def test_defining_properties_does_not_reset_the_project_units():
    s = Session()
    assert s.set_units(units=MM).ok
    s.define_materials_and_sections(
        [{"name": "STEEL", "E": 2.1e5, "nu": 0.3}],
        [{"name": "BEAM", "A": 1e4, "Iy": 4e7, "Iz": 3e8, "J": 8e5}],
    )
    assert s.model["units"] == MM
