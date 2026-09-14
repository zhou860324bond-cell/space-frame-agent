"""对标工具链的验证。

本机跑不了 Abaqus，所以能验的是两件事：写出的 .inp 内容对不对，以及
对比逻辑本身对不对（拿自己的结果冒充参考解，误差必须恒为零）。
真正的对标数值要在装了 Abaqus 的机器上跑出来。
"""
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "abaqus_bench"))

from compare import compare_one, normalized_error, our_results   # noqa: E402
from export_inp import write_inp                                 # noqa: E402
from frame3d import check_equilibrium, solve                     # noqa: E402
from inp_writer import resolve_element                           # noqa: E402
from model_io import from_dict, validate_payload                 # noqa: E402
from models import (all_models, cantilever_strong_axis,
                    refine_uniform_members, space_frame)  # noqa: E402


@pytest.mark.parametrize("model", all_models(), ids=lambda m: m["name"])
def test_every_benchmark_model_is_valid_and_balanced(model):
    payload = {k: v for k, v in model.items() if k not in ("name", "note")}
    assert validate_payload(payload) == []
    frame = from_dict(payload)
    assert check_equilibrium(frame, solve(frame))["ok"]


def test_inp_lists_every_node_and_element(tmp_path):
    model = cantilever_strong_axis()
    path = write_inp(model, "B33", tmp_path / "c.inp")
    text = path.read_text(encoding="ascii")
    assert "*ELEMENT, TYPE=B33" in text
    assert text.count("*ELEMENT, TYPE=B33") == len(model["members"])
    assert "*BEAM GENERAL SECTION" in text and "SECTION=GENERAL" in text
    for node in model["nodes"]:
        assert f"\n {node['id']}, " in text


def test_inp_stays_ascii(tmp_path):
    """Abaqus 6.14 是 Python 2，.inp 里出现非 ASCII 会引发编码问题。"""
    for model in all_models():
        path = write_inp(model, "B31", tmp_path / f"{model['name']}.inp")
        path.read_text(encoding="ascii")            # 非 ASCII 会在这里抛异常


def test_section_line_maps_iy_to_i11_and_iz_to_i22(tmp_path):
    """A, I11, I12, I22, J —— I11 对应本程序的 Iy，I22 对应 Iz。

    这个对应关系本机无法验证，所以基准里放了强弱轴悬殊的截面：一旦对调，
    Abaqus 那边的误差会从 1e-6 量级跳到百分级，一眼可辨。
    """
    model = cantilever_strong_axis()
    section = model["sections"][0]
    assert section["Iy"] != section["Iz"], "判别用的截面必须强弱轴悬殊"
    text = write_inp(model, "B33", tmp_path / "c.inp").read_text(encoding="ascii")
    line = next(l for l in text.splitlines()
                if l.startswith(" %g," % section["A"]) or l.startswith(" 0.012,"))
    cells = [float(c) for c in line.split(",")]
    assert cells[0] == pytest.approx(section["A"])
    assert cells[1] == pytest.approx(section["Iy"])
    assert cells[2] == 0.0
    assert cells[3] == pytest.approx(section["Iz"])
    assert cells[4] == pytest.approx(section["J"])


def test_vertical_members_get_the_fallback_reference_axis(tmp_path):
    """竖直柱的参考向量退化，n1 应当是全局 X —— 与内核的特判保持一致。"""
    text = write_inp(space_frame(), "B33", tmp_path / "s.inp").read_text(encoding="ascii")
    assert "\n 1, 0, 0\n" in text, "竖直杆件应当写出 n1 = (1, 0, 0)"
    assert "\n 0, 0, 1\n" in text, "水平杆件应当写出 n1 = (0, 0, 1)"


def test_b31_exports_native_shear_areas_as_abaqus_stiffness(tmp_path):
    """本程序 Ay/Az 必须映射成 Abaqus 数据顺序 K23/K13，且关闭补偿。"""
    model = deepcopy(cantilever_strong_axis())
    section = model["sections"][0]
    section["Ay"] = 0.009
    section["Az"] = 0.007
    material = model["materials"][0]
    shear_modulus = material["E"] / (2.0 * (1.0 + material["nu"]))

    lines = write_inp(model, "B31", tmp_path / "matched.inp").read_text(
        encoding="ascii").splitlines()
    keyword = lines.index("*TRANSVERSE SHEAR STIFFNESS")
    k23, k13, compensation = (float(value) for value in lines[keyword + 1].split(","))

    assert k23 == pytest.approx(shear_modulus * section["Az"])
    assert k13 == pytest.approx(shear_modulus * section["Ay"])
    assert compensation == 0.0


def test_b33_does_not_export_transverse_shear_stiffness(tmp_path):
    model = deepcopy(cantilever_strong_axis())
    model["sections"][0].update(Ay=0.009, Az=0.007)
    text = write_inp(model, "B33", tmp_path / "euler.inp").read_text(encoding="ascii")
    assert "*TRANSVERSE SHEAR STIFFNESS" not in text


def test_b31_rejects_a_half_defined_shear_area(tmp_path):
    model = deepcopy(cantilever_strong_axis())
    model["sections"][0]["Ay"] = 0.009
    model["sections"][0].pop("Az", None)
    with pytest.raises(ValueError, match="Ay.*Az"):
        write_inp(model, "B31", tmp_path / "ambiguous.inp")


def test_auto_element_follows_the_native_section_theory():
    euler = deepcopy(cantilever_strong_axis())
    timoshenko = deepcopy(euler)
    timoshenko["sections"][0].update(Ay=0.01, Az=0.01)
    assert resolve_element(euler, "auto") == "B33"
    assert resolve_element(timoshenko, "auto") == "B31"


def test_auto_element_rejects_mixed_section_theories():
    model = deepcopy(space_frame())
    model["sections"][0].update(Ay=0.01, Az=0.01)
    with pytest.raises(ValueError, match="混用了"):
        resolve_element(model, "auto")


def test_normalized_error_is_zero_for_identical_fields():
    assert normalized_error([1.0, -2.0, 3.0], [1.0, -2.0, 3.0]) == pytest.approx(0.0)


def test_normalized_error_matches_the_closed_form():
    """整体放大 1% 时，归一化误差应为 0.01/1.01。"""
    base = [1.0, 2.0, -3.0, 0.5]
    scaled = [v * 1.01 for v in base]
    assert normalized_error(scaled, base) == pytest.approx(0.01, rel=1e-12)
    assert normalized_error(base, scaled) == pytest.approx(0.01 / 1.01, rel=1e-12)


def test_normalized_error_is_undefined_against_an_all_zero_reference():
    assert normalized_error([0.0, 1e-20], [0.0, 0.0]) is None


@pytest.mark.parametrize("model", all_models(), ids=lambda m: m["name"])
def test_comparing_our_results_against_themselves_gives_zero(model):
    r = compare_one(model, our_results(model))
    worst = max((v for v in r["errors"].values() if v is not None), default=0.0)
    assert worst < 1e-12
    assert r["nodes_compared"] == len(model["nodes"])
    refined = refine_uniform_members(model, 2)
    original = our_results(model)
    subdivided = our_results(refined)
    for node in original:
        for component in ("u1", "u2", "u3", "ur1", "ur2", "ur3"):
            assert subdivided[node][component] == pytest.approx(
                original[node][component], abs=1e-12)
