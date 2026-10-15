"""节点集中质量与荷载换算质量（重力荷载代表值）。

防的问题：模型里质量只来自杆件密度 ρA，楼板、面层、设备、隔墙都只以荷载出现。
模态与反应谱算的是「空骨架」的振动：周期偏短，地震作用整体失真，而且不报错。
真实楼房做地震分析，第一步就是把重力荷载代表值换成质量。

闭合解：无质量悬臂梁端挂质量 M，f = √(3EI/(M·L³))/(2π)——杆件没有密度时质量阵
只是半正定（转动与内部节点无质量），这条同时防「只有节点质量时特征值求解拒算」。
"""

from __future__ import annotations

from copy import deepcopy
import math

import pytest

from agent import Session
from changes import prune_loads
from model_io import validate_payload
from units import convert_model

E, NU = 2.0e11, 0.3
A, IY, IZ, J = 0.01, 1.0e-5, 2.0e-5, 5.0e-6


def _cantilever(length: float = 3.0, density: float = 0.0, tip_mass: float = 500.0) -> dict:
    return {
        "units": "N-m-Pa",
        "materials": [{"name": "Steel", "E": E, "nu": NU, "density": density}],
        "sections": [{"name": "S", "A": A, "Iy": IY, "Iz": IZ, "J": J}],
        "nodes": [{"id": 1, "x": 0.0, "y": 0.0, "z": 0.0},
                  {"id": 2, "x": length, "y": 0.0, "z": 0.0}],
        "members": [{"id": 1, "i": 1, "j": 2, "section": "S", "material": "Steel"}],
        "supports": [{"node": 1, "fix": [1] * 6}],
        "masses": [{"name": "配重", "node": 2, "mass": tip_mass}],
    }


def _simple_beam(length: float = 6.0) -> dict:
    return {
        "units": "N-m-Pa",
        "materials": [{"name": "Steel", "E": E, "nu": NU, "density": 7850.0}],
        "sections": [{"name": "S", "A": A, "Iy": IY, "Iz": IZ, "J": J}],
        "nodes": [{"id": 1, "x": 0.0, "y": 0.0, "z": 0.0},
                  {"id": 2, "x": length, "y": 0.0, "z": 0.0}],
        "members": [{"id": 1, "i": 1, "j": 2, "section": "S", "material": "Steel"}],
        "supports": [{"node": 1, "fix": [1, 1, 1, 1, 0, 0]},
                     {"node": 2, "fix": [0, 1, 1, 0, 0, 0]}],
        "load_cases": [
            {"name": "DL", "member_spans": [
                {"member": 1, "kind": "uniform", "w1": [0, 0, -1.0e4]}]},
            {"name": "LL", "member_spans": [
                {"member": 1, "kind": "point", "w1": [0, 0, -2.0e4], "a": 2.0}],
             "nodal_loads": [{"node": 2, "load": [0, 0, 3.0e3, 0, 0, 0]}]},
        ],
    }


# --------------------------------------------------------------- 节点质量

def test_massless_cantilever_with_a_tip_mass_matches_the_closed_form():
    length, mass = 3.0, 500.0
    got = Session(model=_cantilever(length, tip_mass=mass)).modal_analysis(num_modes=2)
    assert got.ok, got.payload
    exact = math.sqrt(3 * E * min(IY, IZ) / (mass * length ** 3)) / (2 * math.pi)
    assert got.payload["modes"][0]["frequency_Hz"] == pytest.approx(exact, rel=1e-6)
    assert got.payload["total_mass_kg"] == pytest.approx(mass)


def test_a_tip_mass_lowers_the_frequency_of_a_heavy_beam_and_counts_in_total_mass():
    bare = _cantilever(density=7850.0)
    bare.pop("masses")
    loaded = _cantilever(density=7850.0, tip_mass=800.0)
    f_bare = Session(model=bare).modal_analysis(num_modes=2).payload
    f_loaded = Session(model=loaded).modal_analysis(num_modes=2).payload
    assert f_loaded["modes"][0]["frequency_Hz"] < f_bare["modes"][0]["frequency_Hz"]
    assert f_loaded["total_mass_kg"] == pytest.approx(f_bare["total_mass_kg"] + 800.0)


def test_nodal_mass_survives_the_unit_conversion():
    """kg ↔ t、kg·m² ↔ t·mm²。换错一个系数，频率差 √1000 倍。"""
    si = _cantilever()
    si["masses"][0]["inertia"] = [5.0, 6.0, 7.0]
    mm = convert_model(deepcopy(si), "N-mm-MPa")
    a = Session(model=si).modal_analysis(num_modes=3).payload
    b = Session(model=mm).modal_analysis(num_modes=3).payload
    for ma, mb in zip(a["modes"], b["modes"], strict=True):
        assert mb["frequency_Hz"] == pytest.approx(ma["frequency_Hz"], rel=1e-9)
    assert b["total_mass_kg"] == pytest.approx(a["total_mass_kg"], rel=1e-12)


def test_constant_spectrum_base_shear_equals_tip_mass_times_acceleration():
    """常量谱下各阶基底剪力之和 = 可参与质量 × Sa；这里可参与的只有端部质量。"""
    mass, sa = 500.0, 2.0
    got = Session(model=_cantilever(tip_mass=mass)).response_spectrum_analysis(
        direction="y", num_modes=3, spectrum_points=[[0.0, sa], [10.0, sa]])
    assert got.ok, got.payload
    assert got.payload["base_shear_kN"] == pytest.approx(mass * sa / 1e3, rel=1e-6)


def test_set_nodal_mass_replaces_by_name_and_zero_removes():
    session = Session(model=_cantilever())
    assert session.set_nodal_mass([2], 300.0, name="设备").ok
    assert session.set_nodal_mass([2], 400.0, name="设备").ok
    names = sorted((e["name"], e["mass"]) for e in session.model["masses"])
    assert names == [("设备", 400.0), ("配重", 500.0)]
    assert session.set_nodal_mass([2], 0.0, name="设备").ok
    assert [e["name"] for e in session.model["masses"]] == ["配重"]
    assert not session.set_nodal_mass([9], 1.0).ok
    assert not session.set_nodal_mass([2], -1.0).ok


# ------------------------------------------------------- 重力荷载代表值换算

def test_masses_from_loads_follows_the_lever_rule_and_the_combination_factors():
    session = Session(model=_simple_beam())
    got = session.masses_from_loads({"DL": 1.0, "LL": 0.5})
    assert got.ok, got.payload
    g = 9.80665
    q, length, p, a = 1.0e4, 6.0, 2.0e4, 2.0
    expected = {1: (q * length / 2 + 0.5 * p * (length - a) / length) / g,
                2: (q * length / 2 + 0.5 * p * a / length) / g}
    got_masses = {e["node"]: e["mass"] for e in session.model["masses"]}
    assert got_masses == pytest.approx(expected, rel=1e-12)
    assert got.payload["ignored_upward"] == 1, "LL 里向上的节点力不是重量"

    # 再调一次是替换，不是叠加
    assert session.masses_from_loads({"DL": 1.0, "LL": 0.5}).ok
    assert {e["node"]: e["mass"] for e in session.model["masses"]} == pytest.approx(expected)


def test_self_weight_is_not_counted_twice():
    """带密度的杆件，一致质量阵里已经有 ρA；自重荷载再换一次就算了两遍。"""
    session = Session(model=_simple_beam())
    assert session.add_self_weight(case="DL").ok
    got = session.masses_from_loads({"DL": 1.0})
    assert got.ok
    assert got.payload["skipped_self_weight"] >= 1
    q, length = 1.0e4, 6.0
    assert got.payload["total_mass_kg"] == pytest.approx(q * length / 9.80665, rel=1e-9)


def test_unknown_case_is_rejected_by_name():
    got = Session(model=_simple_beam()).masses_from_loads({"活载": 0.5})
    assert not got.ok and "活载" in got.payload["error"]


def test_masses_follow_their_nodes_when_deleted():
    model = _cantilever()
    model["masses"].append({"name": "x", "node": 1, "mass": 1.0})
    counted = prune_loads(model, dropped_nodes={2})
    assert counted["masses"] == 1
    assert [e["node"] for e in model["masses"]] == [1]


def test_schema_rejects_masses_on_missing_nodes_and_negative_values():
    bad_node = _cantilever()
    bad_node["masses"][0]["node"] = 9
    assert any("masses" in e and "[9]" in e for e in validate_payload(bad_node))
    negative = _cantilever()
    negative["masses"][0]["mass"] = -1.0
    assert validate_payload(negative)


def test_kernel_and_session_agree_on_the_self_weight_marker():
    from masses import SELF_WEIGHT_NOTE as kernel
    from session_base import SELF_WEIGHT_NOTE as session
    assert kernel == session
