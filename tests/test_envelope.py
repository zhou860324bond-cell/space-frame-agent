"""内力包络与控制组合的验证。

最容易犯的错是"先各工况取极值再比"——那样得到的数偏大、位置错乱，
而且**在单工况下也能给出看起来对的结果**，所以不专门验就发现不了。

这里的核心判据是一条构造出来的情形：**同一根杆上，跨中由一个组合控制、
支座由另一个控制**。逐点包络能分辨，先取极值再比的做法分辨不了。
"""

import numpy as np
import pytest

from envelope import (all_envelopes, envelope_cases, governing_summary,
                      member_envelope)
from frame3d import Frame, Material, Member, Node, Section, solve
from internal_forces import member_diagram

E, NU = 2.1e11, 0.3
SEC = Section("S", 0.01, 4e-5, 3e-4, 8e-7)
MAT = Material("STEEL", E, NU, 7850.0)
L, W, P = 6.0, -20e3, -30e3


def beam(n: int = 4) -> Frame:
    """简支梁，n 个单元。"""
    f = Frame()
    for k in range(n + 1):
        f.nodes[k + 1] = Node(k + 1, L * k / n, 0.0, 0.0)
    for k in range(n):
        f.members[k + 1] = Member(k + 1, k + 1, k + 2, "S", "STEEL")
    f.sections["S"] = SEC
    f.materials["STEEL"] = MAT
    f.supports[1] = (1, 1, 1, 1, 0, 0)
    f.supports[n + 1] = (0, 1, 1, 1, 0, 0)
    return f


def one_element(**cases) -> Frame:
    """单单元简支梁，每个关键字是一个工况的均布荷载强度。"""
    f = Frame()
    f.nodes[1] = Node(1, 0.0, 0.0, 0.0)
    f.nodes[2] = Node(2, L, 0.0, 0.0)
    f.members[1] = Member(1, 1, 2, "S", "STEEL")
    f.sections["S"] = SEC
    f.materials["STEEL"] = MAT
    f.supports[1] = (1, 1, 1, 1, 0, 0)
    f.supports[2] = (0, 1, 1, 1, 0, 0)
    f.load_cases.pop("default", None)
    for name, w in cases.items():
        f.case(name).member_loads[1] = (0.0, 0.0, w)
    return f


# ------------------------------------------------- 退化情形

def test_a_single_case_envelope_equals_that_case():
    """只有一个工况时，上下包线都等于它自己——包络不能凭空造出余量。"""
    f = one_element(DL=W)
    sol = solve(f)
    env = member_envelope(f, sol, 1, stations=11)
    d = member_diagram(f, sol, 1, "DL", at_x=env.x)
    for comp in ("N", "Vy", "Mz"):
        assert env.upper[comp] == pytest.approx(d.component(comp))
        assert env.lower[comp] == pytest.approx(d.component(comp))
        assert set(env.upper_case[comp]) == {"DL"}


def test_the_envelope_brackets_every_case_at_every_station():
    """上包线 ≥ 每个工况、下包线 ≤ 每个工况，逐点成立。这是包络的定义。"""
    f = one_element(DL=W, LL=0.5 * W, WL=-0.8 * W)
    sol = solve(f)
    env = member_envelope(f, sol, 1, stations=21)
    for name in env.cases:
        d = member_diagram(f, sol, 1, name, at_x=env.x)
        for comp in ("N", "Vy", "Vz", "Mz", "My", "T"):
            assert np.all(env.upper[comp] >= d.component(comp) - 1e-9), (name, comp)
            assert np.all(env.lower[comp] <= d.component(comp) + 1e-9), (name, comp)


def test_symmetric_cases_give_a_symmetric_envelope():
    """一正一负两个等大工况，上下包线必须互为相反数。"""
    f = one_element(PLUS=W, MINUS=-W)
    env = member_envelope(f, solve(f), 1, stations=11)
    assert env.upper["Mz"] == pytest.approx(-env.lower["Mz"])


# ------------------------------------------------- 控制组合

def test_different_cases_govern_at_different_points_of_one_member():
    """这条是整个模块的核心判据。

    构造：均布荷载工况在跨中弯矩最大，端部集中力偶工况在支座弯矩最大。
    逐点包络能分辨出跨中归 UDL、端部归 MOM；
    "先各工况取极值再比"的做法只会给出一个全局最大值，位置也是错的。
    """
    f = Frame()
    f.nodes[1] = Node(1, 0.0, 0.0, 0.0)
    f.nodes[2] = Node(2, L, 0.0, 0.0)
    f.members[1] = Member(1, 1, 2, "S", "STEEL")
    f.sections["S"] = SEC
    f.materials["STEEL"] = MAT
    f.supports[1] = (1, 1, 1, 1, 1, 0)          # 固接端可以承受弯矩
    f.supports[2] = (0, 1, 1, 1, 0, 0)
    f.load_cases.pop("default", None)
    f.case("UDL").member_loads[1] = (0.0, 0.0, W)
    f.case("TIP").nodal_loads[2] = (0.0, 0.0, 0.6 * P, 0.0, 0.0, 0.0)

    env = member_envelope(f, solve(f), 1, stations=41)
    governs = env.upper_case["Mz"] + env.lower_case["Mz"]
    assert set(governs) == {"UDL", "TIP"}, "两个工况都得在某些点上控制"

    # 支座附近与跨中附近由不同工况控制
    near_support = env.lower_case["Mz"][1]
    mid = int(np.argmin(np.abs(env.x - L / 2)))
    assert env.upper_case["Mz"][mid] != near_support or \
        env.lower_case["Mz"][mid] != near_support


def test_the_extreme_reports_which_case_governs():
    f = one_element(SMALL=0.5 * W, BIG=W)
    env = member_envelope(f, solve(f), 1, stations=21)
    got = env.extreme("Mz")
    assert got["case"] == "BIG"
    assert abs(got["value"]) == pytest.approx(abs(W) * L ** 2 / 8, rel=1e-9)
    assert got["x"] == pytest.approx(L / 2, abs=L / 40)


def test_governing_lists_cases_by_how_often_they_control():
    f = one_element(BIG=W, TINY=0.01 * W)
    env = member_envelope(f, solve(f), 1, stations=21)
    order = env.governing("Mz")
    assert order[0] == "BIG"


# ------------------------------------------------- 组合优先

def test_combos_are_used_when_they_exist():
    """有组合就拿组合做包络。单工况混进来会让包络虚大——
    1.0DL 和 1.3DL 同时在场时，前者永远不控制，只是噪声。"""
    f = one_element(DL=W, LL=0.5 * W)
    f.combos["1.3DL"] = {"DL": 1.3}
    f.combos["DL+LL"] = {"DL": 1.3, "LL": 1.5}
    assert envelope_cases(f) == ("1.3DL", "DL+LL")
    env = member_envelope(f, solve(f), 1, stations=11)
    assert set(env.cases) == {"1.3DL", "DL+LL"}


def test_cases_are_used_when_there_are_no_combos():
    f = one_element(DL=W, LL=0.5 * W)
    assert set(envelope_cases(f)) == {"DL", "LL"}


def test_an_explicit_case_list_overrides_the_default():
    f = one_element(DL=W, LL=0.5 * W)
    f.combos["1.3DL"] = {"DL": 1.3}
    env = member_envelope(f, solve(f), 1, cases=("DL",), stations=5)
    assert env.cases == ("DL",)


# ------------------------------------------------- 全结构汇总

def test_the_summary_finds_the_worst_member_and_its_case():
    f = beam(4)
    f.load_cases.pop("default", None)
    for mid in f.members:
        f.case("DL").member_loads[mid] = (0.0, 0.0, W)
        f.case("HALF").member_loads[mid] = (0.0, 0.0, 0.5 * W)
    s = governing_summary(f, solve(f), stations=21)
    worst = s["worst"]["Mz"]
    assert worst["case"] == "DL"
    assert abs(worst["value"]) == pytest.approx(abs(W) * L ** 2 / 8, rel=1e-3)
    # 跨中那根杆件控制
    assert worst["member"] in (2, 3)


def test_the_summary_separates_global_extremes_from_local_governance():
    """"拿到全局极值"和"在某处控制"是两回事，汇总里必须分开报。

    三个同号组合：最大的取下包线、最小的取上包线，中间那个哪边都不沾——
    它才是真正可以删掉的那个。
    """
    f = one_element(DL=W)
    f.combos["BIG"] = {"DL": 1.3}
    f.combos["MID"] = {"DL": 1.0}
    f.combos["SMALL"] = {"DL": 0.5}
    s = governing_summary(f, solve(f), stations=21)
    assert "MID" in s["never_governs_anywhere"]
    assert set(s["governs_somewhere"]) == {"BIG", "SMALL"}


def test_every_member_gets_an_envelope():
    f = beam(4)
    f.load_cases["default"].member_loads = {mid: (0.0, 0.0, W) for mid in f.members}
    envs = all_envelopes(f, solve(f), stations=9)
    assert set(envs) == set(f.members)


# ------------------------------------------------- 测点对齐

def test_stations_from_all_cases_are_merged():
    """两个工况的集中力位置不同时，测点必须取并集。

    只用其中一套的话，另一个工况的剪力跳跃点会被漏掉，
    包络在那里就是错的——而且图上看不出来。
    """
    from span_loads import POINT, SpanLoad

    f = one_element()
    f.case("A").member_spans[1] = [SpanLoad(POINT, (0.0, 0.0, P), a=1.5)]
    f.case("B").member_spans[1] = [SpanLoad(POINT, (0.0, 0.0, P), a=4.5)]
    env = member_envelope(f, solve(f), 1, stations=5)
    for a in (1.5, 4.5):
        assert np.any(np.isclose(env.x, a, atol=1e-6)), f"漏了 {a} 处的测点"


def test_an_unknown_case_is_refused():
    f = one_element(DL=W)
    with pytest.raises(ValueError, match="NOPE"):
        member_envelope(f, solve(f), 1, cases=("NOPE",))


def test_a_case_that_governs_locally_is_not_called_never_governing():
    """只按全局极值判断"从不控制"是错的。

    构造：DL+WL 在某根梁的端部控制，但全结构最大值归 DL−WL。
    第一版把 DL+WL 列进了 never_governs，按它去删组合会删掉真正起作用的。
    """
    f = Frame()
    for k, (x, z) in enumerate([(0.0, 0.0), (L, 0.0), (0.0, 3.6), (L, 3.6)], 1):
        f.nodes[k] = Node(k, x, 0.0, z)
        f.supports[k] = (0, 1, 0, 1, 0, 1)      # 只留平面内自由度
    f.members[1] = Member(1, 1, 3, "S", "STEEL")
    f.members[2] = Member(2, 3, 4, "S", "STEEL")
    f.members[3] = Member(3, 2, 4, "S", "STEEL")
    f.sections["S"] = SEC
    f.materials["STEEL"] = MAT
    f.supports[1] = (1, 1, 1, 1, 1, 1)
    f.supports[2] = (1, 1, 1, 1, 1, 1)
    f.load_cases.pop("default", None)
    f.case("DL").member_loads[2] = (0.0, 0.0, W)
    f.case("WL").nodal_loads[4] = (80e3, 0.0, 0.0, 0.0, 0.0, 0.0)
    f.combos["DL+WL"] = {"DL": 1.3, "WL": 1.5}
    f.combos["DL-WL"] = {"DL": 1.3, "WL": -1.5}

    sol = solve(f)
    s = governing_summary(f, sol, stations=51)
    assert set(s["governs_somewhere"]) == {"DL+WL", "DL-WL"}
    assert s["never_governs_anywhere"] == [], s["never_governs_anywhere"]

    # 真正的判据：**同一根梁上控制组合发生了对调**。
    # 上包线 i 端归一个、j 端归另一个——这正是"先取极值再比"分辨不出来的东西
    env = member_envelope(f, sol, 2, stations=51)
    assert env.upper_case["Mz"][0] != env.upper_case["Mz"][-1]
    assert env.lower_case["Mz"][0] != env.lower_case["Mz"][-1]
    assert set(env.governing("Mz")) == {"DL+WL", "DL-WL"}


def test_a_case_that_truly_never_governs_is_reported():
    """真正一处都不控制的组合要报出来——它要么多余，要么系数写错了。"""
    f = one_element(DL=W, LL=0.4 * W)
    f.combos["ALL"] = {"DL": 1.3, "LL": 1.5}
    f.combos["HALF"] = {"DL": 0.65, "LL": 0.75}
    f.combos["MID"] = {"DL": 1.0, "LL": 1.1}
    s = governing_summary(f, solve(f), stations=21)
    assert "MID" in s["never_governs_anywhere"]
    assert "ALL" not in s["never_governs_anywhere"]
