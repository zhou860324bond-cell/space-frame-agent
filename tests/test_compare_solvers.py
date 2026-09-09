"""求解器并排对比。

Abaqus 在测试机上一定没有，所以这里验的是「除 Abaqus 那一步之外的全部」：
字段对齐、误差度量、以及拿不到 Abaqus 时的降级行为。
只要这些站得住，真机上跑出来的数就可信。
"""

import numpy as np
import pytest

import abaqus_backend
from abaqus_backend import compare_rows, normalized_error
from agent import Session

MATERIALS = [{"name": "STEEL", "E": 2.1e11, "nu": 0.3}]
SECTIONS = [
    {"name": "COLUMN", "A": 0.012, "Iy": 8.0e-5, "Iz": 2.4e-4, "J": 1.0e-6},
    {"name": "BEAM", "A": 0.010, "Iy": 4.0e-5, "Iz": 3.0e-4, "J": 8.0e-7},
]


@pytest.fixture
def solved():
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    beams = s.generate_frame(spans=[6.0], storeys=[3.6], bays=[6.0],
                             column_section="COLUMN", beam_section="BEAM",
                             material="STEEL").payload["beam_member_ids"]
    s.set_load_cases(cases=[{"name": "DL", "member_loads":
                             [{"member": m, "w": [0, 0, -20e3]} for m in beams]}])
    s.solve_model()
    return s


# ------------------------------ 误差度量 ------------------------------

def test_identical_fields_give_zero_error():
    assert normalized_error([1.0, -2.0, 3.0], [1.0, -2.0, 3.0]) == 0.0


def test_uniform_scaling_gives_that_scale():
    """整体放大 1%，归一化误差就该是 1% —— 度量本身不能带放大或衰减。"""
    b = [3.0, -7.0, 0.5, 11.0]
    assert normalized_error([x * 1.01 for x in b], b) == pytest.approx(0.01)


def test_zero_reference_is_null_not_zero():
    """参考解整体为零时归一化没有意义，必须是 None。

    若这里返回 0.0，界面上一个从未算对过的分量会显示成「完美吻合」。
    """
    assert normalized_error([0.0, 0.0], [0.0, 0.0]) is None
    assert normalized_error([1e-9, 0.0], [0.0, 0.0]) is None


def test_error_is_not_symmetric_reference_is_the_second_argument():
    """以 Abaqus 为参考，分母取的是第二个参数 —— 顺序弄反会得到不同的数。"""
    assert normalized_error([2.0], [1.0]) == pytest.approx(1.0)
    assert normalized_error([1.0], [2.0]) == pytest.approx(0.5)


# ------------------------------ 字段对齐 ------------------------------

def test_native_rows_cover_every_field_the_comparison_reads(solved):
    """自研侧摊平后的字段必须覆盖比较用到的全部分量。

    少一个字段不会报错，compare_rows 会当成 0 —— 于是那一项静悄悄地
    显示成「有偏差」或「完美吻合」，两种都是假的。这条测试盯的就是这个。
    """
    rows = solved._native_rows("DL")
    needed = {k for k, _ in abaqus_backend._COMPARE_FIELDS}
    for nid, row in rows.items():
        assert needed <= set(row), f"节点 {nid} 缺少 {needed - set(row)}"


def test_native_rows_span_every_node(solved):
    assert set(solved._native_rows("DL")) == set(solved.frame.nodes)


def test_native_rows_are_si_units_not_display_units(solved):
    """两侧都必须是 m 与 N。任一侧混进 mm 或 kN，误差会变成 1e3 量级。"""
    rows = solved._native_rows("DL")
    assert max(abs(r["u3"]) for r in rows.values()) < 0.1          # m，不是 mm
    assert max(abs(r["rf3"]) for r in rows.values()) > 1e3         # N，不是 kN


def test_free_nodes_carry_zero_reactions(solved):
    """非支座节点反力恒为零；写出来是为了让两侧字段对齐。"""
    rows = solved._native_rows("DL")
    free = set(solved.frame.nodes) - set(solved.frame.supports)
    assert free
    for nid in free:
        assert rows[nid]["rf1"] == rows[nid]["rf2"] == rows[nid]["rf3"] == 0.0


# ------------------------------ 比对本身 ------------------------------

def test_comparing_a_result_against_itself_is_exactly_zero(solved):
    """自研 vs 自研必须逐分量为零。非零就说明比对链路本身有 bug。"""
    rows = solved._native_rows("DL")
    out = compare_rows(rows, rows)
    assert out["nodes_compared"] == len(rows)
    for key, v in out["errors"].items():
        assert v["e"] in (0.0, None), f"{key} 自比出现 {v['e']}"


def test_a_known_perturbation_comes_back_as_that_number(solved):
    """把参考解整体放大 1%，对比结果就该报 1%。"""
    rows = solved._native_rows("DL")
    perturbed = {n: {k: v * 1.01 for k, v in r.items()} for n, r in rows.items()}
    out = compare_rows(rows, perturbed)
    e = out["errors"]["u3"]["e"]
    assert e == pytest.approx(0.01 / 1.01, rel=1e-9)


def test_peak_node_is_the_one_with_the_largest_displacement(solved):
    rows = solved._native_rows("DL")
    out = compare_rows(rows, rows)
    worst = max(rows, key=lambda n: np.hypot(rows[n]["u1"],
                                             np.hypot(rows[n]["u2"], rows[n]["u3"])))
    assert out["peak"]["node"] == worst
    assert out["peak"]["native_mm"] == pytest.approx(out["peak"]["abaqus_mm"])


def test_peak_display_scale_can_be_mm_native():
    rows = {1: {"u3": 2.5}}
    out = compare_rows(rows, rows, displacement_scale=1.0)
    assert out["peak"]["native_mm"] == pytest.approx(2.5)


def test_disjoint_node_sets_are_reported_not_silently_empty():
    a = {1: {"u3": 1.0}}
    b = {99: {"u3": 1.0}}
    assert "error" in compare_rows(a, b)


def test_only_shared_nodes_are_compared():
    """一侧多出的节点直接略过，不该把对比拖成全零。"""
    a = {1: {"u3": 1.0}, 2: {"u3": 2.0}}
    b = {1: {"u3": 1.0}}
    out = compare_rows(a, b)
    assert out["nodes_compared"] == 1
    assert out["errors"]["u3"]["e"] == 0.0


# ------------------------------ 降级行为 ------------------------------

def test_comparison_refuses_an_invalid_model():
    s = Session()
    r = s.compare_solvers()
    assert not r.ok
    assert "errors" in r.payload


def test_comparison_reports_an_unknown_case(solved):
    r = solved.compare_solvers(case="NOPE")
    assert not r.ok
    assert "NOPE" in r.payload["error"]


def test_abaqus_is_looked_for_beyond_the_PATH(monkeypatch, tmp_path):
    """Abaqus 只把命令放进「Abaqus Command」终端的 PATH。从普通命令行启动时
    单看 PATH 会得出「没装」的结论 —— 机器上明明装着。"""
    monkeypatch.setattr(abaqus_backend.shutil, "which", lambda _: None)
    fake = tmp_path / "Commands"
    fake.mkdir()
    (fake / "abaqus.bat").write_text("@echo off\n")
    monkeypatch.setattr(abaqus_backend, "_WINDOWS_HINTS", (str(fake),))
    assert abaqus_backend.find_abaqus() == str(fake / "abaqus.bat")
    assert abaqus_backend.abaqus_available()


def test_the_PATH_wins_when_it_has_one(monkeypatch):
    monkeypatch.setattr(abaqus_backend.shutil, "which", lambda _: "/opt/abaqus")
    assert abaqus_backend.find_abaqus() == "/opt/abaqus"


def test_no_abaqus_anywhere_is_reported_as_missing(monkeypatch):
    monkeypatch.setattr(abaqus_backend.shutil, "which", lambda _: None)
    monkeypatch.setattr(abaqus_backend, "_WINDOWS_HINTS", ())
    assert abaqus_backend.find_abaqus() is None
    assert not abaqus_backend.abaqus_available()


def test_the_missing_abaqus_message_says_how_to_fix_it(monkeypatch, tmp_path, solved):
    """报「没装」不够 —— 得告诉用户去哪个终端启动。"""
    monkeypatch.setattr(abaqus_backend, "abaqus_available", lambda: False)
    monkeypatch.setattr(abaqus_backend, "find_abaqus", lambda: None)
    with pytest.raises(abaqus_backend.AbaqusError) as exc:
        abaqus_backend.solve(solved.model, tmp_path, case="DL")
    assert "Abaqus Command" in str(exc.value)


def test_comparison_says_which_side_failed(solved, monkeypatch, tmp_path):
    """Abaqus 跑不起来时，错误必须点明是 Abaqus 那一侧 —— 否则用户会以为
    自研求解器坏了。"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(abaqus_backend, "abaqus_available", lambda: False)
    monkeypatch.setattr(abaqus_backend, "find_abaqus", lambda: None)
    r = solved.compare_solvers()
    assert not r.ok
    assert "Abaqus" in r.payload["error"]
    assert "detail" in r.payload
