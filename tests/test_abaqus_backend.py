"""Abaqus 后端的验证。

本机没有 Abaqus，所以真正能验的是**除了跑作业以外的每一环**：日志判读、
CSV 解析、结果整理、以及没装 Abaqus 时的失败路径。这些恰恰是最容易写错的部分——
跑作业那一步反而只是一次 subprocess 调用。
"""
from pathlib import Path

import pytest

import abaqus_backend as AB
from agent import Session
from units import MM, convert_model

MATERIALS = [{"name": "STEEL", "E": 2.1e11, "nu": 0.3}]
SECTIONS = [
    {"name": "COLUMN", "A": 0.012, "Iy": 8e-5, "Iz": 2.4e-4, "J": 1e-6},
    {"name": "BEAM", "A": 0.010, "Iy": 4e-5, "Iz": 3e-4, "J": 8e-7},
]


@pytest.fixture
def session():
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    beams = s.generate_frame(spans=[6.0, 6.0], storeys=[3.6], bays=[6.0],
                             column_section="COLUMN", beam_section="BEAM",
                             material="STEEL").payload["beam_member_ids"]
    s.set_load_cases(cases=[{"name": "DL", "member_loads":
                             [{"member": m, "w": [0, 0, -20e3]} for m in beams]}])
    return s


# ----------------------------------------------------- 日志判读

def test_fatal_lines_are_picked_out_of_the_message_file():
    """返回码不总能反映分析是否真的完成，日志才是准的。"""
    log = ("  STEP 1 STATIC\n"
           " ***ERROR: TOO MANY ATTEMPTS MADE FOR THIS INCREMENT\n"
           " ***WARNING: ELEMENT 12 IS DISTORTED\n")
    scan = AB.scan_log(log)
    assert not scan["ok"]
    assert any("TOO MANY ATTEMPTS" in line for line in scan["fatal"])
    assert any("DISTORTED" in line for line in scan["warnings"])


@pytest.mark.parametrize("line", [
    " ***ERROR: something went wrong",
    " THE ANALYSIS HAS BEEN TERMINATED DUE TO PREVIOUS ERRORS",
    " ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE 7",
    " ***WARNING: SOLVER PROBLEM. ZERO PIVOT WHEN PROCESSING D.O.F. 3",
])
def test_every_fatal_pattern_is_caught(line):
    assert not AB.scan_log(line)["ok"]


def test_a_clean_log_passes():
    assert AB.scan_log(" STEP COMPLETED\n THE ANALYSIS HAS COMPLETED SUCCESSFULLY")["ok"]


def test_warnings_alone_do_not_fail_the_job():
    scan = AB.scan_log(" ***WARNING: ELEMENT 3 HAS A LARGE ASPECT RATIO")
    assert scan["ok"] and scan["warnings"]


# ----------------------------------------------------- CSV 与结果整理

def _write_csv(path: Path, rows: dict[int, dict[str, float]]) -> Path:
    cols = ("u1", "u2", "u3", "ur1", "ur2", "ur3",
            "rf1", "rf2", "rf3", "rm1", "rm2", "rm3")
    lines = ["node," + ",".join(cols)]
    for node, row in rows.items():
        lines.append(str(node) + "," + ",".join(f"{row.get(c, 0.0):.12e}" for c in cols))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_result_csv_round_trips(tmp_path):
    path = _write_csv(tmp_path / "r.csv", {3: {"u3": -1.5e-3, "rf3": 4.2e4}})
    back = AB.read_result_csv(path)
    assert back[3]["u3"] == pytest.approx(-1.5e-3)
    assert back[3]["rf3"] == pytest.approx(4.2e4)


def test_summary_matches_the_shape_of_the_native_solver(session):
    """两个后端的返回结构必须对齐，否则没法直接比对。"""
    native = session.solve_model().payload["cases"]["DL"]
    results = {}
    for nid in session.frame.order():
        d = session.frame.node_dofs(nid)
        u = session.solution["DL"].U[d[:3]]
        r = session.solution["DL"].R[d[:3]]
        results[nid] = {"u1": u[0], "u2": u[1], "u3": u[2],
                        "rf1": r[0], "rf2": r[1], "rf3": r[2]}
    summary = AB.summarise(session.model, results, "DL")

    assert summary["backend"] == "abaqus"
    assert summary["max_displacement_mm"] == pytest.approx(
        native["max_displacement_mm"], rel=1e-6)
    assert summary["at_node"] == native["at_node"]
    expected = session.query_results(what="reactions", case="DL").payload
    assert summary["vertical_total_kN"] == pytest.approx(
        expected["vertical_total_kN"], rel=1e-6)
    assert set(summary["reactions"]) == set(expected["reactions"])


def test_summary_reports_only_support_reactions(session):
    session.solve_model()
    results = {nid: {"u3": -1e-3, "rf3": 1e3} for nid in session.frame.order()}
    summary = AB.summarise(session.model, results, "DL")
    assert len(summary["reactions"]) == len(session.frame.supports)
    assert summary["nodes_returned"] == len(session.frame.nodes)


def test_summary_does_not_multiply_mm_model_displacement_twice(session):
    model_mm = convert_model(session.model, MM)
    summary = AB.summarise(model_mm, {1: {"u3": -2.5}}, "DL")
    assert summary["max_displacement_mm"] == pytest.approx(2.5)


# ----------------------------------------------------- 失败路径

def test_backend_reports_a_clear_error_when_abaqus_is_absent(session, monkeypatch,
                                                             tmp_path):
    monkeypatch.setattr(AB, "find_abaqus", lambda: None)
    monkeypatch.chdir(tmp_path)
    r = session.solve_with_abaqus()
    assert not r.ok
    assert "abaqus" in r.payload["error"].lower()
    assert "solve_model" in r.payload["hint"], "应当指回不需要 Abaqus 的那条路"


def test_backend_refuses_an_invalid_model(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    s.model["nodes"] = []
    r = s.solve_with_abaqus()
    assert not r.ok and "errors" in r.payload


def test_backend_rejects_an_unknown_case(session, monkeypatch, tmp_path):
    # 打桩要打在 solve() 真正用的那个函数上：它取的是 abaqus 命令的路径，
    # 不是那个布尔量。桩打偏了，这条测试会因为「没装 Abaqus」而假通过。
    monkeypatch.setattr(AB, "find_abaqus", lambda: "abaqus")
    monkeypatch.setattr(AB, "find_extractor", lambda root=None: Path("x.py"))
    monkeypatch.chdir(tmp_path)
    r = session.solve_with_abaqus(case="NOPE")
    assert not r.ok and "NOPE" in r.payload["error"]


def test_extractor_is_found_in_the_repo():
    assert AB.find_extractor() is not None, "读 ODB 的脚本必须能被定位到"


@pytest.mark.skipif(not AB.abaqus_available(), reason="本机没有 Abaqus")
def test_real_abaqus_run_matches_the_native_solver(session, tmp_path):
    """装了 Abaqus 才跑：B33 与本程序同为 Euler-Bernoulli，应当吻合到数值精度。"""
    native = session.solve_model().payload["cases"]["DL"]
    summary = AB.solve(session.model, tmp_path, case="DL", element="B33")
    assert summary["max_displacement_mm"] == pytest.approx(
        native["max_displacement_mm"], rel=1e-4)


# ----------------------------------------------------- 新荷载类型的导出边界

def _beam_model(spans=None, settlements=None):
    return {"units": "N-m-Pa", "materials": MATERIALS, "sections": SECTIONS,
            "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                      {"id": 2, "x": 6, "y": 0, "z": 0}],
            "members": [{"id": 1, "i": 1, "j": 2,
                         "section": "BEAM", "material": "STEEL"}],
            "supports": [{"node": 1, "fix": [1] * 6}, {"node": 2, "fix": [1] * 6}],
            "load_cases": [{"name": "C", "member_spans": spans or [],
                            "settlements": settlements or []}]}


def test_a_uniform_span_load_exports_as_dload(tmp_path):
    from inp_writer import write_inp
    path = write_inp(_beam_model(spans=[{"member": 1, "kind": "uniform",
                                         "w1": [0, 0, -5e3]}]),
                     "B33", tmp_path / "u.inp", case="C")
    text = path.read_text()
    assert "*DLOAD" in text and "E1, PZ, -5000" in text


@pytest.mark.parametrize("span", [
    {"member": 1, "kind": "point", "w1": [0, 0, -3e4], "a": 2.0},
    {"member": 1, "kind": "trapezoid", "w1": [0, 0, 0], "w2": [0, 0, -1e4]},
])
def test_unsupported_span_loads_are_refused_not_approximated(span, tmp_path):
    """Abaqus 的 *DLOAD 只能沿单元均布。悄悄按等效均布导出去，
    两边算的就不是同一个模型了 —— 而对比页的全部意义正是"同一个模型"。
    """
    from inp_writer import write_inp
    with pytest.raises(ValueError) as exc:
        write_inp(_beam_model(spans=[span]), "B33", tmp_path / "x.inp", case="C")
    assert span["kind"] in str(exc.value)
    assert "自研求解器不受此限制" in str(exc.value)


def test_a_settlement_exports_as_a_prescribed_boundary(tmp_path):
    """支座沉降 Abaqus 原生支持：带幅值的 *BOUNDARY，且必须写在分析步**里面**。"""
    from inp_writer import write_inp
    path = write_inp(_beam_model(settlements=[{"node": 2,
                                               "d": [0, 0, -0.01, 0, 0, 0]}]),
                     "B33", tmp_path / "s.inp", case="C")
    lines = path.read_text().splitlines()
    step = next(i for i, l in enumerate(lines) if l.startswith("*STEP"))
    assert " 2, 3, 3, -0.01" in lines[step:], "带幅值的边界条件必须在步内"
    assert "PERTURBATION" not in lines[step], "给定位移不能放在线性摄动步里"
