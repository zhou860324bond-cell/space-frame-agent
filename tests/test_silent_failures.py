"""静默失败检测（silent_failures）的单元测试。

"静默失败"是指：求解器跑通了、没有报错，但结果是错的或可疑的。
这类错误比直接报错更危险——用户会以为结果可信。

9 项检测：
1. insufficient_supports    支座约束不足
2. excessive_displacement    位移过大
3. zero_reaction_with_load   有载荷但反力为零
4. zero_internal_force       有载荷但内力为零
5. no_applied_load           **一个荷载都没有却照样求解**
6. near_singular_stiffness   刚度矩阵接近奇异
7. reaction_load_balance     反力与外载荷不平衡
8. load_magnitude_anomaly    载荷量级异常
9. section_orientation        截面主轴方向风险

这里测四件事：
1. 正常模型的检测结果结构正确（8 项齐全、每项字段完整）
2. 已知异常能被对应检测项抓住（截面反向、位移过大等）
3. summarize_findings 能正确汇总通过/警告/失败数
4. format_findings 能输出可读文本，verbose 模式包含全部项
"""

from __future__ import annotations


import pytest

from agent import Session
import silent_failures as sf


MATERIALS = [{"name": "Q355", "E": 2.06e11, "nu": 0.3, "density": 7850.0}]
SECTIONS = [{"name": "COL", "A": 0.0147, "Iy": 4.2e-5, "Iz": 1.18e-3, "J": 9e-7},
            {"name": "BEAM", "A": 0.010, "Iy": 4.0e-5, "Iz": 3.0e-4, "J": 8e-7}]
# 截面主轴反向：Iy > Iz，正常应该 Iz >= Iy（强轴绕 Z）
SECTIONS_REVERSED = [{"name": "COL", "A": 0.0147, "Iy": 1.18e-3, "Iz": 4.2e-5, "J": 9e-7},
                      {"name": "BEAM", "A": 0.010, "Iy": 3.0e-4, "Iz": 4.0e-5, "J": 8e-7}]


def _solved(sections=SECTIONS, load: float = -10e3, spans=(6.0,), storeys=(3.6,)) -> Session:
    s = Session()
    s.define_materials_and_sections(MATERIALS, sections)
    g = s.generate_frame(spans=list(spans), storeys=list(storeys),
                         column_section="COL", beam_section="BEAM", material="Q355")
    s.set_load_cases(cases=[{"name": "DL", "member_loads":
        [{"member": m, "w": [0, 0, load]} for m in g.payload["beam_member_ids"]]}])
    s.solve_model()
    return s


# --------------------------------------------------------- 结构正确性

def test_all_checks_run_on_normal_model():
    s = _solved()
    findings = sf.detect_silent_failures(s.frame, s.solution)
    assert len(findings) == 9, f"应该跑 9 项检测，实得 {len(findings)}"
    # 每项字段必须齐全
    for f in findings:
        for key in ("id", "name", "severity", "status", "message"):
            assert key in f, f"检测项缺少字段 {key}: {f}"
        assert f["status"] in ("pass", "warn", "fail")
        assert f["severity"] in ("critical", "warning", "info")


def test_each_finding_has_a_suggestion_when_not_passing():
    s = _solved(sections=SECTIONS_REVERSED)  # 截面反向会触发 warning
    findings = sf.detect_silent_failures(s.frame, s.solution)
    for f in findings:
        if f["status"] != "pass":
            assert f.get("suggestion"), f"异常项 {f['name']} 必须给出修复建议"


# --------------------------------------------------------- 已知异常触发

def test_reversed_section_triggers_orientation_warning():
    """截面 Iy > Iz（强轴绕 Y 而不是 Z）应该触发截面主轴方向风险。"""
    s = _solved(sections=SECTIONS_REVERSED)
    findings = sf.detect_silent_failures(s.frame, s.solution)
    orient = next(f for f in findings if f["id"] == "section_orientation")
    assert orient["status"] != "pass", "截面主轴反向应该触发警告"


def test_normal_section_passes_orientation_check():
    s = _solved()
    findings = sf.detect_silent_failures(s.frame, s.solution)
    orient = next(f for f in findings if f["id"] == "section_orientation")
    assert orient["status"] == "pass"


def test_excessive_displacement_triggers_on_huge_load():
    """悬臂梁自由端加集中力，端部位移可达结构尺寸的 60%，应触发位移过大。"""
    s = Session()
    s.define_materials_and_sections(MATERIALS, [
        {"name": "BEAM", "A": 0.01, "Iy": 1e-6, "Iz": 1e-6, "J": 1e-8}])
    s.set_model(model={
        "units": "N-m-Pa",
        "materials": MATERIALS,
        "sections": [{"name": "BEAM", "A": 0.01, "Iy": 1e-6, "Iz": 1e-6, "J": 1e-8}],
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                  {"id": 2, "x": 6.0, "y": 0, "z": 0}],
        "members": [{"id": 1, "i": 1, "j": 2, "section": "BEAM", "material": "Q355"}],
        "supports": [{"node": 1, "fix": [1, 1, 1, 1, 1, 1]}],
        "load_cases": [{"name": "DL", "nodal_loads": [
            {"node": 2, "load": [0, 0, -10e3, 0, 0, 0]}]}],
    })
    s.solve_model()
    findings = sf.detect_silent_failures(s.frame, s.solution)
    disp = next(f for f in findings if f["id"] == "excessive_displacement")
    assert disp["status"] != "pass", f"悬臂梁端部位移应过大，实际：{disp['message']}"


def test_normal_model_passes_support_check():
    s = _solved()
    findings = sf.detect_silent_failures(s.frame, s.solution)
    supp = next(f for f in findings if f["id"] == "insufficient_supports")
    assert supp["status"] == "pass"


def test_normal_model_passes_reaction_check():
    s = _solved()
    findings = sf.detect_silent_failures(s.frame, s.solution)
    reac = next(f for f in findings if f["id"] == "zero_reaction_with_load")
    assert reac["status"] == "pass", "有荷载的正常模型应该有非零反力"


def test_normal_model_passes_balance_check():
    s = _solved()
    findings = sf.detect_silent_failures(s.frame, s.solution)
    bal = next(f for f in findings if f["id"] == "reaction_load_balance")
    assert bal["status"] == "pass", "求解器保证平衡，检测应该通过"


# --------------------------------------------------------- only / exclude

def test_only_filter_runs_subset():
    s = _solved()
    findings = sf.detect_silent_failures(s.frame, s.solution,
                                          only=["insufficient_supports", "excessive_displacement"])
    assert len(findings) == 2
    ids = {f["id"] for f in findings}
    assert ids == {"insufficient_supports", "excessive_displacement"}


def test_exclude_filter_skips_specified():
    s = _solved()
    findings = sf.detect_silent_failures(s.frame, s.solution,
                                          exclude=["near_singular_stiffness"])
    ids = {f["id"] for f in findings}
    assert "near_singular_stiffness" not in ids
    assert len(findings) == 8


# --------------------------------------------------------- summarize / format

def test_summarize_counts_correctly():
    s = _solved(sections=SECTIONS_REVERSED)  # 至少触发一个 warning
    findings = sf.detect_silent_failures(s.frame, s.solution)
    summary = sf.summarize_findings(findings)
    total = summary["total_checks"]
    passed = summary["passed"]
    warnings = summary["warnings"]
    critical = summary["critical_issues"]
    assert total == len(findings)
    assert passed + warnings + critical <= total, \
        f"通过({passed}) + 警告({warnings}) + 严重({critical}) 不应超过总数({total})"
    assert "overall" in summary
    assert summary["overall"] in ("pass", "warn", "fail")


def test_format_findings_returns_nonempty_string():
    s = _solved()
    findings = sf.detect_silent_failures(s.frame, s.solution)
    text = sf.format_findings(findings)
    assert isinstance(text, str)
    assert len(text) > 0
    # 默认模式只显示非通过项，正常模型应该提示"全部通过"
    assert "通过" in text or "全部" in text


def test_format_findings_verbose_includes_all_items():
    s = _solved()
    findings = sf.detect_silent_failures(s.frame, s.solution)
    text = sf.format_findings(findings, verbose=True)
    # verbose 模式应该包含全部 8 项的名称
    for f in findings:
        assert f["name"] in text, f"verbose 模式应包含 {f['name']}"


# ------------------------------------------------- 没有荷载：这次的盲区

def _frame_without_load(cases=None) -> Session:
    """建好模型、设好支座，但**不给（有效的）荷载**，然后照样求解。"""
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    s.generate_frame(spans=[6.0], storeys=[3.6], column_section="COL",
                     beam_section="BEAM", material="Q355")
    if cases is not None:
        s.set_load_cases(cases=cases)
    s.solve_model()
    return s


def test_a_correct_model_reports_no_critical_at_all():
    """**正常模型必须一条 critical 都不报。**

    这条是为一次真实事故补的：`near_singular_stiffness` 原先拿 `sol.K` 算
    条件数，而那是**施加边界条件之前**的总刚——空间问题恒有 6 个未约束的
    刚体位移模式，按定义就奇异，条件数恒在 1e17~1e19。

    后果是这一项对**任何**模型都报 critical：柱底固接的门式刚架和一个真正的
    机构给出同一个数（实测正常刚架 24×24、秩 18、亏秩正好 6，条件数
    1.21e+19）。一个 100% 误报的 critical 比没有更糟——它让人学会忽略整块
    告警，真信号跟着一起被埋。

    所以这条断言的不是"某一项通过"，而是**整体不许有假警报**。
    """
    s = _solved()
    findings = sf.detect_silent_failures(s.frame, s.solution)
    bad = [f for f in findings if f["status"] == "fail"]
    assert not bad, f"正常模型不该有任何 critical：{[f['id'] for f in bad]}"


@pytest.mark.parametrize("cases, why", [
    (None, "完全没有建工况"),
    ([{"name": "DL"}], "建了工况但里面一条荷载都没有"),
    ([{"name": "DL", "member_loads": [{"member": 999, "w": [0, 0, -10e3]}]}],
     "杆件号写错——set_load_cases 会失败并保持原模型不变，但调用方没看返回值"),
])
def test_solving_without_any_load_is_reported(cases, why):
    """一个荷载都没有却求出一套全零结果，必须报 critical。

    **这是 zero_reaction_with_load / zero_internal_force 的盲区**：那两条的
    前提都是"有外载荷"，没有载荷时它们直接跳过，于是最基本的一种静默失败
    反而没人管——求解返回 ok=True，弯矩内力全零，界面上看不出异常。

    全零结果"肉眼看就是错的"，但只有人盯着看才看得出来。这条检测替人看。
    """
    s = _frame_without_load(cases)
    findings = {f["id"]: f for f in sf.detect_silent_failures(s.frame, s.solution)}
    hit = findings["no_applied_load"]
    assert hit["status"] == "fail", f"{why}：应当报 no_applied_load"
    assert hit["severity"] == "critical"
    assert hit["suggestion"], "必须给出怎么查的建议"


def test_a_loaded_model_passes_the_no_load_check():
    s = _solved()
    findings = {f["id"]: f for f in sf.detect_silent_failures(s.frame, s.solution)}
    assert findings["no_applied_load"]["status"] == "pass"


# ------------------------------------------- 检测结果要能拦住 solve_model

def test_unusable_result_makes_solve_model_fail():
    """判定为「结果不可用」时，solve_model 必须返回 ok=False。

    **这条不是为了好看，是为了把自修复闭环接上。** 系统提示第 4 条写着
    "solve_model 返回错误清单时，读懂它、改模型、重试"——那个闭环是靠 ok
    触发的。以前这些发现只躺在 payload 里，Agent 看到的是"求解成功"，
    于是照样把一套全零结果如实汇报给用户：可溯源，但是个废数。
    """
    s = _frame_without_load([{"name": "DL"}])       # 空工况
    r = s.solve_model()
    assert not r.ok, "没有任何作用的求解不该报成功"
    assert "unusable" in r.payload
    assert "no_applied_load" in r.payload["unusable"]["checks"]
    assert r.payload["unusable"]["next"], "要把怎么办一并给出来"
    # 结果本身仍然留在 payload 里，不是丢掉，是标记为不可用
    assert "cases" in r.payload


def test_a_normal_model_still_solves_ok():
    s = _solved()
    assert s.solve_model().ok
    assert "unusable" not in s.solve_model().payload


def test_excessive_displacement_does_not_block():
    """位移过大是 critical，但**有意不拦**。

    它的数字是真的，只是大得可疑——用户可能就是要看一个很柔的结构。
    拦住它等于替用户决定什么叫"合理"。名单见 RESULT_INVALIDATING。
    """
    assert "excessive_displacement" not in sf.RESULT_INVALIDATING
    assert "no_applied_load" in sf.RESULT_INVALIDATING


# ------------------------------------------- 外载荷合力要把跨荷载算进来

def test_applied_load_counts_span_loads():
    """`_total_applied_load` 必须把 member_spans 算进来。

    原先只统计 nodal_loads 与满跨 member_loads，跨中点荷载、梯形、局部均布
    一概不算——于是 reaction_load_balance 在任何用了跨荷载的模型上都假报
    "反力与外载荷不平衡"（实测 rich_model 残差 11.1%）。

    这个假阳性以前看不见：near_singular_stiffness 那个 bug 让 overall 恒为
    fail，多一条少一条分不出来。修好那个之后它立刻露头。
    """
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    s.generate_frame(spans=[6.0], storeys=[3.6], column_section="COL",
                     beam_section="BEAM", material="Q355")
    beam = [m["id"] for m in s.model["members"] if m["section"] == "BEAM"][0]
    # 只加一个跨中点荷载，别的什么都没有
    s.set_load_cases(cases=[{"name": "P", "member_spans": [
        {"member": beam, "kind": "point", "w1": [0, 0, -30e3], "a": 3.0}]}])
    assert s.solve_model().ok, "只有跨荷载的模型必须算得出来且不被判不可用"

    total = sf._total_applied_load(s.frame, "P")
    assert abs(float(total[2]) + 30e3) < 1.0, \
        f"竖向合力应当是 -30 kN，实得 {total[2]}"

    findings = {f["id"]: f for f in sf.detect_silent_failures(s.frame, s.solution)}
    assert findings["reaction_load_balance"]["status"] == "pass"
    assert findings["no_applied_load"]["status"] == "pass"


def test_settlement_only_case_is_an_action():
    """支座沉降不产生净外力，但它是实实在在的作用，不该被判成空求解。"""
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    s.generate_frame(spans=[6.0], storeys=[3.6], column_section="COL",
                     beam_section="BEAM", material="Q355")
    base = min(int(n["id"]) for n in s.model["nodes"])
    s.set_load_cases(cases=[{"name": "SET", "settlements": [
        {"node": base, "d": [0, 0, -0.005, 0, 0, 0]}]}])
    s.solve_model()
    findings = {f["id"]: f for f in sf.detect_silent_failures(s.frame, s.solution)}
    assert findings["no_applied_load"]["status"] == "pass", \
        "沉降工况不是空求解"


# ---------------- zero_internal_force 的错因要说对 ----------------
# 荷载全部加在支座被约束的方向上时，原先的提示说「可能是机构，或荷载加在孤立
# 节点上」——两条都不是。刚度矩阵好好的，节点也连着杆；荷载只是直接进了反力。

_MS_ZIF = {"materials": [{"name": "STEEL", "E": 2.1e11, "nu": 0.3}],
           "sections": [{"name": "COLUMN", "A": 0.012, "Iy": 8e-5, "Iz": 2.4e-4, "J": 1e-6},
                        {"name": "BEAM", "A": 0.010, "Iy": 4e-5, "Iz": 3e-4, "J": 8e-7}]}


def _two_span():
    s = Session()
    s.dispatch("define_materials_and_sections", _MS_ZIF)
    s.dispatch("generate_frame", {"spans": [6, 6], "storeys": [4], "column_section": "COLUMN",
                                  "beam_section": "BEAM", "material": "STEEL", "base": "fixed"})
    return s


def test_load_on_a_fixed_support_is_diagnosed_as_such():
    s = _two_span()
    s.dispatch("set_load_cases", {"cases": [{"name": "P", "nodal_loads": [
        {"node": 1, "load": [0, 0, -50e3, 0, 0, 0]}]}]})
    r = s.dispatch("solve_model", {})
    assert not r.ok
    assert r.payload["unusable"]["checks"] == ["zero_internal_force"]
    hint = " ".join(r.payload["unusable"]["next"])
    assert "支座" in hint and "节点 1 的 uz" in hint
    assert "机构" not in hint                     # 不再把错因报成机构


def test_load_partly_on_a_free_direction_is_not_blamed_on_the_support():
    """柱底铰接放开了转动，力矩加在 ry 上就会进杆件——这不是「荷载在支座上」。"""
    s = _two_span()
    s.dispatch("edit_node", {"node_id": 1, "fix": [1, 1, 1, 0, 0, 0]})
    s.dispatch("set_load_cases", {"cases": [{"name": "P", "nodal_loads": [
        {"node": 1, "load": [0, 0, -50e3, 0, 10e3, 0]}]}]})
    r = s.dispatch("solve_model", {})
    names = [d["id"] for d in r.payload["silent_failures"]["summary"].get("critical_details", [])]
    assert "zero_internal_force" not in names


def test_diagnosis_helper_rejects_cases_with_member_loads():
    s = _two_span()
    s.dispatch("set_load_cases", {"cases": [{"name": "P",
        "nodal_loads": [{"node": 1, "load": [0, 0, -50e3, 0, 0, 0]}],
        "member_loads": [{"member": 4, "w": [0, 0, -10e3]}]}]})
    s.dispatch("solve_model", {})
    assert sf._loads_sitting_on_supports(s.frame, "P") is None
