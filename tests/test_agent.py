"""Agent 层验证。

全部用 ScriptedProvider 离线跑——不需要密钥、不依赖网络，因此可以进 CI，
演示翻车时也能靠它回放。真实后端只在 DeepSeekProvider 里，接口完全相同。
"""
from copy import deepcopy
from pathlib import Path

import pytest

from agent import Session, ScriptedProvider, TOOLS, run_turn

MATERIALS = [{"name": "STEEL", "E": 2.1e11, "nu": 0.3}]
SECTIONS = [
    {"name": "COLUMN", "A": 0.012, "Iy": 8.0e-5, "Iz": 2.4e-4, "J": 1.0e-6},
    {"name": "BEAM", "A": 0.010, "Iy": 4.0e-5, "Iz": 3.0e-4, "J": 8.0e-7},
]


def call(cid, name, **args):
    return {"tool_calls": [{"id": cid, "name": name, "arguments": args}]}


def text(content):
    return {"content": content}


# --------------------------------------------------------------- 工具本身

def test_tool_schemas_are_well_formed():
    names = [t["function"]["name"] for t in TOOLS]
    assert len(names) == len(set(names))
    for tool in TOOLS:
        fn = tool["function"]
        assert fn["description"], f"{fn['name']} 缺少描述"
        assert fn["parameters"]["type"] == "object"


def test_generate_frame_expands_topology():
    """模型给参数，代码展开拓扑——三跨两层单开间应当是 24 节点 36 杆。"""
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    r = s.generate_frame(spans=[6, 6, 6], storeys=[3.6, 3.6], bays=[6], beam_load=20e3)
    assert r.ok
    assert r.payload["summary"]["nodes"] == 24
    assert r.payload["summary"]["members"] == 36


def test_generate_frame_rejects_bad_parameters():
    s = Session()
    r = s.generate_frame(spans=[6, -1], storeys=[3.6])
    assert not r.ok and "大于 0" in r.payload["error"]


def test_solve_refuses_a_structurally_invalid_model():
    """算前守门第一级：JSON Schema 查结构，错误里带着出问题的字段路径。"""
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    s.generate_frame(spans=[6], storeys=[3.6])
    s.model["supports"] = []
    r = s.solve_model()
    assert not r.ok
    assert any(e.startswith("[结构]") and "supports" in e for e in r.payload["errors"])
    assert s.solution is None


def test_solve_refuses_a_semantically_invalid_model():
    """第二级：结构合法但语义有问题时，给的是中文提示，可直接回喂给模型。"""
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    s.generate_frame(spans=[6], storeys=[3.6])
    s.model["nodes"].append({"id": 999, "x": 99.0, "y": 0.0, "z": 0.0})
    r = s.solve_model()
    assert not r.ok
    assert any("[语义]" in e and "悬空节点" in e for e in r.payload["errors"])
    assert s.solution is None


def test_solve_reports_equilibrium_per_case():
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    s.generate_frame(spans=[6, 6], storeys=[3.6], bays=[6], beam_load=20e3)
    r = s.solve_model()
    assert r.ok
    for case in r.payload["cases"].values():
        assert case["equilibrium_ok"]


def test_query_before_solve_is_refused():
    s = Session()
    r = s.query_results(what="max_displacement")
    assert not r.ok and "solve_model" in r.payload["error"]


def test_diagnose_locates_missing_restraint():
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    s.generate_frame(spans=[6], storeys=[3.6])
    for support in s.model["supports"]:
        support["fix"] = [1, 0, 1, 1, 1, 1]        # 抽掉 Y 向约束
    r = s.diagnose_supports()
    assert r.ok and r.payload["modes"]
    assert any(p["direction"] == "沿 Y 平动"
               for m in r.payload["modes"] for p in m["participants"])


def test_unknown_tool_is_reported_not_raised():
    assert not Session().dispatch("fly_to_moon", {}).ok


# --------------------------------------------------------------- 对话循环

def test_happy_path_generate_then_solve():
    provider = ScriptedProvider([
        call("1", "define_materials_and_sections", materials=MATERIALS, sections=SECTIONS),
        call("2", "generate_frame", spans=[6, 6, 6], storeys=[3.6, 3.6], bays=[6],
             beam_load=20000, base="fixed"),
        call("3", "solve_model"),
        call("4", "query_results", what="max_displacement"),
        text("三跨两层框架已建好，最大位移出现在顶层角节点。"),
    ])
    out = run_turn("三跨 6 米、两层层高 3.6 米、开间 6 米的框架，梁上 20 kN/m，柱底固接",
                   provider)
    assert not out.stopped_by_limit
    assert [name for name, _ in out.tool_calls] == [
        "define_materials_and_sections", "generate_frame", "solve_model", "query_results"]
    assert out.session.solution is not None


def test_self_repair_loop_after_validation_failure():
    """自修复闭环的一轮：第一次模型不合法 -> 拿到中文错误 -> 改对 -> 算成。"""
    broken = {
        "units": "N-m-Pa", "materials": MATERIALS, "sections": SECTIONS,
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0}, {"id": 2, "x": 6, "y": 0, "z": 0}],
        "members": [{"id": 1, "i": 1, "j": 2, "section": "BEAM", "material": "STEEL"}],
        "supports": [{"node": 1, "fix": [1, 1, 1, 1, 1, 1]}],
        "combos": [{"name": "C1", "factors": {"SNOW": 1.4}}],     # 引用了不存在的工况
    }
    fixed = {**broken, "combos": []}
    fixed["nodal_loads"] = [{"node": 2, "load": [0, 0, -10000, 0, 0, 0]}]

    provider = ScriptedProvider([
        call("1", "set_model", model=broken),
        call("2", "solve_model"),
        call("3", "set_model", model=fixed),
        call("4", "solve_model"),
        text("第一次组合引用了未定义的工况，已改正并求解完成。"),
    ])
    out = run_turn("算一根悬臂梁", provider)
    assert not out.stopped_by_limit
    assert out.session.solution is not None
    # 第一次 solve 的失败信息确实被喂回给了模型
    fed_back = [m for msgs in provider.seen for m in msgs if m.get("role") == "tool"]
    assert any("SNOW" in m["content"] for m in fed_back)


def test_tool_results_are_json_and_carry_the_numbers():
    """模型看到的数字只能来自工具返回的 JSON，这里确认它确实带着数。"""
    provider = ScriptedProvider([
        call("1", "define_materials_and_sections", materials=MATERIALS, sections=SECTIONS),
        call("2", "generate_frame", spans=[6], storeys=[3.6], beam_load=20000),
        call("3", "solve_model"),
        text("完成。"),
    ])
    out = run_turn("单跨单层", provider)
    tool_messages = [m for m in provider.seen[-1] if m.get("role") == "tool"]
    assert any("max_displacement_mm" in m["content"] for m in tool_messages)
    assert out.session.solution is not None


def test_round_limit_stops_a_runaway_loop():
    provider = ScriptedProvider([call(str(k), "validate_model") for k in range(10)])
    out = run_turn("死循环", provider, max_rounds=3)
    assert out.stopped_by_limit
    assert out.rounds == 3
    assert len(out.tool_calls) == 3


def test_session_state_survives_across_rounds():
    s = Session()
    provider = ScriptedProvider([
        call("1", "define_materials_and_sections", materials=MATERIALS, sections=SECTIONS),
        text("材料已定义。"),
    ])
    run_turn("先定材料", provider, session=s)
    provider2 = ScriptedProvider([
        call("2", "generate_frame", spans=[6], storeys=[3.6]),
        text("好了。"),
    ])
    run_turn("再建模型", provider2, session=s)
    assert s.model["materials"][0]["name"] == "STEEL"
    assert s.model["summary"] if False else len(s.model["nodes"]) == 4


# --------------------------------------------------------------- 实测暴露的回归

def test_set_model_carries_forward_materials_and_sections():
    """真实模型踩过的坑：set_model 曾整体替换，把已定义的材料截面冲掉，

    于是它被 "materials is required" 反复打回，四轮都没绕出来。
    现在与 generate_frame 行为一致：缺就自动带上。
    """
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    r = s.set_model({
        "units": "N-m-Pa",
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0}, {"id": 2, "x": 6, "y": 0, "z": 0}],
        "members": [{"id": 1, "i": 1, "j": 2, "section": "BEAM", "material": "STEEL"}],
        "supports": [{"node": 1, "fix": [1, 1, 1, 1, 1, 1]}],
    })
    assert r.ok
    assert s.model["materials"] == MATERIALS
    assert s.validate_model().ok


def test_simply_supported_via_set_model_matches_analytic():
    """整条链路对着解析解验一次：跨中集中力下 PL³/48EI。"""
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    L, P = 8.0, 50e3
    s.set_model({
        "units": "N-m-Pa",
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0}, {"id": 2, "x": L / 2, "y": 0, "z": 0},
                  {"id": 3, "x": L, "y": 0, "z": 0}],
        "members": [{"id": 1, "i": 1, "j": 2, "section": "BEAM", "material": "STEEL"},
                    {"id": 2, "i": 2, "j": 3, "section": "BEAM", "material": "STEEL"}],
        "supports": [{"node": 1, "fix": [1, 1, 1, 1, 0, 0]},
                     {"node": 3, "fix": [0, 1, 1, 1, 0, 0]}],
        "nodal_loads": [{"node": 2, "load": [0, 0, -P, 0, 0, 0]}],
    })
    assert s.solve_model().ok
    got = s.query_results(what="max_displacement").payload["magnitude_mm"]
    E, Iz = MATERIALS[0]["E"], SECTIONS[1]["Iz"]
    # query_results 对外只给到小数点后 6 位，容差按显示精度取
    assert got == pytest.approx(P * L**3 / (48 * E * Iz) * 1000, rel=1e-6)


def test_set_load_cases_builds_cases_and_combos():
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    gen = s.generate_frame(spans=[7.5, 7.5], storeys=[4.0], beam_release=True)
    beams = gen.payload["beam_member_ids"]
    assert beams, "generate_frame 应当回传梁的编号，否则模型没法分工况加载"
    r = s.set_load_cases(
        cases=[{"name": "D", "member_loads": [{"member": m, "w": [0, 0, -15e3]} for m in beams]},
               {"name": "L", "member_loads": [{"member": m, "w": [0, 0, -9e3]} for m in beams]}],
        combos=[{"name": "1.3D+1.5L", "factors": {"D": 1.3, "L": 1.5}}])
    assert r.ok and r.payload["cases"] == ["D", "L"]
    out = s.solve_model()
    assert out.ok and set(out.payload["cases"]) == {"D", "L", "1.3D+1.5L"}


def test_combo_vertical_total_matches_hand_calculation():
    """两跨 7.5 m、1.3×15 + 1.5×9 kN/m -> 竖向反力合计 495 kN，可手算复核。"""
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    beams = s.generate_frame(spans=[7.5, 7.5], storeys=[4.0],
                             beam_release=True).payload["beam_member_ids"]
    s.set_load_cases(
        cases=[{"name": "D", "member_loads": [{"member": m, "w": [0, 0, -15e3]} for m in beams]},
               {"name": "L", "member_loads": [{"member": m, "w": [0, 0, -9e3]} for m in beams]}],
        combos=[{"name": "C", "factors": {"D": 1.3, "L": 1.5}}])
    s.solve_model()
    total = s.query_results(what="reactions", case="C").payload["vertical_total_kN"]
    assert total == pytest.approx((1.3 * 15 + 1.5 * 9) * 15, rel=1e-9)


def test_query_results_carry_coordinates_not_just_node_ids():
    """模型曾凭节点编号瞎猜位置（说成"梁跨中"），所以结果必须带坐标。"""
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    s.generate_frame(spans=[6, 6], storeys=[3.6], beam_load=20e3)
    s.solve_model()
    disp = s.query_results(what="max_displacement").payload
    assert len(disp["coordinates_xyz"]) == 3
    reac = s.query_results(what="reactions").payload
    assert all("xyz" in v and "R" in v for v in reac["reactions"].values())


def test_member_forces_without_id_returns_every_member():
    """模型曾一根一根查，浪费五轮。不给编号就一次给全。"""
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    s.generate_frame(spans=[6, 6], storeys=[3.6], beam_load=20e3)
    s.solve_model()
    rows = s.query_results(what="member_forces").payload["all_members"]
    assert len(rows) == len(s.frame.members)


def test_set_model_rejects_an_incomplete_model_immediately():
    """真实模型踩过的第二个坑：set_model 只存不验，返回"成功"，

    错误要到下一个工具才炸，它就以为是下一个工具的参数写错了，白转好几轮。
    成功/失败信号必须归属正确的工具。
    """
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    r = s.set_model({
        "units": "N-m-Pa",
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0}, {"id": 2, "x": 6, "y": 0, "z": 0}],
        "members": [{"id": 1, "i": 1, "j": 2, "section": "BEAM"}],   # 缺 material，也没 supports
    })
    assert not r.ok
    assert any("supports" in e for e in r.payload["errors"])
    assert any("material" in e for e in r.payload["errors"])


def three_d_simply_supported(fix_i, fix_j):
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    s.set_model({
        "units": "N-m-Pa",
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0}, {"id": 2, "x": 4, "y": 0, "z": 0},
                  {"id": 3, "x": 8, "y": 0, "z": 0}],
        "members": [{"id": 1, "i": 1, "j": 2, "section": "BEAM", "material": "STEEL"},
                    {"id": 2, "i": 2, "j": 3, "section": "BEAM", "material": "STEEL"}],
        "supports": [{"node": 1, "fix": list(fix_i)}, {"node": 3, "fix": list(fix_j)}],
        "nodal_loads": [{"node": 2, "load": [0, 0, -50e3, 0, 0, 0]}],
    })
    return s


def test_solve_failure_carries_the_diagnosis_in_the_same_call():
    """求解失败时直接附上诊断，模型不必再调一次 diagnose_supports。"""
    s = three_d_simply_supported([1, 1, 1, 0, 0, 0], [0, 1, 1, 0, 0, 0])
    r = s.solve_model()
    assert not r.ok
    assert r.payload["diagnosis"], "失败结果里应当直接带上刚体模态"
    assert "rx" in r.payload["hint"], "提示应当点名三维梁的扭转机构"
    assert any(p["direction"] == "绕 X 转动"
               for m in r.payload["diagnosis"] for p in m["participants"])


def test_constraining_one_end_torsion_makes_it_solvable():
    """约束住一端的 rx 之后，同一个三维简支梁就能算了。"""
    s = three_d_simply_supported([1, 1, 1, 1, 0, 0], [0, 1, 1, 0, 0, 0])
    assert s.solve_model().ok


def test_implausible_displacement_is_flagged_with_the_likely_cause():
    """荷载方向写成全局 +Y 时位移会离谱，量级检查要当场标出来并指明常见原因。

    这里必须用三维模型（给了 bays）：平面刚架的面外自由度是被自动约束的，
    面外荷载会被吃掉而不是产生离谱位移，那是另一条检查管的事。
    """
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    beams = s.generate_frame(spans=[7.5, 7.5], storeys=[4.0], bays=[6.0],
                             beam_release=True).payload["beam_member_ids"]
    s.set_load_cases(cases=[{"name": "D",
                             "member_loads": [{"member": m, "w": [0, 15e3, 0]} for m in beams]}])
    case = s.solve_model().payload["cases"]["D"]
    assert "warning" in case
    assert "方向" in case["warning"]


def test_out_of_plane_load_on_a_plane_frame_is_flagged_as_zero_displacement():
    """平面刚架的面外自由度被自动约束，面外荷载会被完全吃掉。

    位移为零而荷载不为零，是一类容易被当成"算对了"的静默错误，必须点破。
    """
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    beams = s.generate_frame(spans=[7.5, 7.5], storeys=[4.0],
                             beam_release=True).payload["beam_member_ids"]
    s.set_load_cases(cases=[{"name": "D",
                             "member_loads": [{"member": m, "w": [0, 15e3, 0]} for m in beams]}])
    case = s.solve_model().payload["cases"]["D"]
    assert case["max_displacement_mm"] == 0.0
    assert "warning" in case and "位移为零" in case["warning"]


def simply_supported_beam(load):
    """单跨简支梁，节点只有两端——跨中挠度**不在任何节点上**。

    这正是"节点最大位移"会骗人的最小模型：model_compiler 只在集中力位置和
    显式内节点处剖分，满跨均布/梯形不触发剖分，于是跨中没有节点可查。
    """
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    s.set_model({
        "units": "N-m-Pa",
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                  {"id": 2, "x": 6, "y": 0, "z": 0}],
        "members": [{"id": 1, "i": 1, "j": 2, "section": "BEAM", "material": "STEEL"}],
        "supports": [{"node": 1, "fix": [1, 1, 1, 1, 0, 0]},
                     {"node": 2, "fix": [0, 1, 1, 1, 0, 0]}],
    })
    load(s)
    return s


def test_headline_displacement_includes_deflection_inside_the_element():
    """简支梁 6 m、20 kN/m：节点位移是 0，真实跨中挠度 16.38 mm。

    以前 solve_model 的头条只报节点那一个数，于是"施加了 20 kN/m"和
    "最大位移 0.0 mm"同时出现在一份结果里。内力早就做了单元内解析恢复
    （internal_forces.member_deflection），位移这一路当时没接上来。
    """
    s = simply_supported_beam(lambda s: s.set_member_load(1, [0, 0, -20e3]))
    case = s.solve_model().payload["cases"]["Load-1"]
    assert case["max_displacement_mm"] == 0.0, "两端都被约束，节点位移本来就是 0"

    E, Iz, L, w = 2.1e11, 3.0e-4, 6.0, 20e3
    exact = 5 * w * L ** 4 / (384 * E * Iz) * 1000.0
    assert case["max_deflection_mm"] == pytest.approx(exact, rel=1e-3)
    assert case["at_x_m"] == pytest.approx(L / 2, abs=1e-6)


def test_zero_nodal_displacement_is_explained_not_blamed_on_the_load():
    """节点位移为零**不等于**荷载加错了，提示不能反过来诬告用户。

    原来这两种情形共用一条警告"检查荷载方向"：
    简支梁均布（正常，响应在单元内）与平面刚架面外荷载（真错）。
    照着它去查荷载方向，前一种永远查不出问题。
    """
    s = simply_supported_beam(lambda s: s.set_member_load(1, [0, 0, -20e3]))
    case = s.solve_model().payload["cases"]["Load-1"]
    assert "warning" not in case, "正常的简支梁不该报警告"
    assert "不是荷载加错了" in case["note"]
    assert "max_deflection_mm" in case["note"]


def test_span_only_loads_count_as_applied_load():
    """只用 member_spans 加载时，"有荷载但位移为零"那一支不能失灵。

    _applied_load_magnitude 原先只统计 nodal_loads 与 member_loads，
    梯形/跨中集中力一概不算——于是纯跨荷载的模型算出 0，那条判据进不去，
    结果是反力 60 kN、位移 0、**一句提示都没有**。
    silent_failures._total_applied_load 早因同一个原因修过，没传播到这里。
    """
    s = simply_supported_beam(
        lambda s: s.set_member_span_load(1, "trapezoid", [0, 0, 0], [0, 0, -20e3]))
    case = s.solve_model().payload["cases"]["Load-1"]
    # 编译后才有 frame；这个数就是那条判据的输入
    assert s._applied_load_magnitude("Load-1") > 0.0
    assert case["max_displacement_mm"] == 0.0
    assert case["max_deflection_mm"] > 1.0, "三角形荷载下跨内确实有挠度"
    assert "note" in case, "以前这里什么都不说"


def test_frame_headline_deflection_beats_the_nodal_one():
    """多层框架上两个数差一倍——这不是极端算例，是默认工况。"""
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    beams = s.generate_frame(spans=[6.0, 6.0], storeys=[4.0],
                             bays=[6.0]).payload["beam_member_ids"]
    s.set_load_cases(cases=[{"name": "D", "member_loads":
                             [{"member": m, "w": [0, 0, -20e3]} for m in beams]}])
    case = s.solve_model().payload["cases"]["D"]
    assert case["max_deflection_mm"] > 1.5 * case["max_displacement_mm"]
    assert case["at_member"] in beams


def test_no_load_at_all_raises_no_zero_displacement_warning():
    """真的没加荷载时不该报这条——那不是错误。"""
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    s.generate_frame(spans=[6.0], storeys=[3.6], bays=[6.0])
    case = s.solve_model().payload["cases"]["default"]
    assert case["max_displacement_mm"] == 0.0
    assert "warning" not in case


def test_correct_load_direction_raises_no_warning():
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    beams = s.generate_frame(spans=[7.5, 7.5], storeys=[4.0], bays=[6.0],
                             beam_release=True).payload["beam_member_ids"]
    s.set_load_cases(cases=[{"name": "D",
                             "member_loads": [{"member": m, "w": [0, 0, -15e3]} for m in beams]}])
    case = s.solve_model().payload["cases"]["D"]
    assert "warning" not in case


# ------------------------- 门式刚架与增量原语 -------------------------

def portal(**kw):
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    r = s.generate_portal_frame(column_section="COLUMN", rafter_section="BEAM",
                                material="STEEL", **kw)
    return s, r


def test_portal_frame_builds_a_pitched_roof():
    """跨中起脊：屋脊节点高于檐口节点，柱只立在非脊站位。"""
    s, r = portal(spans=[24.0], eave_height=6.0, ridge_rise=2.0)
    assert r.ok, r.payload
    zs = sorted({n["z"] for n in s.model["nodes"]})
    assert zs == [0.0, 6.0, 8.0]
    assert r.payload["summary"]["nodes"] == 5      # 两柱脚 + 两檐口 + 一屋脊
    assert r.payload["summary"]["members"] == 4    # 两柱 + 两斜梁


def test_portal_frame_leaves_no_dangling_node_at_the_ridge_station():
    """屋脊站位不立柱，那里不该生成柱脚节点——早期版本在这里留下了悬空节点。"""
    s, _ = portal(spans=[24.0], eave_height=6.0, ridge_rise=2.0)
    assert s.validate_model().ok, s.validate_model().payload


def test_plane_portal_frame_is_solvable_with_pinned_bases():
    """一榀平面刚架放进三维求解器时面外无约束，柱底铰接必然成为机构。

    生成器自动补面外约束（uy、rx、rz），二维程序只是把这件事隐含掉了。
    """
    s, r = portal(spans=[24.0], eave_height=6.0, ridge_rise=2.0,
                  base="pinned", rafter_load=12e3)
    assert s.solve_model().ok


def test_plane_portal_frame_is_a_mechanism_without_plane_restraints():
    s, _ = portal(spans=[24.0], eave_height=6.0, ridge_rise=2.0, base="pinned",
                  rafter_load=12e3, plane_restraints=False)
    out = s.solve_model()
    assert not out.ok and out.payload.get("diagnosis")


def test_portal_frame_reactions_match_the_rafter_load():
    """竖向反力合计 = 斜梁总长 × 线荷载。斜梁是斜的，长度不等于跨度。"""
    import numpy as np
    s, r = portal(spans=[24.0], eave_height=6.0, ridge_rise=2.0,
                  bays=[6.0, 6.0], rafter_load=12e3)
    assert s.solve_model().ok
    rafters = set(r.payload["rafter_member_ids"])
    length = sum(float(np.linalg.norm(s.frame.nodes[m.j].xyz - s.frame.nodes[m.i].xyz))
                 for m in s.frame.members.values() if m.id in rafters)
    total = s.query_results(what="reactions").payload["vertical_total_kN"]
    assert total == pytest.approx(length * 12e3 / 1e3, rel=1e-9)


def test_rafter_ids_exclude_the_longitudinal_ties():
    """系杆默认与斜梁同截面。按名字筛会把系杆一起返回，拿去加载就压错了构件。"""
    s, r = portal(spans=[24.0], eave_height=6.0, ridge_rise=2.0, bays=[6.0, 6.0])
    rafters = set(r.payload["rafter_member_ids"])
    ties = [m for m in s.model["members"]
            if m["section"] == "BEAM" and int(m["id"]) not in rafters]
    assert ties, "多开间时应当有纵向系杆"
    nodes = {int(n["id"]): n for n in s.model["nodes"]}
    for t in ties:                       # 系杆沿 Y，两端 x 相同
        assert nodes[t["i"]]["x"] == pytest.approx(nodes[t["j"]]["x"])


def test_portal_frame_rejects_a_flat_roof():
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    r = s.generate_portal_frame(spans=[24.0], eave_height=6.0, ridge_rise=0.0)
    assert not r.ok and "generate_frame" in r.payload["error"]


def test_add_members_can_brace_a_generated_frame():
    """规整框架加一道斜撑——生成器盖不住的局部改动交给增量原语。"""
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    s.generate_frame(spans=[6.0], storeys=[3.6, 3.6], column_section="COLUMN",
                     beam_section="BEAM", material="STEEL")
    before = len(s.model["members"])
    r = s.add_members(pairs=[[1, 4]], section="BEAM")
    assert r.ok, r.payload
    assert len(s.model["members"]) == before + 1
    assert r.payload["added_member_ids"] == [before + 1]


def test_add_members_rejects_unknown_nodes():
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    s.generate_frame(spans=[6.0], storeys=[3.6])
    r = s.add_members(pairs=[[1, 999]], section="BEAM")
    assert not r.ok and "999" in r.payload["error"]


def test_failed_member_batch_is_atomic():
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    s.generate_frame(spans=[6.0], storeys=[3.6])
    before = list(s.model["members"])
    r = s.add_members(pairs=[[1, 4], [1, 999]], section="BEAM")
    assert not r.ok
    assert s.model["members"] == before


def test_duplicate_member_is_rejected_without_changing_the_model():
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    s.generate_frame(spans=[6.0], storeys=[3.6])
    first = s.model["members"][0]
    before = len(s.model["members"])
    r = s.add_members(pairs=[[first["j"], first["i"]]], section="BEAM")
    assert not r.ok and "已经有杆件" in r.payload["error"]
    assert len(s.model["members"]) == before


def test_failed_node_batch_does_not_add_partial_coordinates():
    s = Session()
    r = s.add_nodes(coordinates=[[0.0, 0.0, 0.0], [1.0, 2.0]])
    assert not r.ok
    assert s.model == {}


def test_remove_node_cleans_every_reference_and_records_one_step():
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    made = s.generate_frame(
        spans=[6.0], storeys=[3.6], column_section="COLUMN",
        beam_section="BEAM", material="STEEL")
    tip = s.add_nodes([[8.0, 0.0, 3.6]]).payload["added_node_ids"][0]
    top = max(n["id"] for n in s.model["nodes"] if n["z"] == 3.6 and n["id"] != tip)
    member = s.add_members([[top, tip]], "BEAM", "STEEL").payload["added_member_ids"][0]
    s.set_load_cases(cases=[{
        "name": "D",
        "nodal_loads": [{"node": tip, "load": [0, 0, -1000, 0, 0, 0]}],
        "member_loads": [{"member": member, "w": [0, 0, -500]}],
    }])
    s.define_set("EDIT", node_ids=[1, tip], member_ids=[made.payload["beam_member_ids"][0], member])
    before_steps = len(s.history.timeline())

    result = s.remove_nodes([tip])
    assert result.ok, result.payload
    assert len(s.history.timeline()) == before_steps + 1
    case = s.model["load_cases"][0]
    assert case.get("nodal_loads") == []
    assert case.get("member_loads") == []
    assert tip not in s.model["sets"]["EDIT"]["nodes"]
    assert member not in s.model["sets"]["EDIT"]["members"]
    assert s.validate_model().ok


def test_delete_all_geometry_is_a_valid_editing_state_but_not_analysis_ready():
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    s.generate_frame(spans=[6.0], storeys=[3.6])
    result = s.remove_nodes([n["id"] for n in s.model["nodes"]])
    assert result.ok
    assert not result.payload["analysis_ready"]
    assert s.model["nodes"] == [] and s.model["members"] == []


def test_add_nodes_snaps_repeated_coordinates_and_returns_input_mapping():
    s = Session()
    result = s.add_nodes([[0, 0, 0], [0, 0, 0], [6, 0, 0]])
    assert result.ok
    assert result.payload["input_node_ids"] == [1, 1, 2]
    assert result.payload["reused_node_ids"] == [1]
    assert len(s.model["nodes"]) == 2


def test_adding_an_existing_coordinate_is_a_true_no_op():
    """精确建点输到已有节点时，不应清结果，也不应污染撤销历史。"""
    from copy import deepcopy

    s = Session()
    assert s.add_nodes([[1.0, 2.0, 3.0]]).ok
    sentinels = tuple(object() for _ in range(4))
    s.frame, s.solution, s.compilation, s.result_db = sentinels
    before_model = deepcopy(s.model)
    before_history = len(s.history.steps)

    result = s.add_nodes([[1.0, 2.0, 3.0]])

    assert result.ok and result.payload["no_change"] is True
    assert result.payload["reused_node_ids"] == [1]
    assert result.payload["input_node_ids"] == [1]
    assert s.model == before_model
    assert (s.frame, s.solution, s.compilation, s.result_db) == sentinels
    assert len(s.history.steps) == before_history


def test_set_model_failure_is_atomic():
    from copy import deepcopy

    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    before = deepcopy(s.model)
    result = s.set_model({"nodes": [{"id": 1, "x": 0, "y": 0, "z": 0}]})
    assert not result.ok
    assert s.model == before


def test_complete_model_rejects_duplicate_ids_before_dict_assembly_hides_them():
    s = Session()
    result = s.set_model({
        "units": "N-m-Pa", "materials": MATERIALS, "sections": SECTIONS,
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                  {"id": 1, "x": 6, "y": 0, "z": 0}],
        "members": [{"id": 1, "i": 1, "j": 1,
                     "section": "BEAM", "material": "STEEL"}],
        "supports": [{"node": 1, "fix": [1] * 6}],
    })
    assert not result.ok
    assert any("节点编号重复" in error for error in result.payload["errors"])


def test_complete_model_rejects_coincident_nodes_with_different_ids():
    s = Session()
    result = s.set_model({
        "units": "N-m-Pa", "materials": MATERIALS, "sections": SECTIONS,
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                  {"id": 2, "x": 0, "y": 0, "z": 0},
                  {"id": 3, "x": 6, "y": 0, "z": 0}],
        "members": [{"id": 1, "i": 1, "j": 3,
                     "section": "BEAM", "material": "STEEL"}],
        "supports": [{"node": 1, "fix": [1] * 6}],
    })
    assert not result.ok
    assert any("坐标重合" in error for error in result.payload["errors"])


def test_multiple_named_forces_on_one_target_are_added_not_silently_overwritten():
    from model_io import from_dict

    model = {
        "units": "N-m-Pa", "materials": MATERIALS, "sections": SECTIONS,
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                  {"id": 2, "x": 6, "y": 0, "z": 0}],
        "members": [{"id": 1, "i": 1, "j": 2,
                     "section": "BEAM", "material": "STEEL"}],
        "supports": [{"node": 1, "fix": [1] * 6}],
        "load_cases": [{"name": "L",
                        "nodal_loads": [
                            {"name": "P1", "node": 2,
                             "load": [0, 0, -10, 0, 0, 0]},
                            {"name": "P2", "node": 2,
                             "load": [0, 0, -20, 0, 0, 0]}],
                        "member_loads": [
                            {"name": "q1", "member": 1, "w": [0, 0, -2]},
                            {"name": "q2", "member": 1, "w": [0, 0, -3]}]}],
    }
    frame = from_dict(model)
    assert frame.load_cases["L"].nodal_loads[2][2] == -30
    assert frame.load_cases["L"].member_loads[1][2] == -5


def test_incremental_named_loads_can_coexist_on_one_target_and_edit_independently():
    s = Session()
    s.add_nodes([[0, 0, 0], [6, 0, 0]])
    s.add_members([[1, 2]])
    assert s.set_nodal_load(2, [0, 0, -10, 0, 0, 0], name="P1").ok
    assert s.set_nodal_load(2, [0, 0, -20, 0, 0, 0], name="P2").ok
    case = s.model["load_cases"][0]
    assert [entry["name"] for entry in case["nodal_loads"]] == ["P1", "P2"]
    assert s.set_nodal_load(2, [0, 0, -30, 0, 0, 0], name="P1").ok
    values = {entry["name"]: entry["load"][2] for entry in
              s.model["load_cases"][0]["nodal_loads"]}
    assert values == {"P1": -30.0, "P2": -20.0}
    unnamed = s.set_nodal_load(2, [0, 0, -1, 0, 0, 0])
    assert not unnamed.ok and "载荷名称" in unnamed.payload["error"]


def test_property_update_can_leave_a_visible_draft_when_a_definition_is_removed():
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    s.generate_frame(
        spans=[6.0], storeys=[3.6], column_section="COLUMN",
        beam_section="BEAM", material="STEEL")
    result = s.define_materials_and_sections(MATERIALS, [SECTIONS[0]])

    assert result.ok
    assert not result.payload["analysis_ready"]
    assert any("BEAM" in warning for warning in result.payload["warnings"])


def test_property_stage_allows_material_then_section_and_both_are_undoable():
    s = Session()

    material = s.define_materials_and_sections(MATERIALS, [])
    assert material.ok, material.payload
    assert s.model["materials"] == MATERIALS
    assert s.model["sections"] == []
    assert len(s.history.timeline()) == 1

    section = s.define_materials_and_sections(MATERIALS, SECTIONS)
    assert section.ok, section.payload
    assert len(s.history.timeline()) == 2
    assert s.undo()
    assert s.model["sections"] == []


def test_duplicate_property_names_are_rejected_without_mutation():
    s = Session()
    before = deepcopy(s.model)
    duplicate = [dict(MATERIALS[0]), dict(MATERIALS[0])]

    result = s.define_materials_and_sections(duplicate, SECTIONS)

    assert not result.ok
    assert s.model == before


def test_parameter_generation_can_create_geometry_before_property_assignment():
    s = Session()
    draft = s.generate_frame(spans=[6.0], storeys=[3.6])
    assert draft.ok
    assert not draft.payload["analysis_ready"]
    assert s.model["nodes"] and s.model["members"]
    assert {member["section"] for member in s.model["members"]} == {""}

    s.define_materials_and_sections(MATERIALS, SECTIONS)
    assigned = s.assign_properties(
        [member["id"] for member in s.model["members"]], "BEAM", "STEEL")
    assert assigned.ok
    assert assigned.payload["analysis_ready"]


def test_add_nodes_then_members_builds_a_cantilever():
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    s.generate_frame(spans=[6.0], storeys=[3.6], column_section="COLUMN",
                     beam_section="BEAM", material="STEEL")
    added = s.add_nodes(coordinates=[[8.0, 0.0, 3.6]])
    assert added.ok
    tip = added.payload["added_node_ids"][0]
    top_right = next(n["id"] for n in s.model["nodes"]
                     if n["x"] == 6.0 and n["z"] == 3.6)
    assert s.add_members(pairs=[[top_right, tip]], section="BEAM").ok
    assert s.validate_model().ok


def test_incremental_bc_and_load_tools_follow_the_model_schema_and_are_undoable():
    s = Session()
    assert s.add_nodes([[0, 0, 0], [6, 0, 0]]).ok
    assert s.add_members([[1, 2]]).ok
    assert s.set_supports([1], [1, 1, 1, 1, 1, 1], name="BC-Fixed").ok
    assert s.add_load_case("Service").ok
    before_load = len(s.history.timeline())

    nodal = s.set_nodal_load(2, [0, 0, -1000, 0, 0, 0], "Service")
    member = s.set_member_load(1, [0, 0, -500], "Service")
    assert nodal.ok and member.ok
    case = s.model["load_cases"][0]
    assert case["nodal_loads"] == [
        {"name": "CF-Node-2", "node": 2,
         "load": [0.0, 0.0, -1000.0, 0.0, 0.0, 0.0]}]
    assert case["member_loads"] == [
        {"name": "Line-Member-1", "member": 1, "w": [0.0, 0.0, -500.0]}]
    assert len(s.history.timeline()) == before_load + 2
    assert s.undo()
    assert s.model["load_cases"][0]["member_loads"] == []


def test_incremental_load_replaces_instead_of_duplicating_and_zero_deletes():
    s = Session()
    s.add_nodes([[0, 0, 0], [6, 0, 0]])
    s.add_members([[1, 2]])

    assert s.set_nodal_load(2, [1, 0, 0, 0, 0, 0]).ok
    assert s.set_nodal_load(2, [2, 0, 0, 0, 0, 0]).ok
    case = s.model["load_cases"][0]
    assert len(case["nodal_loads"]) == 1
    assert case["nodal_loads"][0]["load"][0] == 2.0
    assert s.set_nodal_load(2, [0, 0, 0, 0, 0, 0]).ok
    assert case is not s.model["load_cases"][0]  # 每次走候选副本，不原位污染
    assert s.model["load_cases"][0]["nodal_loads"] == []


def test_named_span_load_replaces_by_name_and_zero_deletes():
    s = Session()
    s.add_nodes([[0, 0, 0], [6, 0, 0]])
    s.add_members([[1, 2]])
    assert s.set_member_span_load(1, "point", [0, 0, -10], a=2,
                                  name="P-1").ok
    assert s.set_member_span_load(1, "point", [0, 0, -20], a=4,
                                  name="P-1").ok
    entries = s.model["load_cases"][0]["member_spans"]
    assert len(entries) == 1 and entries[0]["a"] == 4
    assert s.set_member_span_load(1, "point", [0, 0, 0], a=4,
                                  name="P-1").ok
    assert s.model["load_cases"][0]["member_spans"] == []


def test_changing_a_named_load_type_does_not_leave_the_old_load_behind():
    s = Session()
    s.add_nodes([[0, 0, 0], [6, 0, 0]])
    s.add_members([[1, 2]])
    assert s.set_member_load(1, [0, 0, -10], name="Roof-1").ok
    assert s.set_member_span_load(
        1, "trapezoid", [0, 0, -10], [0, 0, -20], name="Roof-1").ok
    case = s.model["load_cases"][0]
    assert case["member_loads"] == []
    assert [entry["name"] for entry in case["member_spans"]] == ["Roof-1"]


def test_complete_load_definition_rejects_a_name_reused_for_two_load_objects():
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    generated = s.generate_frame(spans=[6], storeys=[3.6])
    member = generated.payload["beam_member_ids"][0]
    node = s.model["members"][0]["j"]
    before = deepcopy(s.model.get("load_cases", []))
    result = s.set_load_cases(cases=[{
        "name": "D",
        "nodal_loads": [{"name": "Load-1", "node": node,
                          "load": [0, 0, -1000, 0, 0, 0]}],
        "member_loads": [{"name": "Load-1", "member": member,
                           "w": [0, 0, -100]}],
    }])
    assert not result.ok
    assert any("载荷名称重复" in error for error in result.payload["errors"])
    assert s.model.get("load_cases", []) == before


def test_geometry_first_workflow_reaches_a_verified_learning_sample(
        tmp_path, monkeypatch):
    """把新的 CAE 顺序钉死：几何草稿可以一路补齐、求解并成为正样本。"""
    monkeypatch.chdir(tmp_path)
    s = Session()
    assert s.add_nodes([[0, 0, 0], [6, 0, 0]]).ok
    member = s.add_members([[1, 2]])
    assert member.ok and not member.payload["analysis_ready"]
    assert s.define_materials_and_sections(MATERIALS, SECTIONS).ok
    assert s.assign_properties([1], "BEAM", "STEEL").ok
    assert s.set_supports([1, 2], [1, 1, 1, 1, 1, 1], name="BC-Ends").ok
    assert s.add_load_case("Service").ok
    assert s.set_member_span_load(
        1, "trapezoid", [0, 0, -1000], [0, 0, -2000],
        case_name="Service", name="Roof-Variable").ok
    assert s.validate_model().ok
    assert s.solve_model().ok
    trace = s.export_learning_trace(label="geometry-first-e2e")
    assert trace.ok and trace.payload["verified"]
    assert Path(trace.payload["path"]).is_file()


def test_remove_members_cleans_up_the_dangling_node_it_creates():
    """**行为变了：从"报错"改成"自动清理"。**

    抽柱、开洞本来就会留下没有杆件相连的节点。以前这里返回一句
    "通常是留下了悬空节点"就完事——而用户以为自己只是删了一根柱，
    完全对不上，而且他也没有别的办法把那个点删掉。
    现在一并清除，并在 payload 里如实说清楚清了哪些。
    """
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    s.generate_frame(spans=[6.0], storeys=[3.6], column_section="COLUMN",
                     beam_section="BEAM", material="STEEL")
    before = len(s.model["nodes"])
    r = s.remove_members(ids=[1])
    assert r.ok, r.payload
    assert r.payload["removed_orphan_nodes"], "该报出清掉了哪些孤立节点"
    assert len(s.model["nodes"]) < before
    # 清理必须连约束一起清，否则约束会指向不存在的节点
    left = {n["id"] for n in s.model["nodes"]}
    assert all(sp["node"] in left for sp in s.model["supports"])


def test_remove_members_also_drops_loads_on_them():
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    beams = s.generate_frame(spans=[6.0, 6.0], storeys=[3.6],
                             column_section="COLUMN", beam_section="BEAM",
                             material="STEEL", beam_load=20e3).payload["beam_member_ids"]
    s.remove_members(ids=[beams[0]])
    remaining = {int(e["member"]) for e in s.model.get("member_loads", [])}
    assert beams[0] not in remaining


# ------------------------- 内力查询 -------------------------

def solved_frame():
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    beams = s.generate_frame(spans=[6.0, 6.0], storeys=[3.6], bays=[6.0],
                             column_section="COLUMN", beam_section="BEAM",
                             material="STEEL").payload["beam_member_ids"]
    s.set_load_cases(cases=[{"name": "DL", "member_loads":
                             [{"member": m, "w": [0, 0, -20e3]} for m in beams]}])
    s.solve_model()
    return s, beams


def test_query_diagram_gives_the_structure_wide_extreme():
    s, _ = solved_frame()
    r = s.query_diagram(component="Mz")
    assert r.ok
    assert r.payload["unit"] == "kN·m"
    assert r.payload["at_member"] in s.frame.members
    assert abs(r.payload["peak"]) > 0


def test_query_diagram_returns_a_distribution_when_asked():
    s, beams = solved_frame()
    r = s.query_diagram(component="Mz", member=beams[0], stations=9)
    assert r.ok
    assert len(r.payload["values"]) == 9 == len(r.payload["x_m"])
    assert r.payload["x_m"][0] == 0.0
    assert r.payload["x_m"][-1] == pytest.approx(r.payload["length_m"])
    assert r.payload["values"][0] == pytest.approx(r.payload["at_i_end"])
    assert r.payload["values"][-1] == pytest.approx(r.payload["at_j_end"])


def test_query_diagram_omits_the_distribution_by_default():
    """默认只给端值与峰值——把两百个数塞进上下文没有意义。"""
    s, beams = solved_frame()
    r = s.query_diagram(component="Mz", member=beams[0])
    assert "values" not in r.payload and "peak" in r.payload


def test_query_diagram_end_values_match_the_member_end_forces():
    s, beams = solved_frame()
    mid = beams[0]
    forces = s.query_results(what="member_forces", case="DL").payload["all_members"]
    r = s.query_diagram(component="Mz", member=mid)
    assert r.payload["at_i_end"] == pytest.approx(-forces[str(mid)]["Mz_i"], rel=1e-12)
    assert r.payload["at_j_end"] == pytest.approx(forces[str(mid)]["Mz_j"], rel=1e-12)


def test_query_diagram_rejects_bad_input():
    s, _ = solved_frame()
    assert not s.query_diagram(component="Q").ok
    assert not s.query_diagram(component="Mz", member=9999).ok
    assert not s.query_diagram(component="Mz", case="NOPE").ok
    assert not Session().query_diagram(component="Mz").ok


@pytest.mark.parametrize("kind", ["deformed", "axial", "moment", "shear"])
def test_plot_results_covers_every_kind(kind, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s, _ = solved_frame()
    r = s.plot_results(kind=kind)
    assert r.ok, r.payload
    assert Path(r.payload["path"]).exists()


def test_member_forces_and_diagram_agree_to_full_precision():
    """同一个量在 query_results 与 query_diagram 里必须一致到显示精度。

    早先前者保留 3 位、后者 6 位，模型看到两个数会以为是两个不同的结果。
    """
    s, beams = solved_frame()
    mid = beams[0]
    table = s.query_results(what="member_forces", case="DL").payload["all_members"]
    d = s.query_diagram(component="Mz", member=mid)
    assert d.payload["at_i_end"] == pytest.approx(-table[str(mid)]["Mz_i"], rel=1e-12)
    assert d.payload["at_j_end"] == pytest.approx(table[str(mid)]["Mz_j"], rel=1e-12)
    n = s.query_diagram(component="N", member=mid)
    assert n.payload["at_j_end"] == pytest.approx(table[str(mid)]["N"], rel=1e-12)


# ------------------------- 自重 -------------------------

HEAVY = [{"name": "STEEL", "E": 2.1e11, "nu": 0.3, "density": 7850.0}]


def weighted_session():
    s = Session()
    s.define_materials_and_sections(HEAVY, SECTIONS)
    s.generate_frame(spans=[6.0], storeys=[3.6], column_section="COLUMN",
                     beam_section="BEAM", material="STEEL")
    s.set_load_cases(cases=[{"name": "DL", "member_loads":
                             [{"member": 3, "w": [0, 0, -20e3]}]}])
    return s


def test_self_weight_lands_in_the_model_as_visible_loads():
    """自重要写成看得见的荷载，不做求解时的暗账。"""
    s = weighted_session()
    r = s.add_self_weight(case="DL")
    assert r.ok, r.payload
    spans = s.model["load_cases"][0]["member_spans"]
    assert len(spans) == len(s.model["members"])
    assert all(e["kind"] == "uniform" for e in spans)
    assert all(e["w1"][2] < 0 for e in spans)


def test_self_weight_shows_up_in_the_reaction():
    """加自重前后竖向反力之差必须正好等于 Σ ρAgL。"""
    s = weighted_session()
    s.solve_model()
    before = s.query_results(what="reactions", case="DL").payload["vertical_total_kN"]
    s.add_self_weight(case="DL")
    s.solve_model()
    after = s.query_results(what="reactions", case="DL").payload["vertical_total_kN"]
    expected = sum(7850.0 * 0.012 * 9.80665 * 3.6 for _ in range(2))   # 两根柱
    expected += 7850.0 * 0.010 * 9.80665 * 6.0                          # 一根梁
    assert after - before == pytest.approx(expected / 1e3, rel=1e-6)


def test_calling_self_weight_twice_does_not_double_it():
    """重复调用最容易犯的错就是叠加两遍，而且完全不报错。"""
    s = weighted_session()
    s.add_self_weight(case="DL")
    first = [dict(e) for e in s.model["load_cases"][0]["member_spans"]]
    r = s.add_self_weight(case="DL")
    assert r.payload["replaced_previous"] == len(first)
    assert s.model["load_cases"][0]["member_spans"] == first


def test_self_weight_keeps_the_user_own_span_loads():
    """替换的只能是上一次自重生成的那些，用户自己加的不能被顺手清掉。"""
    s = weighted_session()
    mine = {"member": 3, "kind": "point", "w1": [0.0, 0.0, -50e3], "a": 3.0}
    s.model["load_cases"][0]["member_spans"] = [dict(mine)]
    s.add_self_weight(case="DL")
    s.add_self_weight(case="DL")
    spans = s.model["load_cases"][0]["member_spans"]
    assert mine in spans
    assert sum(1 for e in spans if e.get("note") == "self-weight") == 3


def test_self_weight_without_density_says_what_to_do():
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    s.generate_frame(spans=[6.0], storeys=[3.6], column_section="COLUMN",
                     beam_section="BEAM", material="STEEL")
    r = s.add_self_weight()
    assert not r.ok
    assert "density" in r.payload["hint"] and "7850" in r.payload["hint"]


def test_self_weight_rejects_an_unknown_case():
    s = weighted_session()
    r = s.add_self_weight(case="NOPE")
    assert not r.ok and "NOPE" in r.payload["error"]


# ------------------------- 新荷载类型走完整链路 -------------------------

def test_a_midspan_point_load_solves_and_shows_up_in_the_diagram():
    """简支梁跨中集中力 M = PL/4，从 set_load_cases 一路验到 query_diagram。"""
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    s.set_model(model={
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0}, {"id": 2, "x": 6, "y": 0, "z": 0}],
        "members": [{"id": 1, "i": 1, "j": 2, "section": "BEAM", "material": "STEEL"}],
        "supports": [{"node": 1, "fix": [1, 1, 1, 1, 0, 0]},
                     {"node": 2, "fix": [0, 1, 1, 1, 0, 0]}]})
    r = s.set_load_cases(cases=[{"name": "P", "member_spans": [
        {"member": 1, "kind": "point", "w1": [0, 0, -30e3], "a": 3.0}]}])
    assert r.ok, r.payload
    assert s.solve_model().ok
    d = s.query_diagram(component="Mz", member=1)
    assert abs(d.payload["peak"]) == pytest.approx(30e3 * 6.0 / 4 / 1e3, rel=1e-6)


def test_a_settlement_solves_and_is_reported():
    """两端固接梁一端沉降 Δ：端弯矩 6EIΔ/L²。"""
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    s.set_model(model={
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0}, {"id": 2, "x": 6, "y": 0, "z": 0}],
        "members": [{"id": 1, "i": 1, "j": 2, "section": "BEAM", "material": "STEEL"}],
        "supports": [{"node": 1, "fix": [1] * 6}, {"node": 2, "fix": [1] * 6}]})
    assert s.set_load_cases(cases=[{"name": "SET", "settlements": [
        {"node": 2, "d": [0, 0, -0.01, 0, 0, 0]}]}]).ok
    assert s.solve_model().ok
    d = s.query_diagram(component="Mz", member=1)
    expected = 6 * 2.1e11 * 3e-4 * 0.01 / 6.0 ** 2 / 1e3
    assert abs(d.payload["peak"]) == pytest.approx(expected, rel=1e-6)


def test_a_settlement_on_a_free_direction_is_blocked_before_solving():
    """算前守门要拦住"沉降加在没约束的方向上"这种无声失效。"""
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    s.set_model(model={
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0}, {"id": 2, "x": 6, "y": 0, "z": 0}],
        "members": [{"id": 1, "i": 1, "j": 2, "section": "BEAM", "material": "STEEL"}],
        "supports": [{"node": 1, "fix": [1] * 6}, {"node": 2, "fix": [0, 1, 1, 1, 0, 0]}]})
    r = s.set_load_cases(cases=[{"name": "SET", "settlements": [
        {"node": 2, "d": [-0.01, 0, 0, 0, 0, 0]}]}])
    assert not r.ok
    assert any("不会生效" in e for e in r.payload["errors"]), r.payload
