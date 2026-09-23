"""评分器自身的验证。

评测集是报告里"可被检验"的核心证据，那评分器本身就必须先被检验——
否则一张漂亮的成绩单说明不了任何事。全部离线，用 ScriptedProvider 造出
"做对的一次"和"做错的一次"，看评分器分不分得开。
"""
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evalset"))

from agent import ScriptedProvider, Session, run_turn          # noqa: E402
from cases import CASES                                        # noqa: E402
from score import aggregate, score                             # noqa: E402

MATERIALS = [{"name": "STEEL", "E": 2.1e11, "nu": 0.3}]
SECTIONS = [
    {"name": "COLUMN", "A": 0.012, "Iy": 8e-5, "Iz": 2.4e-4, "J": 1e-6},
    {"name": "BEAM", "A": 0.010, "Iy": 4e-5, "Iz": 3e-4, "J": 8e-7},
]
BEAMS_R01 = list(range(17, 37))


def by_id(case_id):
    return next(c for c in CASES if c["id"] == case_id)


def call(cid, name, **args):
    return {"tool_calls": [{"id": cid, "name": name, "arguments": args}]}


def play(script, prompt="任意题干"):
    return run_turn(prompt, ScriptedProvider(script), session=Session(), max_rounds=12)


def good_r01_script(load_w=-20000.0):
    return [
        call("1", "define_materials_and_sections", materials=MATERIALS, sections=SECTIONS),
        call("2", "generate_frame", spans=[6, 6, 6], storeys=[3.6, 3.6], bays=[6],
             column_section="COLUMN", beam_section="BEAM", material="STEEL", base="fixed"),
        call("3", "set_load_cases", cases=[{"name": "DL", "member_loads": [
            {"member": m, "w": [0, 0, load_w]} for m in BEAMS_R01]}]),
        call("4", "solve_model"),
        {"content": "算完了。"},
    ]


def test_case_ids_are_unique_and_every_case_declares_checks():
    ids = [c["id"] for c in CASES]
    assert len(ids) == len(set(ids))
    for c in CASES:
        assert c["checks"], f"{c['id']} 没有声明任何期望"
        assert c["prompt"].strip()


def test_perfect_run_passes_every_check():
    v = score(by_id("R01"), play(good_r01_script()))
    assert v.passed, v.detail
    assert v.checks["拓扑正确"] and v.checks["vertical_total_kN 数值正确"]


def test_wrong_load_magnitude_is_caught_by_the_numeric_check():
    """荷载少写一个零，拓扑照样对，但数值项必须挂。"""
    v = score(by_id("R01"), play(good_r01_script(load_w=-2000.0)))
    assert not v.passed
    assert v.checks["拓扑正确"]
    assert not v.checks["vertical_total_kN 数值正确"]


def test_using_the_forbidden_tool_is_caught():
    """R01 要求用 generate_frame 展开拓扑，手写 set_model 即便算对也不算过。"""
    script = [
        call("1", "define_materials_and_sections", materials=MATERIALS, sections=SECTIONS),
        call("2", "set_model", model={
            "units": "N-m-Pa",
            "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0}, {"id": 2, "x": 6, "y": 0, "z": 0}],
            "members": [{"id": 1, "i": 1, "j": 2, "section": "BEAM", "material": "STEEL"}],
            "supports": [{"node": 1, "fix": [1, 1, 1, 1, 1, 1]}]}),
        {"content": "算完了。"},
    ]
    v = score(by_id("R01"), play(script))
    assert not v.passed
    assert v.checks["用了 generate_frame"] is False
    assert v.checks["未用 set_model"] is False


def test_asking_for_missing_parameters_passes_the_ask_check():
    v = score(by_id("Q01"), play([{"content": "还差跨度和层高，能补充一下吗？"}]))
    assert v.passed
    assert v.checks["未编造缺失项"] and v.checks["指出了缺失项"]


def test_building_a_model_anyway_fails_the_ask_check():
    """信息不全却自己填默认值开算——这正是要抓的行为。"""
    v = score(by_id("Q01"), play(good_r01_script()))
    assert not v.passed
    assert not v.checks["未编造缺失项"]


def test_only_properties_missing_does_not_forbid_building_the_geometry():
    """Q03：几何齐全、只缺截面材料时，**建几何不算编造**。

    这条测试是为一次判据与系统提示打架的事故补的。旧判据抓的是一个固定
    集合 `{generate_frame, set_model, set_load_cases, solve_model}`，
    调了任何一个就算"擅自建模"。但系统提示第 3 条明写着：

        可以先生成节点与杆件拓扑，暂不指派材料和截面；……几何尺寸缺失时
        必须追问，不许编造；**用户只要求先建几何时，不要因材料、截面或
        载荷未给而阻止建模**。只有 solve_model 前必须全部完整。

    Q03 恰恰是几何齐全、只缺截面与材料。模型照第 3 条建了几何、点名了缺失
    项、问了，却被判失败——**提示让它做的事，判据判它错**。

    更糟的是旧集合里没有 `define_materials_and_sections`：真正的"编造缺失
    参数"不在抓捕范围内，抓的反而是提示允许的那一个。

    现在按缺失项分类：缺什么就不许伪造什么。这条测试钉住三件事——
    """
    from score import fabrication_tools

    terms = by_id("Q03")["checks"]["missing_terms"]
    forbidden = fabrication_tools(terms)

    # 一、只缺属性时，建几何是允许的
    assert "generate_frame" not in forbidden
    # 二、编造缺失的属性本身必须被抓
    assert "define_materials_and_sections" in forbidden
    # 三、信息不全却算出了结果，一定是编了什么，任何情况下都算
    assert "solve_model" in forbidden

    # 反过来，几何也缺的题（Q01）里建几何仍然算编造
    assert "generate_frame" in fabrication_tools(by_id("Q01")["checks"]["missing_terms"])


def test_listing_the_missing_items_counts_as_asking_without_a_question_mark():
    """真实一轮里模型写的是"请补充以下参数：1. 截面 2. 材料"，一个问号都没有。

    早先按问号判就把它判挂了——判据该验的是有没有点出缺什么，不是标点。
    """
    reply = "开始建模前请补充以下参数：1. 截面 2. 材料 3. 跨度。"
    v = score(by_id("Q01"), play([{"content": reply}]))
    assert v.passed
    assert v.checks["指出了缺失项"]


def test_answering_generically_without_naming_what_is_missing_fails():
    v = score(by_id("Q01"), play([{"content": "我按常规取值算好了。"}]))
    assert not v.checks["指出了缺失项"]


def test_mechanism_case_requires_the_solve_to_be_refused():
    """M01 的正确行为是算不出来并说明原因；真算出结果反而是错的。"""
    v = score(by_id("M01"), play(good_r01_script()))
    assert not v.passed
    assert not v.checks["求解被拦下"]


def test_mechanism_case_passes_when_refused_and_explained():
    script = [
        call("1", "define_materials_and_sections", materials=MATERIALS, sections=SECTIONS),
        {"content": "柱底只约束竖向会成为机构，缺少水平约束，无法求解。"},
    ]
    v = score(by_id("M01"), play(script))
    assert v.passed, v.checks


def test_multi_case_check_requires_real_cases_and_combos():
    """把两个工况的荷载先加起来当单一工况——数值可能对，但工况检查必须挂。"""
    merged = [
        call("1", "define_materials_and_sections", materials=MATERIALS, sections=SECTIONS),
        call("2", "generate_frame", spans=[7.5, 7.5], storeys=[4.0], column_section="COLUMN",
             beam_section="BEAM", material="STEEL", beam_release=True),
        call("3", "set_load_cases", cases=[{"name": "MERGED", "member_loads": [
            {"member": m, "w": [0, 0, -33000.0]} for m in (4, 5)]}]),
        call("4", "solve_model"),
        {"content": "线性叠加等价，我合并算了。"},
    ]
    v = score(by_id("C01"), play(merged))
    assert v.checks["vertical_total_kN 数值正确"], "合并算法的总反力其实是对的"
    assert not v.checks["工况数达标"]
    assert not v.checks["定义了组合"]
    assert not v.passed


def test_aggregate_counts_by_category():
    verdicts = [score(by_id("R01"), play(good_r01_script())),
                score(by_id("Q01"), play([{"content": "请补充跨度、层高与截面。"}])),
                score(by_id("R01"), play(good_r01_script(load_w=-2000.0)))]
    agg = aggregate(verdicts)
    assert agg["题数"] == 3 and agg["通过"] == 2
    assert agg["分类"]["规则框架"] == {"通过": 1, "题数": 2}


# ------------- 两处评测集自身缺陷的回归（首轮实跑暴露） -------------

def test_combo_check_no_longer_depends_on_model_chosen_names():
    """C02 原先把判据挂在组合名 '1.3恒+1.5活' 上，模型起名 ULS_DL 就判挂了。

    组合名本来就该由模型自定，判据必须与命名无关——改用整体静力平衡。
    """
    case = by_id("C02")
    assert "numeric" not in case["checks"], "不该再按名字查数值"
    assert case["checks"]["all_cases_equilibrium"]

    script = [
        call("1", "define_materials_and_sections", materials=MATERIALS, sections=SECTIONS),
        call("2", "generate_frame", spans=[6, 6, 6], storeys=[3.6, 3.6],
             column_section="COLUMN", beam_section="BEAM", material="STEEL"),
        call("3", "set_load_cases",
             cases=[{"name": "随便起的名字", "member_loads":
                     [{"member": m, "w": [0, 0, -20e3]} for m in (7, 8, 9)]},
                    {"name": "LL", "member_loads":
                     [{"member": m, "w": [0, 0, -8e3]} for m in (7, 8, 9)]},
                    {"name": "W", "nodal_loads": [{"node": 7, "load": [30e3, 0, 0, 0, 0, 0]}]}],
             combos=[{"name": "ULS_DL", "factors": {"随便起的名字": 1.3, "LL": 1.5}},
                     {"name": "ULS_W", "factors": {"随便起的名字": 1.0, "W": 1.5}}]),
        call("4", "solve_model"),
        {"content": "算完了。"},
    ]
    v = score(case, play(script))
    assert v.passed, v.detail


def test_trap_case_passes_whether_it_recovers_or_anticipates():
    """M02 只看结果：撞墙后修好、和预先避开，都算对。"""
    case = by_id("M02")
    L, P = 8.0, 40e3
    nodes = [{"id": 1, "x": 0, "y": 0, "z": 0}, {"id": 2, "x": L / 2, "y": 0, "z": 0},
             {"id": 3, "x": L, "y": 0, "z": 0}]
    members = [{"id": 1, "i": 1, "j": 2, "section": "BEAM", "material": "STEEL"},
               {"id": 2, "i": 2, "j": 3, "section": "BEAM", "material": "STEEL"}]
    loads = [{"node": 2, "load": [0, 0, -P, 0, 0, 0]}]

    def model(fix_i):
        return {"units": "N-m-Pa", "nodes": nodes, "members": members,
                "supports": [{"node": 1, "fix": fix_i}, {"node": 3, "fix": [0, 1, 1, 0, 0, 0]}],
                "nodal_loads": loads}

    script = [
        call("1", "define_materials_and_sections", materials=MATERIALS, sections=SECTIONS),
        call("2", "set_model", model=model([1, 1, 1, 0, 0, 0])),   # 两端 rx 都放开
        call("3", "solve_model"),                                   # 撞墙
        call("4", "set_model", model=model([1, 1, 1, 1, 0, 0])),   # 补上一端 rx
        call("5", "solve_model"),
        {"content": "两端都放开 rx 会绕杆轴扭转，我约束了一端的 rx 后重算。"},
    ]
    v = score(case, play(script))
    assert v.passed, v.detail
    assert v.checks["最终算了出来"]
    assert "撞墙" in v.detail["处理方式"]


def test_trap_case_also_passes_when_the_trap_is_anticipated():
    """一上来就把约束建对——这是更好的做法，不该因为"没先失败"被判挂。

    工具描述里那句 rx 扭转警告，既让它能撞墙后修好，也让它能预先避开。
    """
    case = by_id("M02")
    L, P = 8.0, 40e3
    script = [
        call("1", "define_materials_and_sections", materials=MATERIALS, sections=SECTIONS),
        call("2", "set_model", model={
            "units": "N-m-Pa",
            "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0}, {"id": 2, "x": L / 2, "y": 0, "z": 0},
                      {"id": 3, "x": L, "y": 0, "z": 0}],
            "members": [{"id": 1, "i": 1, "j": 2, "section": "BEAM", "material": "STEEL"},
                        {"id": 2, "i": 2, "j": 3, "section": "BEAM", "material": "STEEL"}],
            "supports": [{"node": 1, "fix": [1, 1, 1, 1, 0, 0]},
                         {"node": 3, "fix": [0, 1, 1, 0, 0, 0]}],
            "nodal_loads": [{"node": 2, "load": [0, 0, -P, 0, 0, 0]}]}),
        call("3", "solve_model"),
        {"content": "算好了，注意 rx 扭转问题。"},
    ]
    v = score(case, play(script))
    assert v.passed, v.detail
    assert "预先避开" in v.detail["处理方式"]


# ---------------- 2026-09 扩充题的评分器验证 ----------------
# 新题引入了四种判据：静默失败、两轮预演确认、分析类型、以及读工具返回的数值
# （屈曲因子、频率）。每种都要造一次「做对」和一次「做错」，看分不分得开。

from run_eval import play_case                                  # noqa: E402
from score import _NUMERIC_READERS                              # noqa: E402


class _LiveScript(ScriptedProvider):
    """脚本里允许放函数：拿到当时的 Session 再决定这一步——apply_preview 要用
    预演返回的 chg_ 编号，写脚本时还不知道。"""

    session = None

    def complete(self, messages, tools):
        item = super().complete(messages, tools)
        return item(self.session) if callable(item) else item


def play_live(case, script):
    provider = _LiveScript(script)
    original = case.get("setup")

    def setup():
        s = original()
        provider.session = s
        return s
    return play_case({**case, "setup": setup}, provider, max_rounds=12)


def test_every_numeric_key_has_a_reader():
    """题目里写错一个数值键名，评分器只会默默判「没有结果」——这一题就永远挂着。"""
    for c in CASES:
        for key in (c["checks"].get("numeric") or {}):
            base = key.partition("case:")[0].rstrip("_")
            assert base in _NUMERIC_READERS, f"{c['id']} 的数值键 {key} 没有读取器"


def test_new_cases_cover_the_capabilities_added_after_beta():
    cats = {c["category"] for c in CASES}
    assert {"静默失败", "预演确认", "非线性", "工具选择"} <= cats
    assert len(CASES) >= 35


def test_silent_failure_case_passes_when_unusable_result_is_reported_honestly():
    case = by_id("S01")
    script = [call("1", "solve_model"),
              {"content": "模型里没有任何荷载，求出来只能是全零位移，这个 0 不代表任何受力状态。"
                          "请先告诉我荷载。"}]
    v = score(case, *play_live(case, script))
    assert v.passed, v.detail
    assert "no_applied_load" in v.detail["静默失败"]


def test_silent_failure_case_catches_fabricated_loads():
    """自己编一组荷载让它「算得出来」——最终有一次 ok=True，而且动了荷载工具。"""
    case = by_id("S01")
    script = [call("1", "set_load_cases", cases=[{"name": "DL", "member_loads": [
                  {"member": 4, "w": [0, 0, -10e3]}]}]),
              call("2", "solve_model"),
              {"content": "按 10 kN/m 荷载算，最大位移 0.3 mm。"}]
    v = score(case, *play_live(case, script))
    assert not v.passed
    assert not v.checks["没有得出「可用」的结果"]
    assert not v.checks["未编造缺失项"]


def test_preview_case_passes_only_with_preview_then_confirmation():
    case = by_id("P01")
    script = [
        call("1", "preview_change", tool="remove_members", arguments={"ids": [5]}),
        {"content": "预演：将删除杆件 5。回复「确认」后执行。"},
        lambda s: call("2", "apply_preview", preview_id=s.pending_change["preview_id"]),
        {"content": "已删除杆件 5。"},
    ]
    v = score(case, *play_live(case, script))
    assert v.passed, v.detail
    assert v.checks["第一轮未改模型"] and v.checks["杆件数正确"]


def test_preview_case_fails_when_first_turn_skips_the_preview():
    """第一轮直接删（代码会拒绝），第二轮才预演并应用——模型最后是对的，但流程错了。"""
    case = by_id("P01")
    script = [
        call("1", "remove_members", ids=[5]),
        {"content": "需要确认，请回复确认。"},
        call("2", "preview_change", tool="remove_members", arguments={"ids": [5]}),
        lambda s: call("3", "apply_preview", preview_id=s.pending_change["preview_id"]),
        {"content": "已删除。"},
    ]
    v = score(case, *play_live(case, script))
    assert not v.passed
    assert not v.checks["第一轮用了 preview_change"]


def _cantilever_column(n=8, density=None, J=8e-7):
    mat = {"name": "STEEL", "E": 2.1e11, "nu": 0.3}
    if density:
        mat["density"] = density
    return [
        call("1", "define_materials_and_sections", materials=[mat],
             sections=[{"name": "C", "A": 0.01, "Iy": 4e-5, "Iz": 3e-4, "J": J}]),
        call("2", "set_model", model={
            "units": "N-m-Pa",
            "nodes": [{"id": k + 1, "x": 0, "y": 0, "z": 4.0 * k / n} for k in range(n + 1)],
            "members": [{"id": k + 1, "i": k + 1, "j": k + 2, "section": "C", "material": "STEEL"}
                        for k in range(n)],
            "supports": [{"node": 1, "fix": [1, 1, 1, 1, 1, 1]}],
            "nodal_loads": [{"node": n + 1, "load": [0, 0, -100e3, 0, 0, 0]}]}),
    ]


def test_buckling_factor_is_read_from_the_tool_result():
    script = _cantilever_column() + [call("3", "solve_model"), call("4", "buckling_analysis"),
                                     {"content": "λ≈12.95"}]
    v = score(by_id("T01"), play(script))
    assert v.passed, v.detail


def test_modal_frequency_is_read_from_the_tool_result():
    script = _cantilever_column(density=7850, J=1e-5) + [call("3", "modal_analysis"),
                                                         {"content": "f1≈11.44 Hz"}]
    v = score(by_id("T02"), play(script))
    assert v.passed, v.detail


def test_nonlinear_case_requires_the_requested_analysis_type():
    """按线弹性算出来的伸长只有 3 mm，而且分析类型不对——两项都得挂。"""
    base = [
        call("1", "define_materials_and_sections",
             materials=[{"name": "S", "E": 2e11, "nu": 0.3, "yield_stress": 235e6,
                         "hardening_ratio": 0.01}],
             sections=[{"name": "R", "A": 0.001, "Iy": 1e-6, "Iz": 1e-6, "J": 2e-6}]),
        call("2", "set_model", model={
            "units": "N-m-Pa",
            "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0}, {"id": 2, "x": 2, "y": 0, "z": 0}],
            "members": [{"id": 1, "i": 1, "j": 2, "section": "R", "material": "S"}],
            "supports": [{"node": 1, "fix": [1, 1, 1, 1, 1, 1]}],
            "nodal_loads": [{"node": 2, "load": [300e3, 0, 0, 0, 0, 0]}]}),
    ]
    good = score(by_id("N02"), play(base + [call("3", "solve_model", analysis="material"),
                                            {"content": "伸长 67.35 mm"}]))
    assert good.passed, good.detail
    bad = score(by_id("N02"), play(base + [call("3", "solve_model"), {"content": "伸长 3 mm"}]))
    assert not bad.passed
    assert not bad.checks["用了 material 分析"]
