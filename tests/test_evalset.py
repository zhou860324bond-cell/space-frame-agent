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
    assert v.checks["未擅自建模"] and v.checks["指出了缺失项"]


def test_building_a_model_anyway_fails_the_ask_check():
    """信息不全却自己填默认值开算——这正是要抓的行为。"""
    v = score(by_id("Q01"), play(good_r01_script()))
    assert not v.passed
    assert not v.checks["未擅自建模"]


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
