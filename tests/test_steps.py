"""分析步的传播与失活。

传播是**隐式**的：某一步没提到的东西自动沿用上一步。这正是它好用的地方，
也是它最危险的地方——漏写不等于撤销，而"多算了一个上一步的荷载"这种错
看着完全正常，量级对、图形也像那么回事。

所以这里每一条都盯着"实际生效的到底是什么"，而不是"调用成功了没有"：
结构层的几条对着闭合解验，语义层的几条对着 Abaqus 的规则验。
"""

import pytest

from agent import Session
from steps import INITIAL_STEP, Step, resolve

E, IZ = 2.1e11, 4e-4
SPAN, W_DEAD, W_LIVE = 6.0, 10e3, 15e3

PAYLOAD = {
    "supports": [{"node": 1, "fix": [1, 1, 1, 1, 1, 1]},
                 {"node": 2, "fix": [0, 0, 1, 0, 0, 0]}],
    "load_cases": [{"name": "DL"}, {"name": "LL"}],
}


def two_span_session():
    """两跨连续梁，中支座可以在分析步里拆掉——拆完正好是一端固定的单跨。"""
    s = Session()
    s.add_nodes([[0, 0, 0], [SPAN, 0, 0], [2 * SPAN, 0, 0]])
    s.define_materials_and_sections(
        materials=[{"name": "M", "E": E, "nu": 0.3}],
        sections=[{"name": "S", "A": 0.02, "Iy": 2e-4, "Iz": IZ, "J": 1e-5}])
    s.add_members([[1, 2], [2, 3]], "S", "M")
    s.set_supports([1], fix=[1, 1, 1, 1, 1, 1])
    s.set_supports([2], fix=[0, 0, 1, 0, 0, 0])
    s.set_supports([3], fix=[0, 0, 1, 0, 0, 0])
    s.add_load_case("DL")
    s.add_load_case("LL")
    for member in (1, 2):
        s.set_member_load(member, [0, 0, -W_DEAD], case_name="DL")
        s.set_member_load(member, [0, 0, -W_LIVE], case_name="LL")
    return s


# --- 语义层：传播与失活 ---------------------------------------------------

def test_a_load_declared_once_propagates_to_every_later_step():
    """**这是整个特性的要点。** 第一步声明的荷载，后面每一步都还在。"""
    steps = [Step("一", loads={"DL": "RAMP"}), Step("二"), Step("三")]
    effective = resolve(steps, PAYLOAD)
    assert [e.loads for e in effective] == [{"DL": "RAMP"}] * 3


def test_a_step_that_changes_nothing_says_so_instead_of_looking_empty():
    """没变化的一步要明说"全部沿用"，不能只给一份空的变化列表。

    传播是隐式的；变化列表空着，读的人分不清"这步什么都没做"和
    "这步的变化没被记录"。
    """
    effective = resolve([Step("一", loads={"DL": "RAMP"}), Step("二")], PAYLOAD)
    assert effective[1].changes == ("与上一步相同（全部沿用）",)


def test_deactivating_a_load_removes_it_only_from_that_step_onward():
    steps = [Step("一", loads={"DL": "RAMP", "LL": "RAMP"}),
             Step("二", deactivate_loads=("LL",)),
             Step("三")]
    effective = resolve(steps, PAYLOAD)
    assert set(effective[0].loads) == {"DL", "LL"}
    assert set(effective[1].loads) == {"DL"}
    assert set(effective[2].loads) == {"DL"}, "失活之后不该自己长回来"


def test_a_later_step_can_overwrite_an_amplitude_without_redeclaring_the_rest():
    steps = [Step("一", loads={"DL": "RAMP", "LL": "RAMP"}),
             Step("二", loads={"LL": "STEP"})]
    effective = resolve(steps, PAYLOAD)
    assert effective[1].loads == {"DL": "RAMP", "LL": "STEP"}
    assert "改写荷载 LL=STEP" in effective[1].changes


def test_the_initial_supports_are_the_starting_state():
    """初始状态是模型自带的支座，对应 Abaqus 的 Initial 步。"""
    effective = resolve([Step("一", loads={"DL": "RAMP"})], PAYLOAD)
    assert sorted(effective[0].supports) == [1, 2]


def test_a_support_can_be_overwritten_in_a_later_step():
    steps = [Step("一", loads={"DL": "RAMP"}),
             Step("二", supports={1: {"fix": [1, 1, 1, 0, 0, 0]}})]
    effective = resolve(steps, PAYLOAD)
    assert effective[0].supports[1]["fix"] == [1, 1, 1, 1, 1, 1]
    assert effective[1].supports[1]["fix"] == [1, 1, 1, 0, 0, 0]


def test_the_analysis_type_does_not_propagate():
    """分析类型跟着每一步自己走，不沿用——Abaqus 里 procedure 也是这样。"""
    steps = [Step("一", loads={"DL": "RAMP"}, analysis="pdelta"), Step("二")]
    assert [e.analysis for e in resolve(steps, PAYLOAD)] == ["pdelta", "linear"]


# --- 静默失败的闸门 -------------------------------------------------------

@pytest.mark.parametrize(("label", "steps"), [
    ("失活一个没生效的工况",
     [Step("一", loads={"DL": "RAMP"}), Step("二", deactivate_loads=("LL",))]),
    ("失活一个不存在的支座",
     [Step("一", loads={"DL": "RAMP"}, deactivate_supports=(9,))]),
    ("引用未定义的工况", [Step("一", loads={"没这个": "RAMP"})]),
    ("引用未定义的幅值曲线", [Step("一", loads={"DL": "没这条"})]),
    ("分析步重名", [Step("一", loads={"DL": "RAMP"}), Step("一")]),
])
def test_a_meaningless_step_is_refused_instead_of_silently_skipped(label, steps):
    """**默默跳过是不行的。**

    节点号写错、或者把失活写在激活之前，如果只是静静地不生效，结果会
    完全正常地算出来——只是算的不是那个结构。这一组全部要求报错。
    """
    with pytest.raises(ValueError):
        resolve(steps, PAYLOAD)


def test_removing_every_support_is_refused_before_the_solver_sees_it():
    """拆光支座是机构。在这里拦住，报错才指得到分析步，而不是"刚度矩阵奇异"。"""
    steps = [Step("一", loads={"DL": "RAMP"}, deactivate_supports=(1, 2))]
    with pytest.raises(ValueError, match="机构"):
        resolve(steps, PAYLOAD)


def test_the_first_step_must_bring_its_own_loads():
    with pytest.raises(ValueError, match="一个荷载都没有"):
        resolve([Step("一")], PAYLOAD)


def test_the_initial_step_name_is_reserved():
    with pytest.raises(ValueError, match="保留名字"):
        Step(INITIAL_STEP)


def test_activating_and_deactivating_the_same_thing_is_refused():
    """先后顺序没有公认答案，不替用户猜。"""
    with pytest.raises(ValueError, match="同时激活又失活"):
        Step("一", loads={"DL": "RAMP"}, deactivate_loads=("DL",))


# --- 结构层：对着闭合解 ---------------------------------------------------

def test_deactivating_the_middle_support_really_changes_the_structure():
    """拆掉中支座，两跨连续梁变成 2L 的一端固定一端简支梁。

    判据是闭合解 wL⁴/(184.6 EI)，不是"数变大了"。支座失活如果只是写进了
    报告而没进到刚度矩阵，挠度会纹丝不动——那种错自检查不出来。
    """
    s = two_span_session()
    s.add_step("满载", loads={"DL": "RAMP"})
    s.add_step("拆中支座", deactivate_supports=[2])
    result = s.solve_steps()
    assert result.ok, result.payload

    rows = {row["step"]: row for row in result.payload["steps"]}
    length = 2 * SPAN
    propped = W_DEAD * length**4 / (184.6 * E * IZ) * 1e3
    assert rows["拆中支座"]["max_deflection_mm"] == pytest.approx(propped, rel=0.01)
    assert rows["拆中支座"]["max_deflection_mm"] > (
        5 * rows["满载"]["max_deflection_mm"])


def test_deactivating_the_live_load_scales_the_answer_exactly():
    """线性问题里撤掉活载，挠度按荷载比例缩小——比例对不上就是荷载没撤干净。"""
    s = two_span_session()
    s.add_step("满载", loads={"DL": "RAMP", "LL": "RAMP"})
    s.add_step("撤活载", deactivate_loads=["LL"])
    rows = {row["step"]: row for row in s.solve_steps().payload["steps"]}
    ratio = W_DEAD / (W_DEAD + W_LIVE)
    assert rows["撤活载"]["max_deflection_mm"] == pytest.approx(
        rows["满载"]["max_deflection_mm"] * ratio, rel=1e-6)


def test_a_linear_step_uses_the_amplitude_value_at_the_end_of_the_step():
    """线性步的叠加系数取幅值曲线在 t=1 处的值，**不是一律取 1.0**。

    自定义曲线完全可以收在 0.5。一律取 1.0 会把荷载放大一倍，而结果
    看着完全正常。
    """
    s = two_span_session()
    s.define_amplitude("收在一半", points=[[0, 0], [1, 0.5]])
    s.add_step("全值", loads={"DL": "RAMP"})
    s.add_step("半值", loads={"DL": "收在一半"})
    rows = {row["step"]: row for row in s.solve_steps().payload["steps"]}
    # 报出来的挠度已经四舍五入到 6 位小数，容差不能比报告精度还细
    assert rows["半值"]["max_deflection_mm"] == pytest.approx(
        0.5 * rows["全值"]["max_deflection_mm"], rel=1e-5)


def test_solve_steps_leaves_the_chosen_step_loaded_for_post_processing():
    """会话一次端一份结果，得说清楚端的是哪一步，并且能换。"""
    s = two_span_session()
    s.add_step("满载", loads={"DL": "RAMP", "LL": "RAMP"})
    s.add_step("撤活载", deactivate_loads=["LL"])

    last = s.solve_steps()
    assert last.payload["inspecting"] == "撤活载"
    deflection_of_last = s.solution

    chosen = s.solve_steps(inspect="满载")
    assert chosen.payload["inspecting"] == "满载"
    assert s.solution is not deflection_of_last


def test_an_unknown_step_name_for_inspection_is_refused():
    s = two_span_session()
    s.add_step("满载", loads={"DL": "RAMP"})
    result = s.solve_steps(inspect="没这步")
    assert not result.ok
    assert "没这步" in result.payload["error"]


def test_solving_without_any_step_points_at_solve_model():
    result = two_span_session().solve_steps()
    assert not result.ok
    assert "solve_model" in result.payload["hint"]


# --- 工具层 ---------------------------------------------------------------

def test_a_step_that_cannot_be_resolved_is_not_written_to_the_model():
    """写不进去比写进去再报错好：模型永远处在能结算的状态。"""
    s = two_span_session()
    s.add_step("满载", loads={"DL": "RAMP"})
    before = list(s.model.get("steps") or [])
    bad = s.add_step("撤个没有的", deactivate_loads=["没这个"])
    assert not bad.ok
    assert list(s.model.get("steps") or []) == before


def test_list_steps_shows_both_what_was_declared_and_what_takes_effect():
    """只给声明是不够的：第三步写着一行失活，生效的却是前面传下来的一整套。"""
    s = two_span_session()
    s.add_step("满载", loads={"DL": "RAMP", "LL": "RAMP"})
    s.add_step("撤活载", deactivate_loads=["LL"])
    payload = s.list_steps().payload
    assert payload["declared"][1].get("loads") is None
    assert payload["effective"][1]["loads"] == {"DL": "RAMP"}


def test_a_step_can_be_inserted_in_the_middle():
    s = two_span_session()
    s.add_step("一", loads={"DL": "RAMP"})
    s.add_step("三", deactivate_supports=[2])
    s.add_step("二", loads={"LL": "RAMP"}, after="一")
    assert [x["step"] for x in s.list_steps().payload["effective"]] == [
        "一", "二", "三"]


def test_deleting_a_step_that_others_depend_on_is_refused():
    """删中间一步会让后面的失活失去对象——那时候不该默默写进去。"""
    s = two_span_session()
    s.add_step("加活载", loads={"DL": "RAMP", "LL": "RAMP"})
    s.add_step("撤活载", deactivate_loads=["LL"])
    result = s.delete_step("加活载")
    assert not result.ok
    assert len(s.model["steps"]) == 2, "删除失败时模型必须原样保留"
