"""工具契约：每个工具真调一次，核对它实际返回什么。

**这一组测试是被一次教训逼出来的。**

`result_rows` 第一版是按我猜的 payload 结构写的，配套测试也用我自己编的
payload 喂进去——测试在拿我的猜测验证我的猜测，全绿，而界面上显示"共 0 根杆件"。
是截图抓到的，不是测试。

所以这里立的规矩是：

1. **契约写在一张表里**，它同时就是文档——想知道 `query_envelope` 返回什么，
   看 `CONTRACTS` 而不是去读实现。
2. **真调工具**，拿它实际返回的键去比。自己编的输入只能验格式，
   验不了接口对不对。
3. **新增工具没写契约要能被测出来**，否则这张表很快就跟不上代码。

另外还查一类**交互 bug**：单个工具都对，组合起来才错。
"加自重 → 删一根柱"就是这样——自重写的是 `member_spans`，
而删除的清理只过滤了 `member_loads`，于是留下指向已删杆件的荷载，
校验在下一步才报错，指向的却不是删除那一步。
"""

from __future__ import annotations

import pytest

from agent import TOOLS, Session
from sections import i_section

MAT = [{"name": "Q355", "E": 2.06e11, "nu": 0.3, "density": 7850.0}]
SEC = [i_section("COLUMN", .4, .2, .008, .012),
       i_section("BEAM", .5, .2, .008, .014),
       i_section("BRACE", .15, .15, .006, .008)]


# ------------------------------------------------- 契约表
#
# 每条：工具名 → (必有的键, 说明)。
# 必有的键是**成功时**一定出现的；可选键（比如只有传了 span 才有的
# span_over_deflection）不列，由各自的测试单独盯。

CONTRACTS: dict[str, tuple[tuple[str, ...], str]] = {
    "define_materials_and_sections": (("materials", "sections"),
                                      "回显定义了哪些材料与截面"),
    "generate_frame": (("summary", "beam_member_ids"),
                       "梁编号供分工况加载引用"),
    "generate_portal_frame": (("summary", "rafter_member_ids"),
                              "斜梁编号按几何判别，已排除纵向系杆"),
    "generate_bent": (("summary", "column_member_ids", "beam_member_ids",
                       "rafter_member_ids", "top_node_ids"),
                      "一榀刚架，各类杆件编号分开给"),
    "extrude_bents": (("summary", "bent_count", "tie_member_ids"),
                      "拉伸成几榀、生成了哪些连系梁"),
    "add_bracing": (("summary", "brace_member_ids", "panels"),
                    "加了哪些撑、覆盖几个框格"),
    "add_nodes": (("added_node_ids", "summary"), "新节点编号"),
    "add_members": (("added_member_ids", "summary"), "新杆件编号"),
    "assign_properties": (("members", "analysis_ready"), "批量属性指派"),
    "set_supports": (("nodes", "step"), "Initial 阶段边界条件"),
    "remove_members": (("removed", "summary"), "删了哪些；连带清理见可选键"),
    "edit_member": (("member", "changed", "summary"),
                    "改了哪几项，每项的改前改后都要有"),
    "edit_node": (("node", "changed", "summary"),
                  "同上；坐标与约束都算 changed"),
    "retaper": (("changed_members", "section", "summary"), "改了哪些杆件的截面"),
    "raise_nodes": (("moved_nodes", "dz", "summary"), "抬高了哪些节点"),
    "select_members": (("member_ids", "count"), "按几何条件挑出的编号"),
    "define_set": (("name", "sets"), "集合名与当前全部集合"),
    "list_sets": (("sets", "count"), "各集合的规模概览"),
    "set_model": (("summary",), "整体提交模型"),
    "set_load_cases": (("cases",), "定义了哪些工况"),
    "export_learning_trace": (("path", "steps", "verified", "format"),
                              "可回放的 Agent 建模样本"),
    "add_load_case": (("case", "cases"), "创建空分析工况"),
    "set_nodal_load": (("node", "case", "load"), "节点集中力"),
    "set_member_load": (("member", "case", "load"), "杆件均布荷载"),
    "set_member_span_load": (("member", "case", "kind", "name"),
                              "梯形、三角形或杆中集中力"),
    "set_prescribed_displacement": (("node", "case", "d", "name"),
                                     "分析步给定位移"),
    "set_units": (("units", "was"), "换算前后的单位制"),
    "add_self_weight": (("case", "members", "total_vertical_kN_per_m"),
                        "自重加到哪个工况、覆盖多少杆件"),
    "validate_model": (("summary",), "校验通过时给规模摘要"),
    "preview_analysis_mesh": (("physical_nodes", "physical_members",
                               "analysis_nodes", "analysis_elements",
                               "member_mapping", "split_node_details"),
                              "只读展示物理构件与分析单元映射"),
    "preview_change": (("schema", "tool", "diff", "validation_errors",
                        "would_invalidate_results"),
                       "在模型副本上预演写操作并返回结构化差异"),
    "apply_preview": (("schema", "preview_id", "applied_tool", "result"),
                      "只应用经过新用户消息确认且尚未过期的预演"),
    "solve_model": (("cases",), "每个工况的位移与平衡校核"),
    "query_results": (("case",), "具体键随 what 而变"),
    "query_diagram": (("member", "component", "unit", "peak", "at_x_m"),
                      "单根杆件的内力，**必须带单位**"),
    "query_envelope": (("component", "unit", "peak", "governing_case"),
                       "包络峰值与控制组合，**必须带单位**"),
    "modal_analysis": (("modes", "total_mass_kg", "effective_mass_ratio_xyz"),
                       "各阶频率周期，有效质量比用于判断阶数够不够"),
    "buckling_analysis": (("factors", "critical_factor",
                           "most_compressed_member"),
                          "屈曲因子与最大受压杆件"),
    "diagnose_supports": (("modes",), "检出的刚体模态；无模态时为空表"),
    "sweep": (("rows", "metric", "unit"), "扫描结果表，**必须带单位**"),
    "plot_results": (("path", "case"), "图片落盘路径"),
    "write_report": (("path",), "报告落盘路径"),
    "compare_solvers": ((), "需要 Abaqus，不在测试里跑"),
    "solve_with_abaqus": ((), "需要 Abaqus，不在测试里跑"),
    "analyze_joint_solid": (("node_id", "case"),
                            "节点局部实体：dry_run 只出规格；native 自研求解，"
                            "Abaqus 是可选对标后端"),
}

# 需要外部程序、测试里不实跑的
EXTERNAL = {"compare_solvers", "solve_with_abaqus", "export_learning_trace"}


@pytest.fixture(scope="module")
def solved() -> Session:
    """一个建好并算完的模型，**只给只读类工具共用**。

    要改模型的测试自己建一个（见 `fresh()`）。共用的这个被改过之后，
    后面的测试会发现结果没了——而失败信息指向的是那个无辜的测试。
    module 级的可变 fixture 就是这么制造顺序依赖的，我在这里踩过一次。
    """
    s = Session()
    s.define_materials_and_sections(MAT, SEC)
    g = s.generate_frame(spans=[6.0, 6.0], storeys=[3.6], bays=[6.0],
                         column_section="COLUMN", beam_section="BEAM",
                         material="Q355")
    beams = g.payload["beam_member_ids"]
    s.set_load_cases(
        cases=[{"name": "D", "member_loads":
                [{"member": m, "w": [0, 0, -20e3]} for m in beams]},
               {"name": "L", "member_loads":
                [{"member": m, "w": [0, 0, -10e3]} for m in beams]}],
        combos=[{"name": "1.3D+1.5L", "factors": {"D": 1.3, "L": 1.5}}])
    assert s.solve_model().ok
    return s


def check(result, tool: str) -> dict:
    """断言工具成功，且 payload 含契约要求的全部键。"""
    assert result.ok, f"{tool} 没成功：{str(result.payload)[:200]}"
    required, _ = CONTRACTS[tool]
    missing = [k for k in required if k not in result.payload]
    assert not missing, (
        f"{tool} 的 payload 缺少契约要求的键 {missing}；"
        f"实际给的是 {sorted(result.payload)}")
    return result.payload


# ------------------------------------------------- 表本身要跟得上代码

def test_every_tool_has_a_contract():
    """**新增工具没写契约要能被测出来**，否则这张表很快就跟不上。"""
    names = {f["function"]["name"] for f in TOOLS}
    missing = sorted(names - set(CONTRACTS))
    assert not missing, f"这些工具还没写契约：{missing}"


def test_no_contract_for_a_tool_that_does_not_exist():
    """反过来：删了工具而契约留着，说明表没跟着清。"""
    names = {f["function"]["name"] for f in TOOLS}
    stale = sorted(set(CONTRACTS) - names)
    assert not stale, f"这些契约对应的工具已经不存在：{stale}"


def test_every_contract_explains_itself():
    """契约表同时是文档。只有键名没有说明，等于没写。"""
    for tool, (_, why) in CONTRACTS.items():
        assert len(why) >= 4, f"{tool} 的契约说明太敷衍：{why!r}"


# ------------------------------------------------- 建模类

def test_generators_report_the_ids_you_need_to_load():
    """生成器必须告诉你哪些杆件是梁——**否则没法分工况加载**，
    而用户和大模型都不知道编号规则。"""
    s = Session()
    s.define_materials_and_sections(MAT, SEC)
    p = check(s.generate_frame(spans=[6.0], storeys=[3.6],
                               column_section="COLUMN", beam_section="BEAM",
                               material="Q355"), "generate_frame")
    assert p["beam_member_ids"], "一根梁都没报出来"

    s2 = Session()
    s2.define_materials_and_sections(MAT, SEC)
    p = check(s2.generate_portal_frame(
        spans=[24.0], eave_height=7.5, ridge_rise=1.2,
        column_section="COLUMN", rafter_section="BEAM",
        material="Q355", base="pinned"), "generate_portal_frame")
    assert p["rafter_member_ids"]

    s3 = Session()
    s3.define_materials_and_sections(MAT, SEC)
    p = check(s3.generate_bent(profile=[[0, 7.2], [12, 7.2]],
                               columns=[0, 6, 12], levels=[3.6],
                               column_section="COLUMN", beam_section="BEAM",
                               material="Q355"), "generate_bent")
    assert p["column_member_ids"] and p["rafter_member_ids"]
    p = check(s3.extrude_bents(bays=[6.0], tie_section="BEAM"),
              "extrude_bents")
    assert p["bent_count"] == 2 and p["tie_member_ids"]
    p = check(s3.add_bracing(kind="X", x_range=[-0.1, 6.1],
                             y_range=[-0.1, 0.1], section="BRACE"),
              "add_bracing")
    assert p["brace_member_ids"]


def test_editing_tools_report_what_they_touched():
    s = Session()
    s.define_materials_and_sections(MAT, SEC)
    s.generate_frame(spans=[6.0, 6.0], storeys=[3.6], column_section="COLUMN",
                     beam_section="BEAM", material="Q355")
    check(s.select_members(orientation="vertical"), "select_members")
    check(s.define_set(name="全部柱", orientation="vertical"), "define_set")
    check(s.list_sets(), "list_sets")
    check(s.retaper(member_ids=[1], section="BRACE"), "retaper")
    check(s.raise_nodes(dz=0.2, z_range=[3.5, 3.7]), "raise_nodes")
    check(s.remove_members(ids=[2]), "remove_members")
    check(s.add_nodes(coordinates=[[20.0, 0.0, 0.0]]), "add_nodes")


def test_edit_tools_report_before_and_after():
    """**改前的值必须回报。** 属性面板提交失败时要把控件退回真实值，
    撤销要能说清楚撤的是什么——两者都靠 changed 里的 was。"""
    s = fresh()
    mid = s.model["members"][0]["id"]
    p = check(s.edit_member(member_id=mid, section="BRACE"), "edit_member")
    assert p["changed"]["section"] == {"was": "COLUMN", "now": "BRACE"}
    assert s.solution is None, "改了截面，刚度就变了，旧结果必须失效"

    # 没实际变化时不该记一步：否则面板每次重填表单都会污染建模过程
    again = s.edit_member(member_id=mid, section="BRACE")
    assert again.ok and not again.payload["changed"]

    nid = s.model["nodes"][0]["id"]
    p = check(s.edit_node(node_id=nid, fix=[1, 1, 1, 0, 0, 0]), "edit_node")
    assert p["changed"]["fix"]["was"] == [1, 1, 1, 1, 1, 1]

    bad = s.edit_member(member_id=mid, section="根本没有这个截面")
    assert not bad.ok
    assert "BRACE" in str(bad.payload.get("hint", "")), "报错要列出可选值"
    assert s.model["members"][0]["section"] == "BRACE", "失败不能改坏模型"

    assert not s.edit_member(member_id=99999, section="BRACE").ok


def test_member_reference_vector_rejects_an_axial_direction():
    s = fresh()
    mid = s.model["members"][0]["id"]
    member = next(m for m in s.model["members"] if m["id"] == mid)
    ni = next(n for n in s.model["nodes"] if n["id"] == member["i"])
    nj = next(n for n in s.model["nodes"] if n["id"] == member["j"])
    axis = [nj[k] - ni[k] for k in ("x", "y", "z")]

    bad = s.edit_member(member_id=mid, ref_vector=axis)
    assert not bad.ok
    assert "平行" in bad.payload["error"]
    assert "ref_vector" not in member

    good = s.edit_member(member_id=mid, ref_vector=[0, 1, 0])
    assert good.ok
    assert good.payload["changed"]["ref_vector"] == {"was": None, "now": [0.0, 1.0, 0.0]}


def fresh() -> Session:
    """自己的一份，改坏了不影响别人。"""
    s = Session()
    s.define_materials_and_sections(MAT, SEC)
    g = s.generate_frame(spans=[6.0], storeys=[3.6], column_section="COLUMN",
                         beam_section="BEAM", material="Q355")
    s.set_load_cases(cases=[{"name": "D", "member_loads":
                             [{"member": m, "w": [0, 0, -20e3]}
                              for m in g.payload["beam_member_ids"]]}])
    s.solve_model()
    return s


def test_load_and_unit_tools():
    s = fresh()
    check(s.set_load_cases(cases=[{"name": "D", "member_loads":
                                   [{"member": 3, "w": [0, 0, -20e3]}]}]),
          "set_load_cases")
    p = check(s.set_units(units="N-mm-MPa"), "set_units")
    assert p["was"] == "N-m-Pa" and p["units"] == "N-mm-MPa"


def test_incremental_nonuniform_load_and_prescribed_displacement_tools():
    s = fresh()
    check(s.set_member_span_load(3, "point", [0, 0, -5000], a=3.0,
                                  case_name="D", name="Mid-P"),
          "set_member_span_load")
    check(s.set_prescribed_displacement(1, [0, 0, -0.001, 0, 0, 0],
                                        case_name="D", name="Support-Settle"),
          "set_prescribed_displacement")
    assert s.model["load_cases"][0]["member_spans"][0]["name"] == "Mid-P"
    assert s.model["load_cases"][0]["settlements"][0]["name"] == "Support-Settle"


# ------------------------------------------------- 查询类：单位是硬要求

@pytest.mark.parametrize("tool", ["query_diagram", "query_envelope", "sweep"])
def test_numeric_tools_state_their_unit(solved, tool):
    """**返回数值就必须带单位。** 报告里漏单位是硬伤，工具返回同理——
    界面拿到一个没有单位的数，只能靠猜或者写死，写死就会差一千倍。"""
    calls = {
        "query_diagram": lambda: solved.query_diagram(member=7, component="Mz"),
        "query_envelope": lambda: solved.query_envelope(component="Mz"),
        "sweep": lambda: solved.sweep(what="section_property", target="BEAM",
                                      prop="Iz", values=[1e-4, 3e-4],
                                      metric="max_deflection"),
    }
    p = check(calls[tool](), tool)
    assert p["unit"], f"{tool} 没给单位"


def test_analysis_tools(solved):
    check(solved.validate_model(), "validate_model")
    check(solved.preview_analysis_mesh(), "preview_analysis_mesh")
    check(solved.solve_model(), "solve_model")
    check(solved.query_results(what="max_displacement"), "query_results")
    check(solved.query_results(what="max_deflection"), "query_results")
    check(solved.modal_analysis(num_modes=3), "modal_analysis")
    check(solved.buckling_analysis(num_modes=3), "buckling_analysis")
    check(solved.diagnose_supports(), "diagnose_supports")


def test_deflection_gives_the_span_ratio_only_when_told_the_span(solved):
    """**挠跨比的分母必须是设计跨度，不是杆长。**

    门式刚架的斜梁是两根杆，梁划成几个单元时更是差几倍。所以
    不传 span 时只给"杆长/挠度"，传了才给"跨度/挠度"——
    这个区别不写清楚，用户会拿杆长比当挠跨比用。
    """
    without = solved.query_results(what="max_deflection").payload
    assert "member_length_over_deflection" in without
    assert "span_over_deflection" not in without

    given = solved.query_results(what="max_deflection", span=6.0).payload
    assert given["span_over_deflection"] and given["span_m"] == 6.0


# ------------------------------------------------- 交互：单个都对，组合才错

def test_self_weight_then_delete_a_column():
    """**这条测试来自一个真实的 bug。**

    自重写的是 `member_spans`，而删除的清理当初只过滤了 `member_loads`。
    单独删没问题，加了自重再删就留下指向已删杆件的荷载——
    校验在下一步才失败，而错误信息指向的不是删除那一步。
    """
    s = Session()
    s.define_materials_and_sections(MAT, SEC)
    g = s.generate_frame(spans=[6.0, 6.0], storeys=[3.6], bays=[6.0],
                         column_section="COLUMN", beam_section="BEAM",
                         material="Q355")
    s.set_load_cases(cases=[{"name": "D", "member_loads":
                             [{"member": m, "w": [0, 0, -20e3]}
                              for m in g.payload["beam_member_ids"]]}])
    assert s.add_self_weight().ok

    r = s.remove_members(ids=[1])
    assert r.ok, f"删除后模型不合法：{r.payload.get('errors')}"
    assert r.payload.get("pruned_loads"), "该报出清了哪些荷载"
    assert s.solve_model().ok, "清理干净的话应当还能求解"


def test_every_load_kind_is_pruned_on_delete():
    """四类荷载一处都不能漏，**而且顶层与 load_cases 两个位置都要清**。

    只清一处的话，用哪种写法建的模型会有不同的行为——
    而这种差异不报错，只在下一次校验时冒出来。
    """
    import changes

    model = {
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0}],
        "members": [],
        "nodal_loads": [{"node": 1, "load": [0, 0, -1, 0, 0, 0]},
                        {"node": 9, "load": [0, 0, -1, 0, 0, 0]}],
        "member_loads": [{"member": 1, "w": [0, 0, -1]},
                         {"member": 9, "w": [0, 0, -1]}],
        "member_spans": [{"member": 1, "kind": "uniform"},
                         {"member": 9, "kind": "uniform"}],
        "settlements": [{"node": 1, "d": [0, 0, -0.01, 0, 0, 0]},
                        {"node": 9, "d": [0, 0, -0.01, 0, 0, 0]}],
        "load_cases": [{"name": "D",
                        "nodal_loads": [{"node": 9, "load": [0, 0, -1, 0, 0, 0]}],
                        "member_loads": [{"member": 9, "w": [0, 0, -1]}],
                        "member_spans": [{"member": 9, "kind": "uniform"}],
                        "settlements": [{"node": 9, "d": [0, 0, -0.01, 0, 0, 0]}]}],
    }
    counted = changes.prune_loads(model, dropped_members={9}, dropped_nodes={9})
    for key, _ in changes.LOAD_KEYS:
        assert counted.get(key) == 2, f"{key} 没有在两个位置都清干净：{counted}"
        assert len(model[key]) == 1
        assert model["load_cases"][0][key] == []


def test_the_load_key_list_matches_the_schema():
    """`LOAD_KEYS` 是照 schema 列的。**加了新荷载类型而这里没跟上，
    删除清理就会漏它**——正是当初漏掉 member_spans 的那种漏法。"""
    import changes
    from model_io import MODEL_SCHEMA

    case_props = set(
        MODEL_SCHEMA["properties"]["load_cases"]["items"]["properties"])
    case_props.discard("name")
    assert {k for k, _ in changes.LOAD_KEYS} == case_props, (
        "changes.LOAD_KEYS 与 schema 里的荷载键对不上")


def test_units_round_trip_leaves_the_model_equivalent():
    """换算是**物理等价**的：来回一趟，报出来的 mm / kN 必须一模一样。"""
    s = Session()
    s.define_materials_and_sections(MAT, SEC)
    g = s.generate_frame(spans=[6.0], storeys=[3.6], column_section="COLUMN",
                         beam_section="BEAM", material="Q355")
    s.set_load_cases(cases=[{"name": "D", "member_loads":
                             [{"member": m, "w": [0, 0, -20e3]}
                              for m in g.payload["beam_member_ids"]]}])
    s.solve_model()
    before = s.query_results(what="max_deflection").payload["magnitude_mm"]

    s.set_units(units="N-mm-MPa")
    s.solve_model()
    middle = s.query_results(what="max_deflection").payload["magnitude_mm"]
    s.set_units(units="N-m-Pa")
    s.solve_model()
    after = s.query_results(what="max_deflection").payload["magnitude_mm"]

    assert middle == pytest.approx(before, rel=1e-9)
    assert after == pytest.approx(before, rel=1e-9)


def test_a_set_survives_an_unrelated_change():
    """改荷载不该动到命名集合——**集合是纯命名层**。"""
    s = Session()
    s.define_materials_and_sections(MAT, SEC)
    s.generate_frame(spans=[6.0, 6.0], storeys=[3.6], column_section="COLUMN",
                     beam_section="BEAM", material="Q355")
    s.define_set(name="全部柱", orientation="vertical")
    before = list(s.model["sets"]["全部柱"]["members"])
    s.set_load_cases(cases=[{"name": "D", "member_loads":
                             [{"member": 4, "w": [0, 0, -20e3]}]}])
    assert s.model["sets"]["全部柱"]["members"] == before
