"""界面验证：确认它只是一层壳。

界面上的数字必须和直接调 Session 算出来的完全一致——一旦出现"界面专用"的
计算分支，演示给出的结论就和评测集里的不是同一份代码算的了。
streamlit 没装就跳过，不拖累核心测试。
"""
import json
from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit", reason="未安装 streamlit，跳过界面测试")
from streamlit.testing.v1 import AppTest   # noqa: E402

import abaqus_backend                          # noqa: E402
import model_tables as MT                  # noqa: E402
import sections as S                       # noqa: E402
from agent import Session                  # noqa: E402

APP = Path(__file__).resolve().parent.parent / "gui_app.py"
MATERIALS = [{"name": "STEEL", "E": 2.1e11, "nu": 0.3}]

# 界面默认用工字形按尺寸算特性，比对基准必须走同一条路，
# 否则改了界面默认值这个测试就会假红。
_I_KIND = "工字形 / H 型钢"
_BUILD, _KEYS, _LABELS = S.BUILDERS[_I_KIND]
SECTIONS = [_BUILD("COLUMN", *S.DEFAULT_DIMENSIONS[_I_KIND]),
            _BUILD("BEAM", *S.DEFAULT_DIMENSIONS[_I_KIND])]


def _click(at, label):
    return next(b for b in at.button if b.label == label).click().run()


def _generate(at):
    """点「生成并求解」。表单从侧边栏搬进了前处理页，所以按标题找而不是按位置。"""
    return _click(at, "生成并求解")


@pytest.fixture(scope="module")
def solved_app():
    at = AppTest.from_file(str(APP), default_timeout=300).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    at = _generate(at)
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


@pytest.fixture(scope="module")
def headless_reference():
    """同样的参数，绕开界面直接算一遍，作为比对基准。"""
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    beams = s.generate_frame(spans=[6, 6, 6], storeys=[3.6, 3.6], bays=[6],
                             column_section="COLUMN", beam_section="BEAM",
                             material="STEEL", base="fixed").payload["beam_member_ids"]
    s.set_load_cases(cases=[{"name": "DL", "member_loads":
                             [{"member": m, "w": [0, 0, -20e3]} for m in beams]}])
    s.solve_model()
    return s


def test_first_render_points_at_the_first_stage():
    """空手进来时，界面必须指明从哪个阶段开始 —— 三个阶段并排时尤其要紧。"""
    at = AppTest.from_file(str(APP), default_timeout=300).run()
    assert not at.exception
    assert any("前处理" in i.value for i in at.info)


def test_metrics_match_the_headless_computation(solved_app, headless_reference):
    metrics = {m.label: m.value for m in solved_app.metric}
    ref_disp = headless_reference.query_results(what="max_displacement",
                                                case="DL").payload
    ref_reac = headless_reference.query_results(what="reactions", case="DL").payload

    assert metrics["节点 / 杆件"] == (f"{len(headless_reference.frame.nodes)} / "
                                     f"{len(headless_reference.frame.members)}")
    assert metrics["最大合位移"] == f"{ref_disp['magnitude_mm']:.4f} mm"
    assert metrics["竖向反力合计"] == f"{ref_reac['vertical_total_kN']:.2f} kN"
    assert metrics["静力平衡"] == "通过"


def test_reaction_total_matches_the_applied_load(solved_app):
    """20 根梁 × 6 m × 20 kN/m = 2400 kN，界面上必须是这个数。"""
    metrics = {m.label: m.value for m in solved_app.metric}
    assert metrics["竖向反力合计"] == "2400.00 kN"


def test_the_interface_is_organised_into_three_stages(solved_app):
    """顶层只有前处理 / 分析 / 后处理三个阶段，其余都是二级页。

    早先六个平级标签把"建模的"和"看结果的"混在一层。这条测试固定住新的层级，
    以后再加功能得挂到某个阶段下面，而不是又在顶层多一个标签。
    """
    labels = [t.label for t in solved_app.tabs]
    assert labels == [
        "① 前处理 · 建模",
        "参数化建模", "自然语言建模", "手绘草图", "模型视图", "建模过程",
        "模型编辑", "模型 JSON",
        "② 分析 · 求解与校核",
        "求解状态", "求解器对比", "截面优化",
        "③ 后处理 · 结果",
        "三维视图", "内力图", "数值结果",
    ]


def test_the_parameter_form_moved_out_of_the_sidebar(solved_app):
    """侧边栏清空了 —— 页面宽出一大截，三维视图和表格才好读。"""
    assert not solved_app.sidebar.button
    assert not solved_app.sidebar.text_input
    assert not solved_app.sidebar.number_input


def test_the_status_bar_stays_visible_across_stages(solved_app):
    """节点数、位移、反力、平衡状态是切到哪个阶段都要看见的东西，
    所以它在标签页之外、页面顶部。"""
    metrics = {m.label for m in solved_app.metric}
    assert {"节点 / 杆件", "最大合位移", "竖向反力合计", "静力平衡"} <= metrics


def test_comparison_tab_degrades_gracefully_without_abaqus(solved_app):
    """本机没有 Abaqus 时，这一页要说清楚，而不是报错或假装能跑。

    CI 里通常没有 Abaqus，所以这条降级路径正是 CI 唯一跑得到的那条。
    但开发机可能装了 Abaqus——那种情况下这条降级路径根本不会触发，
    应该 skip 而不是假红。
    """
    if abaqus_backend.abaqus_available():
        pytest.skip("本机装了 Abaqus，降级路径不会触发，跳过此测试")
    assert "没有找到 Abaqus" in " ".join(str(i.value) for i in solved_app.info)
    assert not any(b.label == "用 Abaqus 再算一遍" and not b.disabled
                   for b in solved_app.button)


def test_reaction_and_member_tables_are_rendered(solved_app):
    """3 张结果表 + 7 张可编辑表 + 分析页两张表 + 建模过程步骤表 + 静默检测结果表。"""
    assert len(solved_app.dataframe) == 14


# ------------------------------------------------------------ 模型编辑
#
# AppTest（streamlit 1.62）没有 data_editor 访问器，编辑框在测试树里就是一张
# 普通 dataframe，没法"打字进去"。所以这里改成把内容塞进 session_state 的
# edit_tables —— 界面本来就是从它取初值来铺表格的，走的是同一条路径。
# 表格 ↔ 模型的转换本身由 test_model_tables.py 逐条验，这里只验界面那一段接线。

def _fresh_app():
    at = AppTest.from_file(str(APP), default_timeout=300).run()
    at = _generate(at)
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


def _table(at, columns):
    """按列名认出是哪张表 —— 测试树里拿不到 widget 的 key。"""
    want = list(columns)
    hits = [d.value for d in at.dataframe if list(d.value.columns) == want]
    assert len(hits) == 1, f"期望正好一张 {want} 表，实得 {len(hits)} 张"
    return hits[0]


def _apply(at):
    return _click(at, "应用修改并重算")


def test_every_model_part_gets_an_editable_table(solved_app):
    for columns in (MT.NODE_COLUMNS, MT.MEMBER_COLUMNS, MT.SUPPORT_COLUMNS,
                    MT.NODAL_LOAD_COLUMNS, MT.MEMBER_LOAD_COLUMNS,
                    MT.SPAN_LOAD_COLUMNS, MT.SETTLEMENT_COLUMNS):
        _table(solved_app, columns)


def test_the_editor_is_seeded_from_the_solved_model(solved_app):
    fem = solved_app.session_state.fem
    assert len(_table(solved_app, MT.NODE_COLUMNS)) == len(fem.model["nodes"])
    assert len(_table(solved_app, MT.MEMBER_COLUMNS)) == len(fem.model["members"])
    assert len(_table(solved_app, MT.SUPPORT_COLUMNS)) == len(fem.model["supports"])


def test_applying_untouched_tables_changes_nothing():
    """什么都不改点一下"应用"，模型和结果必须逐位不变。

    这条守的是往返本身：只要表格→模型这一步丢了任何东西，这里立刻就红。
    """
    at = _fresh_app()
    before_model = json.loads(json.dumps(at.session_state.fem.model))
    before = at.session_state.solve_result.payload["cases"]["DL"]
    at = _apply(at)
    assert not at.exception, [str(e.value) for e in at.exception]
    assert not at.error, [str(e.value) for e in at.error]
    after = at.session_state.solve_result.payload["cases"]["DL"]
    assert after["max_displacement_mm"] == before["max_displacement_mm"]
    assert at.session_state.fem.model["members"] == before_model["members"]
    assert at.session_state.fem.model["supports"] == before_model["supports"]


def test_an_edited_coordinate_reaches_the_solver():
    """把最高的一层再抬高 2 m，位移必须跟着变 —— 编辑要真的走到求解器。"""
    at = _fresh_app()
    before = at.session_state.solve_result.payload["cases"]["DL"]["max_displacement_mm"]
    tables = MT.to_tables(at.session_state.fem.model)
    top = max(r["z"] for r in tables["nodes"])
    for row in tables["nodes"]:
        if row["z"] == top:
            row["z"] = top + 2.0
    at.session_state["edit_tables"] = tables
    at = _apply(at.run())

    assert not at.exception, [str(e.value) for e in at.exception]
    assert not at.error, [str(e.value) for e in at.error]
    assert max(n["z"] for n in at.session_state.fem.model["nodes"]) == top + 2.0
    after = at.session_state.solve_result.payload["cases"]["DL"]["max_displacement_mm"]
    assert after != before, "几何变了结果却没变，说明编辑根本没走到求解器"


def test_an_illegal_edit_is_refused_and_the_old_model_survives():
    """只留一根杆件会让模型不成立。界面必须报错并保住原模型，
    而不是把自己改成一个算不出来的状态。"""
    at = _fresh_app()
    kept = json.loads(json.dumps(at.session_state.fem.model))
    tables = MT.to_tables(kept)
    tables["members"] = tables["members"][:1]
    at.session_state["edit_tables"] = tables
    at = _apply(at.run())

    assert not at.exception, [str(e.value) for e in at.exception]
    assert at.error, "应当把校验错误显示出来"
    assert at.session_state.fem.model["members"] == kept["members"]


def test_a_bad_cell_is_reported_with_its_row():
    """单元格里打了字，要说清是哪张表哪一行，而不是抛异常。"""
    at = _fresh_app()
    tables = MT.to_tables(at.session_state.fem.model)
    tables["nodes"][1]["x"] = "六米"
    at.session_state["edit_tables"] = tables
    at = _apply(at.run())

    assert not at.exception, [str(e.value) for e in at.exception]
    messages = [str(e.value) for e in at.error]
    assert any("第 2 行" in m and "节点" in m for m in messages), messages


def test_discarding_edits_restores_the_tables():
    at = _fresh_app()
    tables = MT.to_tables(at.session_state.fem.model)
    tables["nodes"][0]["x"] = 99.0
    at.session_state["edit_tables"] = tables
    at = _click(at.run(), "放弃修改")

    assert not at.exception
    assert _table(at, MT.NODE_COLUMNS).loc[0, "x"] != 99.0


def test_regenerating_clears_the_editor_state():
    """重新生成之后，编辑区不能还留着上一个模型的行。"""
    at = _fresh_app()
    tables = MT.to_tables(at.session_state.fem.model)
    tables["nodes"] = tables["nodes"][:2]
    at.session_state["edit_tables"] = tables
    at.run()
    at.text_input[0].set_value("6, 6, 6, 6").run()
    at = _generate(at)

    assert not at.exception
    assert len(_table(at, MT.NODE_COLUMNS)) == len(at.session_state.fem.model["nodes"])


def test_bad_geometry_is_reported_not_raised():
    at = AppTest.from_file(str(APP), default_timeout=300).run()
    at.text_input[0].set_value("6, -1").run()
    at = _generate(at)
    assert not at.exception, "参数非法应当在界面上报错，而不是抛异常"
    assert at.error, "应当显示一条错误提示"


def test_a_failed_generation_keeps_the_model_you_already_had():
    """生成失败还把手上能算的模型清掉，是最让人恼火的一种行为。"""
    at = _fresh_app()
    kept = at.session_state.fem.model["nodes"]
    at.text_input[0].set_value("6, -1").run()
    at = _generate(at)
    assert at.error
    assert at.session_state.fem.model["nodes"] == kept


def test_switching_case_updates_every_number_in_the_status_bar():
    """切换工况时，指标卡和徽章必须整体换成新工况的数。

    先取数再选工况的写法，会让那一次刷新里一半是新工况、一半还是旧的 ——
    页面上看不出异常，数却对不上。
    """
    at = _fresh_app()
    fem = at.session_state.fem
    tables = MT.to_tables(fem.model)
    # 复制一份荷载到第二个工况，并把它放大一倍
    extra = [{**r, "case": "DL2", "wz": r["wz"] * 2} for r in tables["member_loads"]]
    tables["member_loads"] = tables["member_loads"] + extra
    at.session_state["edit_tables"] = tables
    at = _apply(at.run())
    assert not at.exception, [str(e.value) for e in at.exception]
    assert not at.error, [str(e.value) for e in at.error]

    picker = next(s for s in at.selectbox if s.label == "工况 / 组合")
    first = {m.label: m.value for m in at.metric}["最大合位移"]
    at = picker.set_value("DL2").run()
    second = {m.label: m.value for m in at.metric}["最大合位移"]
    assert second != first
    # 反力也必须跟着换成新工况的，不能只换位移
    assert float({m.label: m.value for m in at.metric}["竖向反力合计"].split()[0]) \
        == pytest.approx(2 * 2400.0, rel=1e-6)


def test_the_analysis_stage_shows_a_governing_combination_table(solved_app):
    """控制组合表是设计时真正要看的那张——它必须在分析页上，而不是藏在别处。"""
    columns = [list(d.value.columns) for d in solved_app.dataframe]
    assert any("控制组合" in cols and "分量" in cols for cols in columns), columns


def test_the_envelope_checkbox_exists_and_is_off_by_default():
    """默认不画包络：单工况模型下包络就等于它自己，先给人看单工况更直白。"""
    at = _fresh_app()
    box = next(c for c in at.checkbox if c.label == "画各组合包络")
    assert box.value is False


def test_turning_on_the_envelope_does_not_break_a_single_case_model():
    """只有一个工况时包络退化成它自己 —— 不能报错，也不能画不出来。"""
    at = _fresh_app()
    next(c for c in at.checkbox if c.label == "画各组合包络").set_value(True).run()
    at = next(s for s in at.selectbox if s.label == "查看单根杆件").set_value(1).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert not at.error, [str(e.value) for e in at.error]


# ------------------------------------------------------------ 模型视图与符号

def test_the_model_view_works_before_solving():
    """模型视图不该依赖求解结果——"先看一眼再算"就是它存在的理由。

    荷载方向加反是最常见也最难从数字上察觉的错误：2400 kN 的反力合计
    看着完全正常，直到你发现荷载是朝上的。
    """
    from plot3d_interactive import figure_model

    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    g = s.generate_frame(spans=[6, 6], storeys=[3.6], column_section="COLUMN",
                         beam_section="BEAM", material="STEEL")
    s.set_load_cases(cases=[{"name": "DL", "member_loads":
                             [{"member": m, "w": [0, 0, -20e3]}
                              for m in g.payload["beam_member_ids"]]}])
    assert s.solution is None, "还没求解"
    assert s.frame is None, "frame 也要到求解时才装配"
    fig, info = figure_model(s.preview_frame(), "DL")
    assert info["loads"]["member"] > 0
    assert info["supports"]["drawn"] > 0


def test_support_symbols_skip_the_automatic_out_of_plane_restraints():
    """平面刚架的面外约束是代码自动加在**每个**节点上的。

    全画出来满屏都是符号，真正的柱脚反而看不见了。只画约束住两个及以上
    平动方向的节点，其余在图注里报个数。
    """
    import viz_symbols as VS

    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    s.generate_frame(spans=[6, 6], storeys=[3.6], column_section="COLUMN",
                     beam_section="BEAM", material="STEEL")   # 不给 bays = 平面
    _, meta = VS.support_traces(s.preview_frame())
    assert meta["drawn"] == 3, "只有三个柱脚是真支座"
    assert meta["skipped"] > 0, "其余节点只有面外约束，不该画"


def test_support_symbols_are_classified():
    import viz_symbols as VS

    assert VS.classify_support((1, 1, 1, 1, 1, 1)) == "固接"
    assert VS.classify_support((1, 1, 1, 0, 0, 0)) == "铰接"
    assert VS.classify_support((1, 1, 0, 0, 0, 0)) == "部分约束"


def test_load_arrows_point_the_same_way_as_the_load():
    """箭头方向必须跟荷载一致。反了的话这张图非但没用，还会误导。"""
    import numpy as np

    import viz_symbols as VS

    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    g = s.generate_frame(spans=[6.0], storeys=[3.6], column_section="COLUMN",
                         beam_section="BEAM", material="STEEL")
    s.set_load_cases(cases=[{"name": "DL", "member_loads":
                             [{"member": m, "w": [0, 0, -20e3]}
                              for m in g.payload["beam_member_ids"]]}])
    traces, _ = VS.load_traces(s.preview_frame(), "DL")
    span = next(t for t in traces if t.name == "杆间荷载")
    assert np.all(np.asarray(span.w) < 0), "向下的荷载，箭头也得朝下"


def test_the_two_load_groups_are_scaled_independently():
    """节点荷载与杆间荷载各自归一。共用一把尺子的话，量级小的那类会消失——
    而"看不见"和"不存在"在图上分不出来。"""
    import numpy as np

    import viz_symbols as VS

    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    g = s.generate_frame(spans=[6.0], storeys=[3.6], column_section="COLUMN",
                         beam_section="BEAM", material="STEEL")
    beams = g.payload["beam_member_ids"]
    s.set_load_cases(cases=[{
        "name": "DL",
        "member_loads": [{"member": m, "w": [0, 0, -20e3]} for m in beams],
        "nodal_loads": [{"node": 3, "load": [0, 0, -1.0, 0, 0, 0]}]}])
    traces, _ = VS.load_traces(s.preview_frame(), "DL")
    nodal = next(t for t in traces if t.name == "节点荷载")
    span = next(t for t in traces if t.name == "杆间荷载")
    # 1 N 的节点力比 20 kN/m 的线荷载小四个数量级，但箭头必须还看得见
    assert np.abs(np.asarray(nodal.w)).max() > 0.2 * np.abs(np.asarray(span.w)).max()


def test_the_deformed_view_can_hide_the_undeformed_outline(solved_app):
    box = next(c for c in solved_app.checkbox if c.label == "叠加未变形")
    assert box.value is True, "默认叠加——不然读不出变形了多少"


def test_the_model_tree_lists_every_part(solved_app):
    """模型树要一眼看清模型有什么。翻七张表太慢。"""
    labels = [e.label for e in solved_app.expander]
    for want in ("材料", "截面", "几何", "约束", "荷载工况"):
        assert any(want in lab for lab in labels), (want, labels)


def test_preview_frame_does_not_need_a_solve():
    """看模型不该要求先求解——荷载加反、支座漏了，正是该在算之前看出来的。"""
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    s.generate_frame(spans=[6.0], storeys=[3.6], column_section="COLUMN",
                     beam_section="BEAM", material="STEEL")
    assert s.frame is None and s.solution is None
    frame = s.preview_frame()
    assert len(frame.nodes) == len(s.model["nodes"])
    assert s.frame is None, "预览不该把 frame 留下，否则会掩盖忘了求解的问题"


def test_preview_frame_refuses_an_invalid_model():
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    with pytest.raises(ValueError, match="不合法"):
        s.preview_frame()


# ------------------------------------------------------------ 建模过程

def test_the_build_history_records_the_parametric_path(solved_app):
    """界面走的是**直接调用**（fem.generate_frame(...)），不走 dispatch。

    第一版只在 dispatch 里记账，于是参数化建模这条路径的时间轴永远是空的——
    而那恰恰是用户最常看的一条。这条测试盯住这个。
    """
    hist = solved_app.session_state.fem.history
    assert len(hist) >= 3, "至少：定材料截面、生成拓扑、设荷载"
    tools = [s.tool for s in hist.steps]
    assert "define_materials_and_sections" in tools
    assert "generate_frame" in tools
    assert "set_load_cases" in tools


def test_queries_and_solves_do_not_pollute_the_timeline():
    """查十次结果不该多出十条一模一样的快照——时间轴会全是噪声。"""
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    s.generate_frame(spans=[6.0], storeys=[3.6], column_section="COLUMN",
                     beam_section="BEAM", material="STEEL")
    s.set_load_cases(cases=[{"name": "DL", "member_loads":
                             [{"member": 3, "w": [0, 0, -20e3]}]}])
    before = len(s.history)
    s.solve_model()
    for _ in range(3):
        s.query_results(what="max_deflection", case="DL")
        s.query_envelope(component="Mz")
    assert len(s.history) == before


def test_each_step_carries_a_readable_summary_not_json():
    """看时间轴是为了快速理解"这步干了什么"，不是为了读 JSON。"""
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    s.generate_frame(spans=[6.0, 6.0], storeys=[3.6, 3.6], bays=[6.0],
                     column_section="COLUMN", beam_section="BEAM",
                     material="STEEL", base="pinned")
    summary = s.history[1].summary
    assert "2 跨 2 层" in summary and "1 开间" in summary
    assert "柱底铰接" in summary
    assert "{" not in summary, "别把参数原样倒出来"


def test_a_keyword_only_tool_still_gets_its_arguments_into_the_summary():
    """generate_portal_frame 的签名是 (self, **kwargs)，bind 会把参数整包
    塞进 arguments["kwargs"]。不展平的话摘要会变成"0 跨，檐口 None m"。"""
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    s.generate_portal_frame(spans=[24.0], eave_height=7.5, ridge_rise=1.2,
                            column_section="COLUMN", rafter_section="BEAM",
                            material="STEEL", base="pinned")
    summary = s.history[1].summary
    assert "1 跨" in summary and "7.5" in summary and "1.2" in summary
    assert "None" not in summary


def test_a_snapshot_replays_to_exactly_that_model():
    """回看第 k 步要能重建出当时的模型——否则时间轴只是张图片。"""
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    s.generate_frame(spans=[6.0], storeys=[3.6], column_section="COLUMN",
                     beam_section="BEAM", material="STEEL")
    after_geometry = len(s.model["nodes"])
    s.set_load_cases(cases=[{"name": "DL", "member_loads":
                             [{"member": 3, "w": [0, 0, -20e3]}]}])

    snap = s.history[1].model
    assert len(snap["nodes"]) == after_geometry
    assert not snap.get("load_cases"), "第 2 步时还没有荷载"
    assert s.history[2].model.get("load_cases"), "第 3 步才有"

    replay = Session()
    replay.model = snap
    frame = replay.preview_frame()
    assert len(frame.nodes) == after_geometry


def test_a_snapshot_is_a_copy_not_a_reference():
    """快照必须是深拷贝。存引用的话后面每改一次模型，
    整条时间轴的历史都会跟着变成最新状态——回看就没意义了。"""
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    s.generate_frame(spans=[6.0], storeys=[3.6], column_section="COLUMN",
                     beam_section="BEAM", material="STEEL")
    snapshot_nodes = len(s.history[1].model["nodes"])
    s.add_nodes(coordinates=[[99.0, 0.0, 0.0]])
    assert len(s.history[1].model["nodes"]) == snapshot_nodes


def test_the_delta_says_what_changed():
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    s.generate_frame(spans=[6.0], storeys=[3.6], column_section="COLUMN",
                     beam_section="BEAM", material="STEEL")
    assert "节点" in s.history[1].changed and "杆件" in s.history[1].changed
    s.set_load_cases(cases=[{"name": "DL", "member_loads":
                             [{"member": 3, "w": [0, 0, -20e3]}]}])
    assert "工况" in s.history[2].changed


def test_a_failed_call_is_recorded_too():
    """失败的那一步也要留在时间轴上——"我刚才试了什么、为什么没成"
    和"我做成了什么"一样重要。"""
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    s.set_model(model={"nodes": [], "members": [], "supports": []})
    assert len(s.history) == 2
    assert not s.history[1].ok
    assert s.history[1].error


def test_the_timeline_slider_replays_without_error(solved_app):
    box = next(sl for sl in solved_app.slider if sl.label == "回看到第几步")
    assert box.value == len(solved_app.session_state.fem.history)
