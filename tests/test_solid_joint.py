"""局部实体节点子模型：能离线验证的那一半。

这个模块分成两半：**建模型规格**（纯 Python，把梁模型的杆端力翻译成实体
子模型的边界条件）和**跑 Abaqus**（需要 6.14 安装）。后一半在 CI 里跑不了，
但前一半是全部数值语义所在——名义应力算错，后面再精确的实体分析也没用。
所以这里把前一半按解析解钉死。

热点应力外推和收敛判据同样是纯数学，一并在这里用构造数据验证：外推公式
要能从已知直线上还原出截距，收敛判据要能把"奇异发散"和"真收敛"分开。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import sections as S
import solid_joint as sj
from agent import Session


TUBE_D, TUBE_T = 0.219, 0.008
SPAN, HEIGHT, TIP_LOAD = 4.0, 3.0, 20000.0


def _tube_properties() -> tuple[float, float]:
    inner = TUBE_D - 2.0 * TUBE_T
    area = math.pi * (TUBE_D ** 2 - inner ** 2) / 4.0
    inertia = math.pi * (TUBE_D ** 4 - inner ** 4) / 64.0
    return area, inertia


def _l_joint(*, section=None, load_at_joint=False) -> Session:
    """柱 (1)->(2) 与梁 (2)->(3) 在节点 2 相交，梁端受竖向力。

    选这个算例是因为它的节点内力有闭式解：梁端弯矩 = P·L，柱端还多一个
    轴力 P，两者都能手算到底。
    """
    session = Session()
    assert session.add_nodes([[0, 0, 0], [0, 0, HEIGHT], [SPAN, 0, HEIGHT]]).ok
    assert session.define_materials_and_sections(
        [{"name": "Q355", "E": 2.06e11, "nu": 0.3}],
        [section or S.circular_tube("P219x8", TUBE_D, TUBE_T)]).ok
    name = (section or {}).get("name", "P219x8")
    assert session.add_members([[1, 2], [2, 3]], section=name, material="Q355").ok
    assert session.set_supports([1], fix=[1, 1, 1, 1, 1, 1]).ok
    assert session.set_load_cases([{"name": "LC1"}]).ok
    target = 2 if load_at_joint else 3
    assert session.set_nodal_load(
        target, [0, 0, -TIP_LOAD, 0, 0, 0], case_name="LC1").ok
    assert session.solve_model().ok
    return session


# ------------------------------------------------- 截面反演

def test_circular_dimensions_round_trip_through_A_and_I():
    """由 A 与 I 反演外径内径必须能原样还原——这是整个实体几何的地基。"""
    section = S.circular_tube("P219x8", TUBE_D, TUBE_T)
    outer, inner = sj.infer_circular_dimensions(section)
    assert outer == pytest.approx(TUBE_D, rel=1e-12)
    assert inner == pytest.approx(TUBE_D - 2.0 * TUBE_T, rel=1e-12)


def test_solid_circle_inverts_to_zero_inner_diameter():
    outer, inner = sj.infer_circular_dimensions(S.solid_circle("R100", 0.1))
    assert outer == pytest.approx(0.1, rel=1e-12)
    assert inner == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------- 规格的数值语义

def test_nominal_stress_matches_the_closed_form_bending_stress():
    """名义正应力必须等于 M·c/I，不是"差不多"。

    这条是整个模块最值钱的一行断言：Kt 的分母就是它，分母错了，Kt 就是
    一个漂亮但无意义的数。
    """
    session = _l_joint()
    spec = sj.prepare_joint_spec(session, node_id=2)
    _area, inertia = _tube_properties()
    expected_mpa = (TIP_LOAD * SPAN) * (TUBE_D / 2.0) / inertia / 1e6

    beam = next(arm for arm in spec.arms if arm.member_id == 2)
    assert beam.nominal_normal_mpa == pytest.approx(expected_mpa, rel=1e-9)
    assert spec.nominal_normal_mpa == pytest.approx(expected_mpa, rel=1e-9)


def test_the_column_arm_also_carries_the_axial_term():
    """柱臂比梁臂多出的正应力应当正好是 P/A——弯矩两边相同，差的只有轴力。"""
    session = _l_joint()
    spec = sj.prepare_joint_spec(session, node_id=2)
    area, _inertia = _tube_properties()

    column = next(arm for arm in spec.arms if arm.member_id == 1)
    beam = next(arm for arm in spec.arms if arm.member_id == 2)
    difference = column.nominal_normal_mpa - beam.nominal_normal_mpa
    assert difference == pytest.approx(TIP_LOAD / area / 1e6, rel=1e-9)


def test_cut_face_forces_are_the_reaction_on_the_local_model():
    """切割面上的力是外部结构**作用于**子模型的力，方向与杆端力相反。

    符号搞反的话实体模型受力方向整个翻转，而应力云图看起来照样"合理"，
    这类错误不测出来就发现不了。
    """
    session = _l_joint()
    spec = sj.prepare_joint_spec(session, node_id=2)
    beam = next(arm for arm in spec.arms if arm.member_id == 2)

    # 梁把 20 kN 向下的力传给节点；节点域受到的竖向力应为向下 20 kN
    assert beam.force_n[2] == pytest.approx(-TIP_LOAD, rel=1e-6)
    # 梁端弯矩 P·L，换算成 N·mm
    assert max(abs(v) for v in beam.moment_nmm) == pytest.approx(
        TIP_LOAD * SPAN * 1000.0, rel=1e-6)


def test_the_anchor_arm_is_the_one_leading_to_a_support():
    """锚固臂要能复现地选中——否则同一个节点两次分析会得到不同的约束。"""
    session = _l_joint()
    spec = sj.prepare_joint_spec(session, node_id=2)
    assert spec.anchor_member == 1        # 柱通向固定支座


def test_arms_are_long_enough_to_keep_the_cut_face_away_from_the_joint():
    session = _l_joint()
    spec = sj.prepare_joint_spec(session, node_id=2)
    for arm in spec.arms:
        assert arm.length_mm >= 1.5 * arm.outer_diameter_mm


def test_tube_arms_report_a_wall_thickness_and_solid_ones_do_not():
    """壁厚是热点外推的参考长度。实心截面没有 t，就必须写 None，
    不能拿半径凑一个"看起来像 t"的数。"""
    session = _l_joint()
    tube = next(iter(sj.prepare_joint_spec(session, node_id=2).arms))
    assert tube.wall_thickness_mm == pytest.approx(TUBE_T * 1000.0, rel=1e-9)

    solid = _l_joint(section=S.solid_circle("R219", TUBE_D))
    arm = next(iter(sj.prepare_joint_spec(solid, node_id=2).arms))
    assert arm.wall_thickness_mm is None


def test_a_direct_nodal_load_is_reported_as_a_warning():
    """节点上直接挂力时，实体模型并不知道力在节点域里怎么扩散。
    这件事必须说出来，不能默默按截面力处理。"""
    session = _l_joint(load_at_joint=True)
    spec = sj.prepare_joint_spec(session, node_id=2)
    assert any("节点荷载" in text for text in spec.warnings)


# ------------------------------------------------- 拒绝路径

def test_an_unsolved_model_is_refused():
    session = Session()
    with pytest.raises(sj.SolidJointError, match="solve_model"):
        sj.prepare_joint_spec(session, node_id=1)


def test_a_node_with_a_single_member_is_refused():
    session = _l_joint()
    with pytest.raises(sj.SolidJointError, match="至少需要两根"):
        sj.prepare_joint_spec(session, node_id=3)


def test_an_unknown_node_is_refused():
    session = _l_joint()
    with pytest.raises(sj.SolidJointError, match="没有节点"):
        sj.prepare_joint_spec(session, node_id=99)


def test_an_unknown_case_is_refused():
    session = _l_joint()
    with pytest.raises(sj.SolidJointError, match="没有工况"):
        sj.prepare_joint_spec(session, node_id=2, case="不存在的工况")


def test_a_non_circular_section_is_refused():
    """工字形节点需要板件、焊缝、加劲肋几何；缺这些就自动拼实体，会得到
    "精确但不是用户结构"的答案。必须明确拒绝，不能猜。"""
    session = _l_joint(section=S.i_section("HW200", 0.2, 0.2, 0.008, 0.012))
    with pytest.raises(sj.SolidJointError):
        sj.prepare_joint_spec(session, node_id=2)


# ------------------------------------------------- 热点应力外推

def test_extrapolation_recovers_the_intercept_of_a_known_line():
    """构造一条已知直线，外推值必须精确等于它在焊趾处的截距。

    这是外推公式本身的验证：如果表面应力沿路径就是线性的，那么
    5/3·σ(0.4t) − 2/3·σ(1.0t) 必须原样还原 σ(0)。
    """
    thickness = 8.0
    intercept, slope = 300.0, -20.0        # σ(x) = 300 - 20x
    samples = [(x, intercept + slope * x) for x in (0.0, 2.0, 4.0, 6.0, 8.0)]
    got = sj.hot_spot_stress_linear(samples, thickness)
    assert got["hot_spot_mpa"] == pytest.approx(intercept, rel=1e-12)
    assert got["sigma_at_0_4t_mpa"] == pytest.approx(
        intercept + slope * 0.4 * thickness, rel=1e-12)


def test_extrapolation_ignores_the_singular_spike_next_to_the_toe():
    """焊趾附近的尖峰不参与外推——这正是外推法存在的理由。

    构造：0.4t 之外是干净的直线，0.4t 以内插进一个高得离谱的奇异峰值。
    外推值必须仍然等于直线的截距，完全不受尖峰影响。
    """
    thickness = 10.0
    line = [(x, 200.0 - 10.0 * x) for x in (4.0, 6.0, 8.0, 10.0, 12.0)]
    spike = [(0.0, 9999.0), (1.0, 4000.0), (2.0, 1200.0)]
    got = sj.hot_spot_stress_linear(spike + line, thickness)
    assert got["hot_spot_mpa"] == pytest.approx(200.0, rel=1e-12)


def test_extrapolation_refuses_a_range_that_does_not_cover_the_two_points():
    """只覆盖半边就外推，等于把噪声乘上放大系数。宁可不给数。"""
    with pytest.raises(sj.SolidJointError, match="没有覆盖外推区"):
        sj.hot_spot_stress_linear([(0.0, 300.0), (1.0, 290.0)], 10.0)


def test_extrapolation_refuses_a_section_without_a_wall_thickness():
    with pytest.raises(sj.SolidJointError, match="壁厚"):
        sj.hot_spot_stress_linear([(0.0, 1.0), (10.0, 2.0)], None)


# ------------------------------------------------- 收敛 / 发散判据

def _levels(sizes, values):
    return [{"mesh_size_mm": h, "max_abs_principal_mpa": v}
            for h, v in zip(sizes, values)]


def test_a_singular_peak_is_called_diverging_not_converged():
    """奇异点上 σ ~ h^(-λ)：网格越细峰值越高，逐次变化不缩小。

    旧判据只比最后两档、变化小于 5% 就报"已收敛"，正是这条让奇异问题被
    判成收敛、然后拿虚构的 Kt 去写报告。
    """
    sizes = [8.0, 4.0, 2.0, 1.0]
    values = [100.0 * (h ** -0.3) for h in sizes]   # 幂律发散
    got = sj.diagnose_peak_convergence(_levels(sizes, values))
    assert got["verdict"] == "diverging"
    assert got["log_log_slope"] == pytest.approx(-0.3, abs=1e-9)


def test_a_genuinely_converging_peak_is_called_converging():
    """真收敛：误差按网格尺寸平方衰减，逐次变化明显缩小。"""
    sizes = [8.0, 4.0, 2.0, 1.0]
    values = [250.0 - 0.5 * h ** 2 for h in sizes]
    got = sj.diagnose_peak_convergence(_levels(sizes, values))
    assert got["verdict"] == "converging"


def test_two_mesh_levels_cannot_decide_anything():
    """两档只能算出"最后一次变化有多大"，而奇异点上相邻两档的变化本来
    就可以很小。分不出来就必须说分不出来。"""
    got = sj.diagnose_peak_convergence(_levels([4.0, 2.0], [100.0, 101.0]))
    assert got["verdict"] == "inconclusive"
    assert "三档" in got["reason"]


def test_a_diverging_run_gets_no_stress_concentration_factor():
    """奇异几何下峰值除以名义应力不是应力集中系数，是网格的函数。

    这里直接验证判据的输出会让上层拒绝给 Kt——不是提醒，是拒绝。
    """
    sizes = [8.0, 4.0, 2.0, 1.0]
    values = [100.0 * (h ** -0.3) for h in sizes]
    got = sj.diagnose_peak_convergence(_levels(sizes, values))
    assert got["verdict"] != "converging"
    assert "奇异" in got["reason"]


# ------------------------------------------------- 生成的 Abaqus 脚本

def test_the_generated_script_is_valid_python_with_no_leftover_placeholders():
    """脚本是拼出来的字符串，写错了只有在装了 Abaqus 的机器上才会暴露。
    至少把"能不能解析"和"%% 有没有展开"这两件事在这里挡掉。"""
    session = _l_joint()
    spec = sj.prepare_joint_spec(session, node_id=2)
    text = sj._script_text(spec, [50.0, 35.0, 24.0], "out.json")

    compile(text, "generated_joint_script.py", "exec")
    assert "%%" not in text, "%% 没有展开，脚本里会留下字面量"
    assert "surface_samples" in text, "缺少表面取样，热点外推就没有输入"


def test_sampling_failure_is_contained_and_does_not_lose_the_solve():
    """取样是在"已经解完"的作业上做几何整理。它失败不该让整个作业作废。"""
    session = _l_joint()
    spec = sj.prepare_joint_spec(session, node_id=2)
    text = sj._script_text(spec, [50.0, 35.0, 24.0], "out.json")
    assert "except Exception as exc:" in text
    assert "surface_samples_error" in text


# ------------------------------------------------- Agent 工具入口

def test_the_tool_is_registered_and_reaches_the_session_method():
    """原来这个模块没有任何地方 import，功能写完了也够不着。
    工具表里有名字、Session 上有同名方法，dispatch 才走得通。"""
    from agent import TOOLS

    names = {entry["function"]["name"] for entry in TOOLS}
    assert "analyze_joint_solid" in names
    assert callable(getattr(Session, "analyze_joint_solid", None))


def test_the_tool_allows_one_native_mesh_but_explains_three_for_convergence():
    """自研后端允许先跑一档；是否足以判断收敛由结果明确写 inconclusive。"""
    from agent import TOOLS

    entry = next(e for e in TOOLS if e["function"]["name"] == "analyze_joint_solid")
    props = entry["function"]["parameters"]["properties"]
    assert props["mesh_sizes_mm"]["minItems"] == 1
    assert props["backend"]["enum"] == ["native", "abaqus"]


def test_native_hotspot_requires_three_stable_mesh_levels_before_publishing_kt():
    from native_joint import diagnose_hotspot_convergence

    stable = diagnose_hotspot_convergence([(16.0, 100.0), (12.0, 103.0),
                                            (8.0, 104.0)])
    assert stable["verdict"] == "converging"
    unstable = diagnose_hotspot_convergence([(16.0, 100.0), (12.0, 110.0),
                                              (8.0, 140.0)])
    assert unstable["verdict"] == "inconclusive"
    assert unstable["relative_changes"][1] > unstable["relative_changes"][0]


def test_native_hotspot_one_mesh_is_explicitly_inconclusive():
    from native_joint import diagnose_hotspot_convergence

    got = diagnose_hotspot_convergence([(8.0, 1200.0)])
    assert got["verdict"] == "inconclusive"
    assert "三档" in got["reason"]


def test_dry_run_returns_the_spec_without_needing_abaqus():
    """本机没装 Abaqus 时，这个工具仍应给出杆端力和名义应力——
    那是 Kt 的分母，能手算核对。"""
    session = _l_joint()
    result = session.analyze_joint_solid(node_id=2, dry_run=True)
    assert result.ok, result.payload
    _area, inertia = _tube_properties()
    expected = (TIP_LOAD * SPAN) * (TUBE_D / 2.0) / inertia / 1e6
    assert result.payload["nominal_normal_mpa"] == pytest.approx(expected, rel=1e-9)
    assert len(result.payload["arms"]) == 2


def test_the_tool_reports_refusals_as_a_failed_result_not_an_exception():
    """工具层不能把异常抛给对话循环——那会中断整轮，用户只看到一句报错。"""
    session = _l_joint()
    result = session.analyze_joint_solid(node_id=99, dry_run=True)
    assert not result.ok
    assert "没有节点" in result.payload["error"]


def test_dispatch_routes_the_tool_by_name():
    session = _l_joint()
    result = session.dispatch("analyze_joint_solid",
                              {"node_id": 2, "dry_run": True})
    assert result.ok, result.payload


def test_dry_run_payload_satisfies_the_declared_tool_contract():
    """契约表说这个工具会给 node_id 和 case，dry_run 就得真的给。

    契约测试里的共享模型用的是工字形截面，这个工具会拒绝它，所以那边跑不到；
    契约的实际校验落在这里。
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_tool_contracts import CONTRACTS

    required, _note = CONTRACTS["analyze_joint_solid"]
    payload = _l_joint().analyze_joint_solid(node_id=2, dry_run=True).payload
    missing = [key for key in required if key not in payload]
    assert not missing, f"缺少契约要求的键 {missing}；实际给的是 {sorted(payload)}"


def test_the_script_imports_every_module_its_api_calls_depend_on():
    """Abaqus 的成员是随模块导入才注册的，漏一个就是一次 AttributeError。

    第一次真跑就死在这上面：只 import 了 mesh 和 regionToolset，于是
    `mdb.Model()` 造出来的模型根本没有 fieldOutputRequests。而且后面还连着
    埋了 Coupling / EncastreBC / Job 好几个同类地雷——一次修一个要来回跑很多趟。

    这条测试把"用了哪个 API 就必须导哪个模块"钉死，靠的是文本检查，
    不需要装 Abaqus。
    """
    session = _l_joint()
    spec = sj.prepare_joint_spec(session, node_id=2)
    text = sj._script_text(spec, [50.0, 35.0, 24.0], "out.json")

    needs = {
        "FieldOutputRequest": "step",
        "StaticStep": "step",
        "Coupling": "interaction",
        "EncastreBC": "load",
        "ConcentratedForce": "load",
        "Moment": "load",
        "mdb.Job": "job",
        "ConstrainedSketch": "sketch",
        "ElemType": "mesh",
        "HomogeneousSolidSection": "section",
        "regionToolset.Region": "regionToolset",
    }
    imported = {line.strip() for line in text.splitlines()
                if line.startswith("import ") or line.startswith("from ")}
    blob = " ".join(imported)
    for api, module in sorted(needs.items()):
        if api in text:
            assert module in blob, f"脚本用了 {api}，却没 import {module}"


def test_the_script_does_not_edit_a_default_output_request_that_may_not_exist():
    """mdb.Model() 造出来的模型没有任何默认输出请求，
    去 setValues 一个 'F-Output-1' 就是在够一个不存在的东西。"""
    session = _l_joint()
    spec = sj.prepare_joint_spec(session, node_id=2)
    text = sj._script_text(spec, [50.0, 35.0, 24.0], "out.json")
    assert "fieldOutputRequests['F-Output-1']" not in text
    assert "FieldOutputRequest(name=" in text


def test_the_build_script_never_submits_the_job_itself():
    """从 CAE 里 job.submit() 时 standard.exe 不是 CAE 的子进程——Abaqus 驱动
    另起一个，这台机器需要的 MKL 开关传不过去，于是静默中止。

    所以建模脚本只写 .inp，求解由调用方直接 `abaqus job=` 驱动：那是实测
    跑通过的配置。这条防止有人图省事把 submit 加回来。
    """
    session = _l_joint()
    spec = sj.prepare_joint_spec(session, node_id=2)
    build = sj._script_text(spec, [8.0, 6.4, 5.2], "meta.json", phase="build")
    assert "job.writeInput" in build
    assert "job.submit" not in build, "建模阶段不该提交作业"
    assert "waitForCompletion" not in build


def test_the_two_phases_share_one_template():
    """两个阶段共用同一份模板。拆成两个文件之后 SPEC、坐标约定和那些辅助
    函数迟早会各改各的，而它们必须完全一致才对得上。"""
    session = _l_joint()
    spec = sj.prepare_joint_spec(session, node_id=2)
    build = sj._script_text(spec, [8.0, 6.4, 5.2], "a.json", phase="build")
    post = sj._script_text(spec, [8.0, 6.4, 5.2], "b.json", phase="post")
    for text in (build, post):
        compile(text, "joint_script.py", "exec")
        assert "def surface_samples" in text and "def build" in text
    assert "PHASE = 'build'" in build
    assert "PHASE = 'post'" in post


def test_the_runner_drives_the_solver_directly_and_checks_the_sta():
    """求解必须由 Python 直接调 `abaqus job=`，而且要亲自确认 .sta 里写着
    COMPLETED SUCCESSFULLY——作业中止时 Abaqus 的返回码并不总能反映出来。"""
    import inspect

    source = inspect.getsource(sj.run_joint_analysis)
    assert 'f"job={job}"' in source, "没有直接驱动求解器"
    assert '"interactive"' in source
    assert "COMPLETED SUCCESSFULLY" in source, "没有亲自校验 .sta"



# ------------------------------------------------- 网格尺寸

def test_mesh_reference_length_comes_from_the_wall_not_the_diameter():
    """第一次真跑就栽在这上面：默认档位按外径取，D=219 得到 55/37/24 mm，
    而壁厚只有 8 mm。60 mm 的四面体铺不出一层 8 mm 的壁。"""
    session = _l_joint()
    spec = sj.prepare_joint_spec(session, node_id=2)
    assert sj.mesh_reference_length(spec) == pytest.approx(TUBE_T * 1000.0, rel=1e-9)
    # 若按直径取会是 20 mm 以上，必须明显小于它
    assert sj.mesh_reference_length(spec) < spec.arms[0].outer_diameter_mm / 10.0


def test_a_solid_section_falls_back_to_a_fraction_of_the_diameter():
    """实心圆没有壁，也就没有"一个单元都放不下"的问题，
    但仍要有个有限的参考尺寸，不能返回 None 或 0。"""
    session = _l_joint(section=S.solid_circle("R219", TUBE_D))
    spec = sj.prepare_joint_spec(session, node_id=2)
    got = sj.mesh_reference_length(spec)
    assert got == pytest.approx(TUBE_D * 1000.0 / 10.0, rel=1e-9)


def test_the_reference_length_takes_the_thinnest_arm():
    """一根臂网格铺坏了整个模型就坏了，所以取最严的那个。"""
    thin = sj.JointArm(member_id=1, direction=(1.0, 0.0, 0.0),
                       outer_diameter_mm=219.0, inner_diameter_mm=211.0,
                       length_mm=400.0, force_n=(0.0, 0.0, 0.0),
                       moment_nmm=(0.0, 0.0, 0.0), nominal_normal_mpa=1.0,
                       wall_thickness_mm=4.0)
    thick = sj.JointArm(member_id=2, direction=(0.0, 0.0, 1.0),
                        outer_diameter_mm=219.0, inner_diameter_mm=195.0,
                        length_mm=400.0, force_n=(0.0, 0.0, 0.0),
                        moment_nmm=(0.0, 0.0, 0.0), nominal_normal_mpa=1.0,
                        wall_thickness_mm=12.0)
    spec = sj.JointSpec(node_id=1, case="LC1", material="Q355",
                        elastic_modulus_mpa=206000.0, poisson=0.3,
                        anchor_member=1, hotspot_radius_mm=300.0,
                        nominal_normal_mpa=1.0, arms=(thick, thin))
    assert sj.mesh_reference_length(spec) == pytest.approx(4.0)


def test_arms_stay_long_enough_after_the_default_was_shortened():
    """短臂默认从 3D 收到 2D 是为了控制单元数（按 h^-3 涨）。
    但下限不能破：切割面必须离节点足够远。"""
    session = _l_joint()
    spec = sj.prepare_joint_spec(session, node_id=2)
    for arm in spec.arms:
        assert arm.length_mm >= 1.5 * arm.outer_diameter_mm


def test_the_generated_script_does_not_keep_intersection_faces():
    """所有 cell 共用同一个截面属性，保留相贯内表面只会逼网格器去贴合
    相贯线，多铺一批薄片单元。"""
    session = _l_joint()
    spec = sj.prepare_joint_spec(session, node_id=2)
    text = sj._script_text(spec, [8.0, 6.4, 5.2], "out.json")
    assert "keepIntersections=OFF" in text


def test_a_mesh_coarser_than_the_wall_is_refused_before_abaqus_is_called():
    """壁厚方向连一个单元都放不下时，网格器只能铺退化四面体，
    求解器**直接崩而不是报错**——实测 1837 个单元里 725 个畸变。

    所以这道闸门要在调 Abaqus 之前就拦下来，别等它崩。
    """
    session = _l_joint()
    with pytest.raises(sj.SolidJointError, match="超过参考尺寸"):
        sj.run_joint_analysis(session, node_id=2,
                              mesh_sizes_mm=[60.0, 45.0, 35.0])


def test_the_refusal_names_the_arm_that_sets_the_limit():
    """报"太粗了"没用，得说是哪根杆件的哪个壁厚定的这个上限。"""
    session = _l_joint()
    with pytest.raises(sj.SolidJointError) as caught:
        sj.run_joint_analysis(session, node_id=2, mesh_sizes_mm=[60.0, 45.0, 35.0])
    message = str(caught.value)
    assert "最薄的管壁" in message and "t=8" in message


def test_the_default_sizes_pass_their_own_guard(monkeypatch):
    """默认档位不能被自己的闸门拦下来。

    这里没装 Abaqus，所以它最终会因为找不到 Abaqus 而失败——**恰恰是这个
    报错**证明网格闸门已经放行了。
    """
    session = _l_joint()
    import abaqus_backend
    monkeypatch.setattr(abaqus_backend, "find_abaqus", lambda: None)
    with pytest.raises(sj.SolidJointError, match="找不到 Abaqus"):
        sj.run_joint_analysis(session, node_id=2)


# ------------------------------------------------- 桌面入口与结果表

def test_desktop_registers_a_selected_node_entry_for_the_solid_model():
    """内核工具够不着就不是产品能力；桌面端必须有选中节点后的明确入口。"""
    from desktop import commands
    from desktop.main_window import MainWindow

    command = next(c for c in commands.COMMANDS if c.name == "solid_joint")
    assert command.handler == "run_solid_joint"
    assert MainWindow._NEEDS_SELECTION["solid_joint"] == ("node",)


def test_desktop_solid_result_is_a_mesh_convergence_table():
    from desktop import result_rows

    payload = {
        "node_id": 2, "case": "LC1", "nominal_normal_mpa": 296.4,
        "stress_concentration_factor": 10.15,
        "peak_convergence": {"verdict": "diverging"},
        "files": {"contour_png": "joint.png"},
        "meshes": [
            {"mesh_size_mm": 8.0, "nodes": 100, "elements": 50,
             "max_mises_mpa": 4000.0, "max_abs_principal_mpa": 4900.0,
             "p99_abs_principal_mpa": 1260.0},
            {"mesh_size_mm": 6.4, "nodes": 200, "elements": 120,
             "max_mises_mpa": 4600.0, "max_abs_principal_mpa": 5400.0,
             "p99_abs_principal_mpa": 1270.0},
        ],
    }
    title, columns, rows, locators = result_rows.to_rows("solid_joint", payload)
    assert "Kt=10.15" in title and "奇异发散" in title
    assert "C3D10 单元数" in columns and len(rows) == 2
    assert locators == [("node", 2), ("node", 2)]


def test_agent_routes_the_default_solid_backend_to_native(monkeypatch):
    import native_joint

    seen = {}

    def fake(session, node_id, case, anchor, sizes):
        seen["args"] = (node_id, case, anchor, sizes)
        return {"node_id": node_id, "case": case or "LC1", "backend": "native"}

    monkeypatch.setattr(native_joint, "run_native_joint_analysis", fake)
    result = _l_joint().analyze_joint_solid(2, mesh_sizes_mm=[20.0])
    assert result.ok and result.payload["backend"] == "native"
    assert seen["args"] == (2, None, None, [20.0])


def test_gmsh_generates_positive_c3d10_and_cut_faces_when_available():
    pytest.importorskip("gmsh")
    import native_joint

    session = _l_joint(section=S.solid_circle("R219", TUBE_D))
    spec = sj.prepare_joint_spec(session, 2)
    mesh, faces = native_joint.generate_joint_mesh(spec, 80.0)
    assert len(mesh.elements) > 0
    assert all(len(faces[arm.member_id]) >= 3 for arm in spec.arms)
    for conn in mesh.elements[:20]:
        _B, det = __import__("solid3d")._b_matrix(
            mesh.nodes[conn], np.full(4, 0.25))
        assert det > 0.0


def test_a_stable_but_wobbly_sequence_is_not_called_inconclusive():
    """非结构网格每档独立重剖分，已经收敛的问题峰值也会抖零点几个百分点。

    这三个数是带孔板的**实测值**（2.5 / 1.8 / 1.2 mm 三档）：明明稳得很，
    但"逐次变化逐档缩小"不成立——先降后升。判据只会报 inconclusive，
    于是一个已经验证过的算例反而拿不到结论。所以补了"全程稳定"这条通道。
    """
    got = sj.diagnose_peak_convergence(
        _levels([2.5, 1.8, 1.2], [2.614, 2.609, 2.620]))
    assert got["verdict"] == "converging"
    assert got["value_span"] < 0.01
    assert "散布" in got["reason"]


def test_never_call_it_converged_without_actually_refining_the_mesh():
    """**没真的细化过网格，就没有资格说收敛。**

    三档几乎一样粗的网格当然彼此接近，那只说明它们一样粗，不说明峰值有极限。
    这和"只比最后两档"是同一类错误的两个面：都是拿一个太弱的证据下结论。
    """
    got = sj.diagnose_peak_convergence(_levels([2.0, 1.9, 1.8],
                                               [100.0, 101.0, 100.5]))
    assert got["verdict"] == "inconclusive"
    assert "没有真正细化" in got["reason"]


def test_the_stable_channel_does_not_let_a_singular_peak_through():
    """新通道不能把奇异问题放进来。

    σ ~ h^(-λ) 在两倍细化下的散布远大于容差；而且纯幂律是单调上升、逐次变化
    不缩小的，发散判据先拦掉它。两道门都试一遍。
    """
    for lam in (0.1, 0.2, 0.3, 0.5):
        sizes = [8.0, 4.0, 2.0, 1.0]
        got = sj.diagnose_peak_convergence(
            _levels(sizes, [100.0 * h ** -lam for h in sizes]))
        assert got["verdict"] == "diverging", lam
