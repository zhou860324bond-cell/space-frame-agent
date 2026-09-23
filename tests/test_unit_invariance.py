"""工具输出与单位制无关。

项目的约定是**报出去的数一律用固定的显示单位**：位移 mm、力 kN、弯矩
kN·m、应力 MPa。所以同一个结构，无论模型存成 N-m-Pa 还是 N-mm-MPa，
工具返回的数字必须一样。

这条一旦破了，破得很安静。实测过两处：

* ``strength.py`` 把应力硬编码成 ``/1e6`` 当 MPa。毫米制下应力本来就是
  MPa，再除一次把 150 MPa 报成 0.0 MPa——数看着"很安全"，结论恰恰相反；
* ``response_spectrum_analysis`` 的重力加速度默认 9.81。毫米制下该是
  9806.65，差 1000 倍，地震作用整体跟着小 1000 倍。

两处都不报错。所以这里不逐个去盯，而是把工具跑两遍、逐个数字对。
"""

import pytest

from agent import Session
from sections import i_section
from units import convert_model


def frame_session(units="N-m-Pa"):
    """一榀有压有弯的平面框架，强度、稳定、模态、反应谱都跑得起来。"""
    session = Session()
    session.add_nodes([[0, 0, 0], [0, 0, 4.0], [6.0, 0, 4.0], [6.0, 0, 0]])
    session.define_materials_and_sections(
        materials=[{"name": "Q355", "E": 2.06e11, "nu": 0.3, "density": 7850,
                    "yield_stress": 3.55e8, "allow_tension": 3.05e8,
                    "allow_compression": 3.05e8}],
        # 工字形由 sections.i_section 算出全套属性（含剪应力要的 S/b），
        # 折算应力校核才跑得起来——只给 A/I/J 的话它会明确弃权。
        sections=[i_section("S", 0.4, 0.2, 0.012, 0.008)])
    session.add_members([[1, 2], [2, 3], [3, 4]], "S", "Q355")
    session.set_supports([1], fix=[1, 1, 1, 1, 1, 1])
    session.set_supports([4], fix=[1, 1, 1, 1, 1, 1])
    session.set_member_load(2, [0.0, 0.0, -30e3])
    session.set_nodal_load(2, [40e3, 0.0, 0.0, 0.0, 0.0, 0.0])
    if units != "N-m-Pa":
        session.model = convert_model(session.model, units)
    return session


def numbers(payload, path=""):
    """把返回值里所有的数摊平成 {路径: 值}，字符串一并留着供比对。"""
    out = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            out.update(numbers(value, f"{path}.{key}"))
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            out.update(numbers(value, f"{path}[{index}]"))
    elif isinstance(payload, bool) or payload is None:
        out[path] = payload
    elif isinstance(payload, (int, float)):
        out[path] = float(payload)
    return out


#: 要两种单位制下对比的工具：(方法名, 参数)。
#: **新增分析工具就在这里加一行**，否则它的单位无关性没人看着。
TOOLS = [
    ("solve_model", {}),
    ("modal_analysis", {"num_modes": 6}),
    ("buckling_analysis", {"num_modes": 3}),
    ("check_strength", {}),
    ("response_spectrum_analysis",
     {"direction": "x", "num_modes": 10, "alpha_max": 0.08, "tg": 0.35}),
    # 查询类工具也在给用户报数字，而且各自做单位换算——
    # 求解对了不等于报出去的对。
    ("query_results", {"what": "max_displacement"}),
    ("query_results", {"what": "max_deflection"}),
    ("query_results", {"what": "reactions"}),
    ("query_results", {"what": "member_forces", "member_id": 2}),
    ("query_envelope", {"component": "Mz"}),
    ("query_diagram", {"component": "Mz"}),
]


def run(session, name, kwargs):
    if name != "solve_model":
        session.solve_model()
    return getattr(session, name)(**kwargs)


def _label(entry) -> str:
    name, kwargs = entry
    extra = kwargs.get("what") or kwargs.get("component")
    return f"{name}-{extra}" if extra else name


@pytest.mark.parametrize(("name", "kwargs"), TOOLS,
                         ids=[_label(entry) for entry in TOOLS])
def test_the_numbers_do_not_depend_on_the_unit_system(name, kwargs):
    """同一个结构，两种单位制，工具报出来的每个数都必须一样。"""
    si = run(frame_session("N-m-Pa"), name, kwargs)
    mm = run(frame_session("N-mm-MPa"), name, kwargs)
    assert si.ok == mm.ok, f"{name} 在两种单位制下成败不同：{si.payload} / {mm.payload}"
    if not si.ok:
        pytest.skip(f"{name} 在这个模型上跑不起来：{si.payload}")

    left, right = numbers(si.payload), numbers(mm.payload)
    shared = set(left) & set(right)
    assert shared, f"{name} 的返回值里没有数字可比"
    differing = {
        key: (left[key], right[key]) for key in sorted(shared)
        if not _same(left[key], right[key])
    }
    assert not differing, (
        f"{name} 的这些数随单位制变了：{differing}。"
        "报出去的值应当一律用固定显示单位（mm / kN / kN·m / MPa），"
        "不随模型存成哪套单位而变。")


def _same(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        return a == b
    return a == pytest.approx(b, rel=1e-6, abs=1e-9)


def test_the_tool_list_covers_the_analysis_tools():
    """**新增分析工具要在 TOOLS 里加一行**，否则这道闸门管不到它。"""
    covered = {name for name, _ in TOOLS}
    expected = {"solve_model", "modal_analysis", "buckling_analysis",
                "check_strength", "response_spectrum_analysis",
                "query_results", "query_envelope", "query_diagram"}
    assert expected <= covered, f"这些分析工具没进闸门：{sorted(expected - covered)}"


def test_the_spectrum_uses_the_gravity_of_the_unit_system():
    """GB 50011 的 α 是无量纲的地震影响系数，要乘 g 才是加速度。

    g 硬编码成 9.81 的话，毫米制下地震作用整体小 1000 倍——而结果看着
    完全正常，只是"这栋楼地震下很安全"。
    """
    si = frame_session("N-m-Pa").response_spectrum_analysis(
        direction="x", num_modes=10, alpha_max=0.08, tg=0.35)
    mm = frame_session("N-mm-MPa").response_spectrum_analysis(
        direction="x", num_modes=10, alpha_max=0.08, tg=0.35)
    assert si.ok and mm.ok
    assert mm.payload["base_shear_kN"] == pytest.approx(
        si.payload["base_shear_kN"], rel=1e-9)
    assert si.payload["base_shear_kN"] > 0
