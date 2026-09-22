"""单位换算的字段覆盖闸门。

`convert_model` 换的是**数值**，物理量不变：同一个结构换算前后算出来的位移
和内力必须逐位相同。漏掉一个字段不会报错，模型校验照样通过——只是那一项
带着旧单位进了计算。实测过三处：

* ``member_spans.b``（部分跨荷载的终点）：a 换了 b 没换，毫米制下区间变成
  "从 1000 到 4"，悬臂挠度从 −12.47 mm 变成 +0.228 mm，**连符号都反**；
* ``members.connections``（半刚性连接刚度）：差 360 倍；
* ``steps.supports.spring``（分析步里改写的支座弹簧）：整个没换，而顶层
  那个换了——同一种量在两个地方，只换一处。

三处都是"加了新字段，换算这边没跟上"。所以这里不只是补上这三条测试，而是
**按 schema 逐个字段点名**：Schema 里新增一个数值字段而不在下面的表里分类，
第一条测试就会红并报出字段路径。
"""

import pytest

from model_io import MODEL_SCHEMA, validate_payload
from units import _factors, convert_model

#: Schema 里每个数值字段的量纲。键是字段路径，值是 `units._factors` 里的
#: 系数名，或 None 表示无量纲/不换算。
#:
#: **新增字段就要在这里加一行。** 不加的话下面第一条测试会指名道姓地红，
#: 而不是等到某个用毫米制的用户算出一个差几百倍的结果。
DIMENSIONS = {
    "units": None,
    "schema_version": None,
    "nodes.id": None,
    "nodes.x": "length", "nodes.y": "length", "nodes.z": "length",
    "materials.name": None,
    "materials.E": "modulus",
    "materials.nu": None,
    "materials.density": "density",
    "materials.yield_stress": "modulus",
    "materials.hardening_ratio": None,
    "materials.alpha": None,              # 1/℃，与长度力无关
    "materials.allow_tension": "modulus",
    "materials.allow_compression": "modulus",
    "sections.name": None,
    "sections.A": "area",
    "sections.Iy": "inertia", "sections.Iz": "inertia", "sections.J": "inertia",
    "sections.Ay": "area", "sections.Az": "area",
    "sections.cy": "length", "sections.cz": "length",
    "sections.circular": None,
    "sections.Sy": "first_moment", "sections.Sz": "first_moment",
    "sections.by": "length", "sections.bz": "length",
    "sections.S_flange": "first_moment", "sections.c_web": "length",
    "members.id": None, "members.i": None, "members.j": None,
    "members.section": None, "members.material": None,
    "members.ref_vector": None,           # 方向，只看比例
    "members.releases.i": None, "members.releases.j": None,
    "members.connections.i": "connection", "members.connections.j": "connection",
    "members.offset_i": "length", "members.offset_j": "length",
    "members.mu_y": None, "members.mu_z": None,
    "supports.name": None, "supports.node": None, "supports.fix": None,
    "supports.spring": "support_spring",
    "sets": None,
    "nodal_loads.name": None, "nodal_loads.node": None,
    "nodal_loads.load": "force_and_moment",
    "member_loads.name": None, "member_loads.member": None,
    "member_loads.w": "line_load",
    "member_loads.frame": None,           # 坐标系名，不是数
    "member_spans.name": None, "member_spans.member": None,
    "member_spans.kind": None, "member_spans.note": None,
    "member_spans.w1": "span_intensity", "member_spans.w2": "span_intensity",
    "member_spans.a": "length", "member_spans.b": "length",
    "settlements.name": None, "settlements.node": None,
    "settlements.d": "settlement",
    "member_strains.name": None, "member_strains.member": None,
    "member_strains.lack_of_fit": "length",
    "member_strains.delta_t": None,       # 温度
    "member_strains.gradient_t": None,    # 温度，折算成曲率时才带上长度
    "load_cases.name": None,
    "combos.name": None, "combos.factors": None,
    "amplitudes": None,                   # (伪时间, 系数) 都无量纲
    "steps.name": None, "steps.analysis": None, "steps.increments": None,
    "steps.loads": None, "steps.deactivate_loads": None,
    "steps.deactivate_supports": None,
    # 分析步里改写的支座：与顶层 supports 同一批字段，同一批量纲。
    "steps.supports": "support_entry",
}

#: 工况块的字段在顶层和 load_cases 里各出现一次，路径去掉前缀后是同一批。
_CASE_BLOCKS = ("nodal_loads", "member_loads", "member_spans",
                "settlements", "member_strains")


def _paths(schema, prefix=""):
    """走一遍 schema，给出每个叶子字段的路径。"""
    out = set()
    if not isinstance(schema, dict):
        return out
    if "properties" in schema:
        for name, child in schema["properties"].items():
            path = f"{prefix}{name}" if not prefix else f"{prefix}.{name}"
            grand = child.get("items", child)
            if ("properties" in grand
                    and not str(grand.get("type")) == "string"):
                out |= _paths(grand, path)
            else:
                out.add(path)
    if "additionalProperties" in schema and isinstance(
            schema["additionalProperties"], dict) and prefix:
        # "键是节点号/曲线名"这类映射按整块记一个路径：拆到里面去的话，
        # 路径里会出现一个随模型而变的键名，表就没法写了。
        out.add(prefix)
    return out


def _normalise(path: str) -> str:
    """load_cases.member_spans.a 与顶层的 member_spans.a 是同一个字段。"""
    for block in _CASE_BLOCKS:
        marker = f"load_cases.{block}."
        if path.startswith(marker):
            return path[len("load_cases."):]
    return path


def test_every_schema_field_is_classified():
    """**Schema 里新增一个字段，这里就必须给它一个量纲。**

    不分类的话，换算漏掉它不会有任何提示——漏过三次了，每次都是"加了新
    字段，units.py 这边没跟上"，而错误只在换过单位的模型上才现形。
    """
    found = {_normalise(p) for p in _paths(MODEL_SCHEMA)}
    found = {p for p in found if not p.startswith("load_cases.")
             or p == "load_cases.name"}
    missing = sorted(found - set(DIMENSIONS))
    assert not missing, (
        f"这些 schema 字段还没有分类量纲：{missing}。"
        "在 DIMENSIONS 里加一行：要换的给系数名，不换的写 None 并注明为什么。")


def test_no_stale_entries_in_the_table():
    """反过来：表里不该留着 schema 已经删掉的字段。"""
    found = {_normalise(p) for p in _paths(MODEL_SCHEMA)}
    stale = sorted(set(DIMENSIONS) - found)
    assert not stale, f"这些字段在 schema 里已经没有了：{stale}"


# --- 数值层：真的换了，而且换对了 -----------------------------------------

def kitchen_sink():
    """**用上每一个会换算的字段**的模型。漏用一个，下面的测试就测不到它。"""
    return {
        "schema_version": 1, "units": "N-m-Pa",
        "nodes": [{"id": 1, "x": 0.0, "y": 0.0, "z": 0.0},
                  {"id": 2, "x": 6.0, "y": 0.0, "z": 0.0}],
        "materials": [{"name": "M", "E": 2.1e11, "nu": 0.3, "density": 7850.0,
                       "yield_stress": 3.55e8, "alpha": 1.2e-5,
                       "allow_tension": 2.15e8, "allow_compression": 2.15e8}],
        "sections": [{"name": "S", "A": 0.02, "Iy": 2e-4, "Iz": 4e-4,
                      "J": 1e-5, "Ay": 0.012, "Az": 0.008,
                      "cy": 0.2, "cz": 0.1, "Sy": 1e-4, "Sz": 2e-4,
                      "by": 0.01, "bz": 0.008,
                      "S_flange": 1.5e-4, "c_web": 0.18}],
        "members": [{"id": 1, "i": 1, "j": 2, "section": "S", "material": "M",
                     "offset_i": [0.1, 0.0, 0.0], "offset_j": [0.0, 0.0, 0.05],
                     "connections": {"i": {"rz": 1e8, "ux": 5e7},
                                     "j": {"ry": 2e8}}}],
        # 平动刚接、转动弹性支承：fix 与 spring 不能撞在同一个自由度上，
        # 撞了会被语义校验挡住（刚性约束会让弹簧完全失效）。
        # rx 刚接（沉降要给在被约束的方向上），ry/rz 弹性支承。
        "supports": [{"node": 1, "fix": [1, 1, 1, 1, 0, 0],
                      "spring": [0.0, 0.0, 0.0, 0.0, 5e7, 6e7]}],
        "nodal_loads": [{"node": 2, "load": [1e3, 2e3, 3e3, 4e3, 5e3, 6e3]}],
        "member_loads": [{"member": 1, "w": [0.0, 0.0, -20e3]}],
        "member_spans": [
            {"member": 1, "kind": "partial", "w1": [0, 0, -15e3],
             "a": 1.0, "b": 4.0},
            {"member": 1, "kind": "point", "w1": [0, 0, -30e3], "a": 3.0},
            {"member": 1, "kind": "trapezoid", "w1": [0, 0, -5e3],
             "w2": [0, 0, -9e3]},
        ],
        "settlements": [{"node": 1, "d": [0.01, 0.0, 0.0, 0.001, 0.0, 0.0]}],
        "member_strains": [{"member": 1, "lack_of_fit": 0.002,
                            "delta_t": 30.0, "gradient_t": 15.0}],
        "steps": [{"name": "S1", "loads": {"default": "RAMP"},
                   "supports": {"2": {"fix": [0, 0, 1, 0, 0, 0],
                                      "spring": [0.0, 0.0, 5e6,
                                                 0.0, 0.0, 7e7]}}}],
        "amplitudes": {"c": [[0.0, 0.0], [1.0, 1.0]]},
    }


def _expected(name: str, index: int, factors: dict) -> float:
    """按量纲名给出某一分量该乘的系数。"""
    if name == "force_and_moment":
        return factors["force"] if index < 3 else factors["moment"]
    if name == "support_spring":
        return (factors["spring_translation"] if index < 3
                else factors["spring_rotation"])
    if name == "settlement":
        return factors["length"] if index < 3 else 1.0
    return factors[name]


def test_the_round_trip_restores_every_value():
    """换过去再换回来，每一个数都要回到原处。

    这条不挑字段：只要 convert_model 对某个字段做了不对称的处理（换一半、
    换错方向），来回一趟就对不上。
    """
    original = kitchen_sink()
    back = convert_model(convert_model(original, "N-mm-MPa"), "N-m-Pa")

    def walk(a, b, path=""):
        if isinstance(a, dict):
            assert set(a) == set(b), f"{path} 的键变了"
            for key in a:
                walk(a[key], b[key], f"{path}.{key}")
        elif isinstance(a, list):
            assert len(a) == len(b), f"{path} 的长度变了"
            for k, (x, y) in enumerate(zip(a, b, strict=True)):
                walk(x, y, f"{path}[{k}]")
        elif isinstance(a, (int, float)) and not isinstance(a, bool):
            assert b == pytest.approx(a, rel=1e-12, abs=1e-18), (
                f"{path} 来回换算之后变了：{a} → {b}")
        else:
            assert a == b, f"{path} 变了：{a} → {b}"

    walk(original, back)


@pytest.mark.parametrize("field", [
    "members.connections.i", "members.connections.j",
    "member_spans.b", "steps.supports.spring",
])
def test_the_fields_that_were_missed_are_now_converted(field):
    """三处实测漏过的字段，各自钉一条。

    参数化点名而不是合成一条：将来再漏，失败信息里直接有字段名。
    """
    factors = _factors("N-m-Pa", "N-mm-MPa")
    original = kitchen_sink()
    converted = convert_model(original, "N-mm-MPa")

    if field == "member_spans.b":
        before = original["member_spans"][0]["b"]
        after = converted["member_spans"][0]["b"]
        assert after == pytest.approx(before * factors["length"])
    elif field.startswith("members.connections"):
        end = field.rsplit(".", 1)[1]
        before = original["members"][0]["connections"][end]
        after = converted["members"][0]["connections"][end]
        for dof, value in before.items():
            factor = (factors["spring_rotation"] if dof.startswith("r")
                      else factors["spring_translation"])
            assert after[dof] == pytest.approx(value * factor), dof
    else:
        before = original["steps"][0]["supports"]["2"]["spring"]
        after = converted["steps"][0]["supports"]["2"]["spring"]
        for index, value in enumerate(before):
            assert after[index] == pytest.approx(
                value * _expected("support_spring", index, factors)), index


def test_the_physics_is_unchanged_by_the_unit_system():
    """项目自己的判据：同一个结构，两种单位制下挠度必须是同一个数。

    这条比逐个系数核对更硬——它不关心实现细节，只问结果对不对。
    """
    from frame3d import solve
    from model_io import from_dict

    payload = kitchen_sink()
    # steps 里那个支座让节点 2 竖向受约束，去掉它才好比挠度
    payload.pop("steps")
    si = from_dict(payload)
    si_tip = solve(si).U[si.node_dofs(2)[2]] * 1e3

    mm_payload = convert_model(payload, "N-mm-MPa")
    mm = from_dict(mm_payload)
    mm_tip = solve(mm).U[mm.node_dofs(2)[2]]

    assert mm_tip == pytest.approx(si_tip, rel=1e-9)


def test_the_kitchen_sink_is_a_legal_model():
    """**这条是闸门的自检。** 上面那个模型要是本身不合法，它用到的那些字段
    就可能根本不会出现在真实模型里——测了个寂寞。
    """
    assert validate_payload(kitchen_sink()) == []


def test_the_kitchen_sink_uses_every_field_that_gets_converted():
    """会换算的字段必须都在 kitchen_sink 里出现过，否则来回换算那条测不到它。"""
    payload = kitchen_sink()

    def present(node, parts):
        """列表里**任意一条**用到就算数——梯形荷载的 w2 只在第三条上。"""
        if not parts:
            return True
        if isinstance(node, list):
            return any(present(item, parts) for item in node)
        if not isinstance(node, dict) or parts[0] not in node:
            return False
        return present(node[parts[0]], parts[1:])

    missing = sorted(path for path, dim in DIMENSIONS.items()
                     if dim is not None
                     and not present(payload, path.split(".")))
    assert not missing, (
        f"这些会换算的字段没被 kitchen_sink 用到：{missing}。"
        "加进去，否则来回换算那条测不出它们。")
