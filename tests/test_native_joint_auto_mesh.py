"""默认网格超过自研求解器规模时自动放粗。

实测一个 6 杆 D219×8 圆管节点：默认 2.5t 网格 9.4 万个 C3D10、55 万自由度，
是上限的三倍多，而界面上没有改网格的入口——不放粗，这类节点只能失败。
放粗后网格比壁厚粗几倍，二次单元边中节点贴曲面会把薄壁单元拧翻
（detJ < 0），所以放粗档改为边中节点取直线。

这里用一个"单元数按尺寸 1.8 次方反比"的假网格代替 Gmsh，只测放粗的
决策：什么时候放、放多少、带什么选项、说了什么、放不下时怎么报。
"""

from types import SimpleNamespace

import pytest

import native_joint
from solid_joint import SolidJointError


def _fake_mesher(calls, elements_at_reference=94147, reference=8.0):
    def generate(spec, size, hotspot, options=None):
        calls.append((size, hotspot, dict(options or {})))
        elements = int(elements_at_reference * (2.5 * reference / size) ** 1.8)
        nodes = int(elements * 5.8 / 3)
        # 只报长度、不占内存：放粗逻辑只看单元数与节点数
        mesh = SimpleNamespace(elements=range(elements), nodes=range(nodes))
        return mesh, {}
    return generate


def test_an_oversized_default_mesh_is_coarsened_until_it_fits(monkeypatch):
    calls = []
    monkeypatch.setattr(native_joint, "generate_joint_mesh", _fake_mesher(calls))
    size, hotspot, mesh, _cuts, note = native_joint._fit_default_mesh(
        None, 20.0, 8.0, max_elements=80000, max_dof=180000)
    assert len(mesh.elements) <= 80000 and 3 * len(mesh.nodes) <= 180000
    assert size > 20.0 and hotspot == pytest.approx(max(8.0, 0.4 * size))
    assert len(calls) <= 3, "按规律一次算准，不该一档一档地试"
    # 第一次是原样的默认网格；放粗后的那次边中节点取直线
    assert calls[0][2] == {}
    assert calls[-1][2] == {"Mesh.SecondOrderLinear": 1}
    assert note and "自动放粗" in note and "Abaqus" in note


def test_a_default_mesh_that_fits_is_left_exactly_as_it_was(monkeypatch):
    """没超限就不放粗、不加选项、不写警告——对过标的默认路径一点不动。"""
    calls = []
    monkeypatch.setattr(native_joint, "generate_joint_mesh",
                        _fake_mesher(calls, elements_at_reference=20000))
    size, _hotspot, _mesh, _cuts, note = native_joint._fit_default_mesh(
        None, 20.0, 8.0, max_elements=80000, max_dof=180000)
    assert size == 20.0 and note is None
    assert calls == [(20.0, 8.0, {})]


def test_a_joint_too_big_even_after_coarsening_points_to_abaqus(monkeypatch):
    """几何本身约束了网格（放粗也降不下来）时，重试有限次后明确指向 Abaqus。"""
    calls = []

    def stubborn(spec, size, hotspot, options=None):
        calls.append(size)
        return SimpleNamespace(elements=range(200000), nodes=range(400000)), {}

    monkeypatch.setattr(native_joint, "generate_joint_mesh", stubborn)
    with pytest.raises(SolidJointError, match="Abaqus"):
        native_joint._fit_default_mesh(None, 20.0, 8.0,
                                       max_elements=80000, max_dof=180000)
    assert len(calls) == native_joint._DEFAULT_MESH_RETRIES + 1


def test_three_levels_run_coarse_to_fine_and_all_fit(monkeypatch):
    """收敛判断：最细档是规模内能划到的最细，另外两档各放粗 1.5、2.25 倍。"""
    calls = []
    monkeypatch.setattr(native_joint, "generate_joint_mesh", _fake_mesher(calls))
    plan, options, note = native_joint._default_plan(
        None, 8.0, 3, max_elements=80000, max_dof=180000)
    sizes = [size for size, _prepared in plan]
    assert sizes == sorted(sizes, reverse=True), "由粗到细"
    assert sizes[0] == pytest.approx(2.25 * sizes[-1])
    assert sizes[1] == pytest.approx(1.5 * sizes[-1])
    assert plan[-1][1] is not None, "最细档直接用放粗时已经划好的网格，不重划"
    assert plan[0][1] is None and plan[1][1] is None
    assert options == {"Mesh.SecondOrderLinear": 1} and note


def test_one_level_is_the_quick_default(monkeypatch):
    calls = []
    monkeypatch.setattr(native_joint, "generate_joint_mesh",
                        _fake_mesher(calls, elements_at_reference=20000))
    plan, options, note = native_joint._default_plan(
        None, 8.0, 1, max_elements=80000, max_dof=180000)
    assert [size for size, _p in plan] == [20.0]
    assert options is None and note is None
