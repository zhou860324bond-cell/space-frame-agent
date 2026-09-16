"""绘图验证。不比对像素，只确认图画得出来、数值摘要对、符号约定没搞反。"""
from pathlib import Path

import pytest

from agent import Session
from plot3d import plot_axial, plot_deformed, plot_diagram

MATERIALS = [{"name": "STEEL", "E": 2.1e11, "nu": 0.3}]
SECTIONS = [
    {"name": "COLUMN", "A": 0.012, "Iy": 8e-5, "Iz": 2.4e-4, "J": 1e-6},
    {"name": "BEAM", "A": 0.010, "Iy": 4e-5, "Iz": 3e-4, "J": 8e-7},
]


@pytest.fixture
def solved():
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    beams = s.generate_frame(spans=[6, 6], storeys=[3.6], bays=[6],
                             column_section="COLUMN", beam_section="BEAM",
                             material="STEEL").payload["beam_member_ids"]
    s.set_load_cases(cases=[{"name": "DL", "member_loads":
                             [{"member": m, "w": [0, 0, -20e3]} for m in beams]}])
    s.solve_model()
    return s


def test_deformed_plot_writes_a_file_and_reports_the_peak(solved, tmp_path):
    out = tmp_path / "d.png"
    info = plot_deformed(solved.frame, solved.solution, "DL", out)
    assert out.exists() and out.stat().st_size > 5000
    expected = solved.query_results(what="max_displacement", case="DL").payload
    assert info["max_displacement_mm"] == pytest.approx(expected["magnitude_mm"], rel=1e-9)
    assert info["at_node"] == expected["node"]
    assert info["max_centerline_displacement_mm"] >= info["max_displacement_mm"]
    assert info["at_member"] in solved.frame.members


def test_axial_plot_writes_a_file(solved, tmp_path):
    out = tmp_path / "a.png"
    info = plot_axial(solved.frame, solved.solution, "DL", out)
    assert out.exists() and out.stat().st_size > 5000
    assert info["max_abs_axial_kN"] > 0


def test_axial_plot_uses_tension_positive_forces(solved):
    """重力作用下柱子受压，绘图取的分量必须让它们是负值。"""
    res = solved.solution["DL"]
    columns = [m.id for m in solved.frame.members.values() if m.section == "COLUMN"]
    assert columns
    assert all(res.member_forces[mid][6] < 0 for mid in columns)


def test_plot_tool_refuses_before_solving(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert not Session().plot_results(kind="deformed").ok


def test_plot_tool_writes_into_results_dir(solved, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = solved.plot_results(kind="axial", case="DL")
    assert r.ok
    assert Path(r.payload["path"]).exists()
    assert "results" in r.payload["path"]


def test_plot_tool_rejects_unknown_case(solved, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = solved.plot_results(kind="deformed", case="NOPE")
    assert not r.ok and "available" in r.payload


def test_static_diagram_keeps_physical_values_after_switching_to_mm(solved, tmp_path):
    si = plot_diagram(solved.frame, solved.solution, "Mz", "DL", tmp_path / "si.png")
    assert solved.set_units("N-mm-MPa").ok
    assert solved.solve_model().ok
    mm = plot_diagram(solved.frame, solved.solution, "Mz", "DL", tmp_path / "mm.png")
    assert mm["peak"] == pytest.approx(si["peak"], rel=1e-8)
    assert mm["at_x"] == pytest.approx(si["at_x"], abs=1e-4)


# --------------------------------------------------------------------- 缩放守卫

def test_safe_scale_treats_a_denormal_peak_as_zero():
    """解析上为零的量算出来常常不是恰好 0.0，而是非规格化浮点数。

    这是 `test_plot_results_covers_every_kind[shear]` **偶发**失败的机理：

    - 剪力在那个算例里解析上恒为零；
    - 浮点求和顺序随 BLAS 线程调度变化，同一份代码有时算出恰好 0.0，
      有时算出一个 denormal（小到 5e-324）；
    - 旧代码的守卫是 `peak > 0`，denormal 过得去，接着
      `0.09 * size / peak` 溢出成 inf，画出来的坐标全是 inf/nan，
      matplotlib 抛 "Axis limits cannot be NaN or Inf"。

    单独跑那条用例永远是绿的——这正是"偶发"最难查的地方。所以这里不去
    复现那次随机，直接把**机理**钉死：喂 denormal，必须当零处理。
    """
    import math

    import viz_theme as T

    for denormal in (5e-324, 1e-320, 1e-310):
        assert denormal > 0.0, "这个值得是正的，否则测的不是想测的东西"
        assert not math.isfinite(0.09 * 10.0 / denormal), \
            "前提变了：这个除法不再溢出，这条测试就失去意义了"
        assert T.safe_scale(10.0, denormal, 0.09) == 0.0

    # 正常值不受影响：倍数照给
    assert T.safe_scale(10.0, 2.0, 0.09) == pytest.approx(0.45)
    # 巨大但有限的倍数是安全的，不该被误伤——偏移量 = 值 × 倍数 ≤
    # peak × 倍数 = fraction × size，天然有界。
    assert T.safe_scale(10.0, 1e-300, 0.09) == pytest.approx(9e299)
    # 零与负数一律压平
    assert T.safe_scale(10.0, 0.0, 0.09) == 0.0
    assert T.safe_scale(0.0, 1.0, 0.09) == 0.0


def test_every_drawing_scale_goes_through_the_guard():
    """四个缩放点必须都走 safe_scale，不能再出现裸的 `size / peak`。

    这个 bug 同时存在于 plot3d 与 plot3d_interactive 的两处，共四个调用点。
    只修一处的话，另一处会在某次重构后把同样的偶发失败带回来——
    这个仓库已经吃过一次同类亏（OpenGL 探测在 conftest 与 doctor 里各有
    一份，只修一份不够）。
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    for name in ("src/plot3d.py", "src/plot3d_interactive.py"):
        text = (root / name).read_text(encoding="utf-8")
        bare = re.findall(r"[\d.]+\s*\*\s*size\s*/\s*peak", text)
        assert not bare, f"{name} 里还有绕过 safe_scale 的裸除法：{bare}"
        assert "T.safe_scale(" in text, f"{name} 没有用 safe_scale"
