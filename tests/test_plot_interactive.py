"""交互式三维视图的验证。

不比像素，验三件事：图元齐全、悬停信息带着节点/杆件编号、
以及**轴力接近零的杆件仍有轮廓可见**——否则梁会连同几何一起从画面里消失。
"""
import pytest

plotly = pytest.importorskip("plotly", reason="未安装 plotly，跳过交互视图测试")

import viz_theme as T                                        # noqa: E402
from agent import Session                                    # noqa: E402
from plot3d_interactive import figure_axial, figure_deformed  # noqa: E402

MATERIALS = [{"name": "STEEL", "E": 2.1e11, "nu": 0.3}]
SECTIONS = [
    {"name": "COLUMN", "A": 0.012, "Iy": 8e-5, "Iz": 2.4e-4, "J": 1e-6},
    {"name": "BEAM", "A": 0.010, "Iy": 4e-5, "Iz": 3e-4, "J": 8e-7},
]


@pytest.fixture(scope="module")
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


def test_deformed_figure_reports_the_same_peak_as_the_tools(solved):
    fig, meta = figure_deformed(solved.frame, solved.solution, "DL")
    expected = solved.query_results(what="max_displacement", case="DL").payload
    assert meta["max_displacement_mm"] == pytest.approx(expected["magnitude_mm"], rel=1e-6)
    assert meta["at_node"] == expected["node"]
    assert meta["max_centerline_displacement_mm"] >= meta["max_displacement_mm"]
    assert len(fig.data) >= 4          # 原始位置、变形后、节点、峰值、支座


def test_deformed_trace_contains_member_interior_stations(solved):
    fig, _ = figure_deformed(solved.frame, solved.solution, "DL")
    moved = next(t for t in fig.data if str(t.name).startswith("变形后"))
    # 旧图每根杆只有两个端点加一个断点，梁弯曲永远显示成直杆。
    assert len(moved.x) > 3 * len(solved.frame.members)


def test_deformed_hover_text_names_the_node_and_components(solved):
    fig, _ = figure_deformed(solved.frame, solved.solution, "DL")
    node_trace = next(t for t in fig.data if t.name == "节点")
    sample = node_trace.text[0]
    assert "节点" in sample and "合位移" in sample and "ux" in sample


def test_axial_figure_keeps_a_visible_outline_over_every_member(solved):
    """接近零的杆件颜色落在中性灰上；压在最上层的发丝线保证几何始终可读。"""
    fig, _ = figure_axial(solved.frame, solved.solution, "DL")
    outline = fig.data[-1]                       # 轮廓画在最后一层
    assert outline.line.color == T.VIEWPORT_REFERENCE
    assert outline.mode == "lines"
    # 每根杆件两个端点加一个断点
    assert len(outline.x) == 3 * len(solved.frame.members)


def test_axial_hover_names_the_member_and_states_tension_or_compression(solved):
    fig, _ = figure_axial(solved.frame, solved.solution, "DL")
    templates = [t.hovertemplate for t in fig.data if t.hovertemplate]
    assert templates
    assert any("受压" in t or "受拉" in t for t in templates)
    assert all("杆件" in t for t in templates)


def test_axial_colour_scale_is_diverging_with_a_neutral_midpoint(solved):
    """发散编码：两个对立色相 + 中性灰中点，不用彩虹，中点不上色相。"""
    assert T.DIVERGING_SCALE[0][1] == T.DIVERGING_LOW
    assert T.DIVERGING_SCALE[1] == (0.5, T.DIVERGING_MID)
    assert T.DIVERGING_SCALE[2][1] == T.DIVERGING_HIGH
    r, g, b = (int(T.DIVERGING_MID[i:i + 2], 16) for i in (1, 3, 5))
    assert max(r, g, b) - min(r, g, b) < 20, "中点必须接近中性灰"


def test_aspect_ratio_follows_the_real_geometry(solved):
    """真实比例，不把 18×6×7 的框架塞进立方体。"""
    fig, _ = figure_deformed(solved.frame, solved.solution, "DL")
    assert fig.layout.scene.aspectmode == "data"
