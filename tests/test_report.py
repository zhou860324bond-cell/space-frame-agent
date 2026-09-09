"""报告的验证。

报告最要紧的一条：**里面的数必须与工具返回的完全一致**。报告是排版层，
不是第二个计算入口——一旦它自己算了什么，整个"每个数字可溯源"的主张就破了。
所以这里逐项把报告里的数和 `query_*` 的返回值对起来。

其次是"缺什么就说什么"：没有密度就不做模态、全受拉就不做屈曲，
对应小节直接略过并在附录里写明原因，而不是塞一段"分析失败"或者干脆不提。
"""

from pathlib import Path

import pytest

from agent import Session
from report import gather, to_markdown

E, RHO = 2.1e11, 7850.0
MATERIALS = [{"name": "STEEL", "E": E, "nu": 0.3, "density": RHO}]
LIGHT = [{"name": "STEEL", "E": E, "nu": 0.3}]           # 无密度
SECTIONS = [{"name": "COLUMN", "A": 0.012, "Iy": 8e-5, "Iz": 2.4e-4, "J": 1e-6},
            {"name": "BEAM", "A": 0.010, "Iy": 4e-5, "Iz": 3e-4, "J": 8e-7}]


def build(materials=None, combos=True) -> Session:
    s = Session()
    s.define_materials_and_sections(materials or MATERIALS, SECTIONS)
    g = s.generate_frame(spans=[6.0, 6.0], storeys=[3.6],
                         column_section="COLUMN", beam_section="BEAM",
                         material="STEEL")
    beams = g.payload["beam_member_ids"]
    s.set_load_cases(
        cases=[{"name": "DL", "member_loads":
                [{"member": m, "w": [0, 0, -20e3]} for m in beams]},
               {"name": "WL", "nodal_loads":
                [{"node": 6, "load": [60e3, 0, 0, 0, 0, 0]}]}],
        combos=([{"name": "1.3DL", "factors": {"DL": 1.3}},
                 {"name": "DL+WL", "factors": {"DL": 1.3, "WL": 1.5}}]
                if combos else None))
    s.solve_model()
    return s


@pytest.fixture
def doc(tmp_path):
    return gather(build(), out_dir=tmp_path)


# ------------------------------------------------- 数字可溯源

def test_every_number_matches_the_tools(tmp_path):
    """报告是排版层，不是第二个计算入口。逐项对起来。"""
    s = build()
    d = gather(s, out_dir=tmp_path)
    case = d["case"]

    nodal = s.query_results(what="max_displacement", case=case).payload
    inner = s.query_results(what="max_deflection", case=case).payload
    reac = s.query_results(what="reactions", case=case).payload
    assert d["displacement"]["nodal_mm"] == nodal["magnitude_mm"]
    assert d["displacement"]["deflection_mm"] == inner["magnitude_mm"]
    assert d["displacement"]["node"] == nodal["node"]
    assert d["reactions"]["total_vertical"] == reac["vertical_total_kN"]
    assert len(d["reactions"]["rows"]) == len(reac["reactions"])

    for row in d["envelope"]:
        comp = {v: k for k, v in
                {"N": "轴力 N", "Vy": "剪力 Vy", "Vz": "剪力 Vz", "T": "扭矩 T",
                 "My": "弯矩 My", "Mz": "弯矩 Mz"}.items()}[row["分量"]]
        got = s.query_envelope(component=comp).payload
        assert row["最不利值"] == got["peak"]
        assert row["控制组合"] == got["governing_case"]
        assert row["杆件"] == got["at_member"]


def test_the_report_uses_the_controlling_case_by_default(tmp_path):
    s = build()
    assert gather(s, out_dir=tmp_path)["case"] == s._controlling_case()


def test_an_explicit_case_is_honoured(tmp_path):
    assert gather(build(), case="WL", out_dir=tmp_path)["case"] == "WL"


def test_an_unsolved_session_is_refused(tmp_path):
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    with pytest.raises(ValueError, match="还没有求解"):
        gather(s, out_dir=tmp_path)


# ------------------------------------------------- 缺什么说什么

def test_modal_is_skipped_with_a_reason_when_there_is_no_density(tmp_path):
    """没有密度就没有质量矩阵。略过并写明原因，比塞一段"分析失败"干净，
    也比干脆不提诚实——读的人得知道这一节是被明确跳过的。"""
    d = gather(build(materials=LIGHT), out_dir=tmp_path)
    assert "modal" not in d
    assert "模态分析" in d["skipped"]
    assert "density" in d["skipped"]["模态分析"]


def test_the_skipped_section_appears_in_the_markdown(tmp_path):
    text = to_markdown(gather(build(materials=LIGHT), out_dir=tmp_path))
    assert "未包含的内容" in text
    assert "模态分析" in text


def test_buckling_is_skipped_when_nothing_is_in_compression(tmp_path):
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    s.set_model(model={
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                  {"id": 2, "x": 6, "y": 0, "z": 0}],
        "members": [{"id": 1, "i": 1, "j": 2,
                     "section": "BEAM", "material": "STEEL"}],
        "supports": [{"node": 1, "fix": [1] * 6},
                     {"node": 2, "fix": [0, 1, 1, 1, 1, 1]}]})
    s.set_load_cases(cases=[{"name": "T", "nodal_loads":
                             [{"node": 2, "load": [50e3, 0, 0, 0, 0, 0]}]}])
    s.solve_model()
    d = gather(s, out_dir=tmp_path)
    assert "buckling" not in d
    assert "屈曲分析" in d["skipped"]


# ------------------------------------------------- 内容完整

def test_all_expected_sections_are_present(doc):
    text = to_markdown(doc)
    for heading in ("模型概况", "求解与校验", "位移", "支座反力",
                    "内力极值与控制组合", "自振特性", "稳定", "图"):
        assert heading in text, heading


def test_the_report_explains_why_the_two_displacements_differ(doc):
    """节点位移和杆件挠度差着量级，报告必须解释，否则读的人会以为算错了。"""
    text = to_markdown(doc)
    assert "跨中根本没有节点" in text


def test_the_buckling_section_carries_the_caveat(doc):
    """λ 是上限不是承载力。这句在工具、界面、报告里都要出现——
    读报告的人不一定看过工具返回。"""
    text = to_markdown(doc)
    assert "初始缺陷" in text and "上限" in text


def test_figures_are_written_and_referenced(doc, tmp_path):
    assert doc["figures"], doc["skipped"]
    for label, fig in doc["figures"].items():
        assert Path(fig["path"]).exists(), label
        assert fig["w"] > 0 and fig["h"] > 0, "尺寸要带上，排版按它缩放"
    text = to_markdown(doc)
    for label in doc["figures"]:
        assert f"![{label}]" in text


def test_figures_are_trimmed_of_whitespace(tmp_path):
    """三维图周围一大圈留白，不裁的话报告里就是"大白框里一张小图"。"""
    pytest.importorskip("PIL")
    from PIL import Image, ImageChops

    d = gather(build(), out_dir=tmp_path)
    for label, fig in d["figures"].items():
        with Image.open(fig["path"]).convert("RGB") as im:
            bg = Image.new("RGB", im.size, im.getpixel((0, 0)))
            box = ImageChops.difference(im, bg).getbbox()
        # 裁过之后内容应当几乎铺满整张图（留了 8 px 的边）
        assert box[0] <= 10 and box[1] <= 10, (label, box)


def test_markdown_writes_to_disk(tmp_path, doc):
    path = tmp_path / "r.md"
    text = to_markdown(doc, path)
    assert path.read_text(encoding="utf-8") == text


# ------------------------------------------------- Agent 工具

def test_the_tool_writes_both_formats(tmp_path, monkeypatch):
    pytest.importorskip("docx", reason="未安装 python-docx，跳过 Word 输出")
    monkeypatch.chdir(tmp_path)
    r = build().write_report()
    assert r.ok, r.payload
    for fmt, path in r.payload["files"].items():
        assert Path(path).exists(), fmt
    assert Path(r.payload["files"]["docx"]).stat().st_size > 10_000


def test_markdown_works_without_python_docx(tmp_path, monkeypatch):
    """Word 输出是可选依赖。没装 python-docx 时 Markdown 必须照常出——
    报告不能因为一个可选依赖就整个交不出来。"""
    monkeypatch.chdir(tmp_path)
    import report as _report
    monkeypatch.setattr(_report, "to_docx", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError(_report._DOCX_HINT)))
    r = build().write_report(fmt="both")
    assert r.ok
    assert Path(r.payload["files"]["markdown"]).exists()
    assert "python-docx" in r.payload["partial"]["docx"]


def test_the_tool_can_write_markdown_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = build().write_report(fmt="markdown")
    assert set(r.payload["files"]) == {"markdown"}


def test_a_docx_failure_still_leaves_the_markdown(tmp_path, monkeypatch):
    """docx 依赖可选的 python-docx，它出问题不该把 markdown 也拖下水。"""
    monkeypatch.chdir(tmp_path)
    import report as _report
    monkeypatch.setattr(_report, "to_docx",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("假装挂了")))
    r = build().write_report(fmt="both")
    assert r.ok
    assert "markdown" in r.payload["files"]
    assert "docx" in r.payload["partial"]


def test_the_tool_refuses_an_unknown_format(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert not build().write_report(fmt="pdf").ok


def test_the_tool_needs_results():
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    assert not s.write_report().ok


def test_the_docx_is_a_real_zip(tmp_path, monkeypatch):
    """.docx 是个 ZIP。生成器出错时常写出一个空文件或半截文件，
    这条比"文件存在"严格一档。"""
    import zipfile

    pytest.importorskip("docx", reason="未安装 python-docx，跳过 Word 输出")
    monkeypatch.chdir(tmp_path)
    r = build().write_report(fmt="docx")
    path = Path(r.payload["files"]["docx"])
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
    assert "word/document.xml" in names
    assert any(n.startswith("word/media/") for n in names), "图应当嵌进去了"
