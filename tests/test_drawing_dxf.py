"""结构图纸 DXF 入口：实体 → 文字证据 → 与图片入口同一套组装。

防的问题：DXF 里坐标是精确的，最容易犯的错是「既然有坐标就直接量」——那样
轴网间距、标高都会绕过标注原文，图纸改写过的尺寸也查不出来。这里要求间距取
DIMENSION 实测值并保留原文、标高取标高文字、图层用途只由用户或大模型判断；
缺图层、缺尺寸时进核对表，不从几何估算。
"""

from __future__ import annotations

import pytest

ezdxf = pytest.importorskip("ezdxf", reason="未安装 ezdxf，跳过 DXF 入口测试")

from ezdxf.enums import TextEntityAlignment  # noqa: E402

from drawing_assemble import CONFIRM, CONFLICT, MISSING, OK, assemble, resolve  # noqa: E402
from drawing_dxf import (DxfError, judge_layers, model_layer_items, read_dxf,  # noqa: E402
                         reread, scan_layers)
from drawing_evidence import EVIDENCE_FORMAT, validate_evidence  # noqa: E402
from model_io import validate_payload  # noqa: E402

ROLES = {"轴线": "axis", "尺寸": "dimension", "标高": "level", "梁": "beam", "柱": "column",
         "截面表": "table", "说明": "notes"}
LEVELS = {"%%p0.000": 0.0, "3.600": 3600.0, "7.200": 7200.0}


def _text(msp, text, at, layer, height=250.0, center=False):
    entity = msp.add_text(text, height=height, dxfattribs={"layer": layer})
    entity.set_placement(at, align=TextEntityAlignment.MIDDLE_CENTER if center else None)
    return entity


def _frame_dxf(path, *, dims=((0, 6000), (6000, 12000), (0, 12000)), override=None,
               extra_axis_at=None):
    """两跨（6000+6000）两层（3.600、7.200）钢框架立面单线图，单位毫米。"""
    doc = ezdxf.new("R2010", setup=True)
    doc.header["$INSUNITS"] = 4
    msp = doc.modelspace()
    axes = [("1", 0.0), ("2", 6000.0), ("3", 12000.0)]
    if extra_axis_at is not None:
        axes.append(("1", extra_axis_at))
    for label, x in axes:
        msp.add_line((x, -2000), (x, 8000), dxfattribs={"layer": "轴线"})
        msp.add_circle((x, -2400), 400, dxfattribs={"layer": "轴线"})
        _text(msp, label, (x, -2400), "轴线", 350, center=True)
    for index, (a, b) in enumerate(dims):
        base = -1200 if b - a < 12000 else -800
        dim = msp.add_linear_dim(base=(a, base), p1=(a, -2000), p2=(b, -2000),
                                 text=(override or {}).get(index, "<>"), dimstyle="Standard",
                                 dxfattribs={"layer": "尺寸"})
        dim.render()
    for text, y in LEVELS.items():
        msp.add_lwpolyline([(-1500, y), (-1800, y + 300), (-1200, y + 300)], close=True,
                           dxfattribs={"layer": "标高"})
        _text(msp, text, (-1900, y + 400), "标高")
    for x in (0.0, 6000.0, 12000.0):
        msp.add_line((x, 0), (x, 7200), dxfattribs={"layer": "柱"})
        _text(msp, "KZ1", (x + 150, 1800), "柱")
    for y in (3600.0, 7200.0):
        msp.add_line((0, y), (12000, y), dxfattribs={"layer": "梁"})
        _text(msp, "KL1", (3000, y + 100), "梁")
    for y, cells in ((10000, ("编号", "规格")), (9500, ("KZ1", "HW300×300×10×15")),
                     (9000, ("KL1", "HN400×200×8×13"))):
        for x, cell in zip((15000, 17000), cells, strict=True):
            _text(msp, cell, (x, y), "截面表")
    _text(msp, "钢材均采用 Q355B", (15000, 8000), "说明")
    _text(msp, "柱脚刚接", (15000, 7600), "说明")
    doc.saveas(path)
    return str(path)


def _image_evidence():
    """同一榀框架由图片入口抄出的证据（立面图只有一组轴线）。"""
    src = {"image": "P1", "bbox": [0.1, 0.1, 0.2, 0.2], "confidence": 0.95}
    levels = [("1F", "±0.000"), ("2F", "3.600"), ("3F", "7.200")]
    return {
        "format": EVIDENCE_FORMAT,
        "images": [{"id": "P1", "name": "立面图.png", "width": 2000, "height": 1000}],
        "units": {"length": {"value": "mm", "text": "尺寸以毫米计", "source": src},
                  "elevation": {"value": "m", "text": "标高以米计", "source": src}},
        "grids": {"x": {"axes": [{"label": v, "source": src} for v in ("1", "2", "3")],
                        "spacings": [{"from": "1", "to": "2", "text": "6000", "source": src},
                                     {"from": "2", "to": "3", "text": "6000", "source": src}],
                        "total": {"text": "12000", "source": src}}},
        "levels": [{"name": n, "text": t, "source": src} for n, t in levels],
        "members": [{"label": "KL1", "kind": "beam", "from": "1", "to": "3",
                     "levels": ["2F", "3F"], "source": src}]
        + [{"label": "KZ1", "kind": "column", "at": [x], "from_level": "1F",
            "to_level": "3F", "source": src} for x in ("1", "2", "3")],
        "sections": [{"label": "KZ1", "text": "HW300×300×10×15", "source": src},
                     {"label": "KL1", "text": "HN400×200×8×13", "source": src}],
        "material": {"text": "钢材 Q355", "source": src},
        "support": {"kind": "fixed", "text": "柱脚刚接", "source": src},
    }


def _row(assembly, row_id):
    return next(row for row in assembly.rows if row.id == row_id)


def _settle(evidence):
    """按核对表处理：确认单位与材料，选标高单位与柱脚做法。"""
    for row_id, value in (("units.length", None), ("units.elevation", "m"),
                          ("support", "固接"), ("material", None)):
        evidence = resolve(evidence, row_id, value)
    return evidence


@pytest.fixture
def frame(tmp_path):
    return _frame_dxf(tmp_path / "框架立面.dxf")


def test_dxf_frame_assembles_same_model_as_image_entry(frame):
    """两跨两层框架：DXF 入口与图片入口组装出的模型逐项相同（坐标、杆件、截面）。"""
    evidence = read_dxf(frame, ROLES)
    assert validate_evidence(evidence) == []
    assert evidence["notes"] == []
    result = assemble(_settle(evidence))
    assert result.ready, [(r.id, r.status, r.message) for r in result.pending]
    image = assemble(_image_evidence())
    image_model = assemble(resolve(_image_evidence(), "material")).model
    assert image.model is not None
    assert result.model == image_model
    assert validate_payload(result.model) == []
    assert {(n["x"], n["z"]) for n in result.model["nodes"]} == {
        (x, z) for x in (0.0, 6.0, 12.0) for z in (0.0, 3.6, 7.2)}
    assert len(result.model["members"]) == 6 + 4


def test_values_come_from_entities_with_handles(frame):
    """间距取 DIMENSION 实测值、原文另存；每条出处是能在文件里找到的实体句柄。"""
    evidence = read_dxf(frame, ROLES)
    doc = ezdxf.readfile(frame)
    spacing = evidence["grids"]["x"]["spacings"][0]
    assert spacing["value"] == 6000.0 and spacing["text"] == "6000"
    assert doc.entitydb[spacing["source"]["handles"][0]].dxftype() == "DIMENSION"
    assert [level["text"] for level in evidence["levels"]] == ["±0.000", "3.600", "7.200"]
    for level in evidence["levels"]:
        kinds = {doc.entitydb[h].dxftype() for h in level["source"]["handles"]}
        assert kinds == {"TEXT", "LWPOLYLINE"}         # 标高文字 + 旁边的标高符号
    unit = assemble(evidence)
    row = _row(unit, "units.length")
    assert row.status == CONFIRM and "$INSUNITS" in row.source and "$INSUNITS" in row.message
    assert "#" in _row(unit, "grids.x.spacings.0").source
    assert {s["label"]: s["text"] for s in evidence["sections"]} == {
        "KZ1": "HW300×300×10×15", "KL1": "HN400×200×8×13"}


def test_unassigned_layer_is_not_read_and_geometry_is_not_measured(frame):
    """尺寸图层没指定用途时不读标注，也不拿轴线坐标代替间距；指定后重读即可。"""
    roles = dict(ROLES)
    del roles["尺寸"]
    evidence = read_dxf(frame, roles)
    result = assemble(evidence)
    layer_index = next(i for i, item in enumerate(evidence["layers"]) if item["name"] == "尺寸")
    layer_row = _row(result, f"layers.{layer_index}")
    assert layer_row.status == CONFIRM and layer_row.value.startswith("未判断")
    assert _row(result, "grids.x.spacings+1:2").status == MISSING
    assert result.model is None
    fixed = reread(resolve(evidence, layer_row.id, "尺寸标注"), frame)
    assert _row(assemble(fixed), f"layers.{layer_index}").status == OK
    assert assemble(_settle(fixed)).ready


def test_missing_dimension_goes_to_checklist(tmp_path):
    """图上没有标 2→3 的尺寸：进核对表缺失项，不按轴线几何估成 6000。"""
    path = _frame_dxf(tmp_path / "缺尺寸.dxf", dims=((0, 6000),))
    result = assemble(_settle(read_dxf(path, ROLES)))
    assert _row(result, "grids.x.spacings+2:3").status == MISSING
    assert result.model is None


def test_dimension_text_override_conflicts_with_measurement(tmp_path):
    """标注文字被改写成与实测不同的数时报冲突；用户照图改写后以文字为准、实测值留档。"""
    path = _frame_dxf(tmp_path / "改写.dxf", override={0: "6500"})
    evidence = _settle(read_dxf(path, ROLES))
    row = _row(assemble(evidence), "grids.x.spacings.0")
    assert row.status == CONFLICT and "6500" in row.message and "6000" in row.message
    fixed = resolve(evidence, row.id, "6000")
    item = fixed["grids"]["x"]["spacings"][0]
    assert item["value_before"] == 6000.0 and item["source"] == {"user": True}
    assert assemble(fixed).ready


def test_layer_roles_judged_by_model_need_confirmation(frame):
    """大模型只判断图层用途，结果逐个待确认；不认识的用途按未判断处理。"""
    seen = []

    def call(system, text):
        seen.append(text)
        return ('{"layers": {"轴线": "axis", "尺寸": "dimension", "标高": "level",'
                ' "梁": "beam", "柱": "column", "截面表": "table", "说明": "材料"}}')

    roles = judge_layers(call, scan_layers(frame))
    assert roles["说明"] is None and roles["轴线"] == "axis"
    assert "KZ1" in seen[0]                        # 模型看得到文字样本，用来判断用途
    evidence = read_dxf(frame, model_layer_items(frame, roles, "测试模型"))
    rows = [row for row in assemble(evidence).rows if row.category == "图层"]
    assert rows and all(row.status == CONFIRM for row in rows)
    assert any("大模型" in row.message for row in rows)
    assert evidence["grids"]["x"]["spacings"]          # 数值仍来自实体


def test_two_views_with_same_axis_label_are_refused(tmp_path):
    """同一轴号出现在两处（多个视图）时说明原因与做法，不擅自挑一个。"""
    path = _frame_dxf(tmp_path / "两个视图.dxf", extra_axis_at=50000.0)
    with pytest.raises(DxfError, match="另存为单独的 DXF"):
        read_dxf(path, ROLES)


def test_invalid_file_and_bad_roles_explain_next_step(tmp_path):
    """不是 DXF 的文件、无效的图层用途都给出原因和下一步。"""
    bad = tmp_path / "坏文件.dxf"
    bad.write_bytes(b"not a dxf at all")
    with pytest.raises(DxfError, match="另存为 DXF"):
        read_dxf(bad)
    with pytest.raises(DxfError, match="用途"):
        read_dxf(_frame_dxf(tmp_path / "框架.dxf"), {"轴线": "轴"})


def test_validate_evidence_checks_entity_handles(frame):
    """实体句柄出处的形状检查：句柄必须是非空字符串数组。"""
    evidence = read_dxf(frame, ROLES)
    evidence["members"][0]["source"] = {"file": "D1", "handles": [3]}
    errors = validate_evidence(evidence)
    assert any("members[0].source.handles" in e for e in errors)
