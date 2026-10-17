"""图纸证据 → 模型 JSON 的确定性组装与核对表。

防的问题都来自旧草图识别：坐标从像素量、截面强制留空、只能落到一个平面。
新路径的数值必须全部来自图上文字，读不懂或没读到就进核对表，不估算。
"""

from __future__ import annotations

from copy import deepcopy

import pytest

import sections
from agent import Session
from drawing_assemble import (CONFIRM, CONFLICT, MISSING, OK, PIXEL_SCALE_ALARM,
                              assemble, resolve)
from drawing_evidence import (EVIDENCE_FORMAT, parse_designation, parse_material,
                              parse_number, validate_evidence)
from model_io import validate_payload


def _src(conf=0.95, image="P1"):
    return {"image": image, "bbox": [0.1, 0.1, 0.2, 0.2], "confidence": conf}


def _axes(labels, positions):
    return [{"label": label, "pos": pos, "source": _src()}
            for label, pos in zip(labels, positions, strict=True)]


def _evidence():
    """两跨（6000+6000）× 一跨（7200）× 两层（3.600、7.200）的钢框架平面+立面证据。"""
    beams = []
    for along, a, b in (("A", "1", "3"), ("B", "1", "3")):
        beams.append({"label": "KL1", "kind": "beam", "along": along, "from": a, "to": b,
                      "levels": ["2F", "3F"], "source": _src()})
    for along in ("1", "2", "3"):
        beams.append({"label": "KL2", "kind": "beam", "along": along, "from": "A",
                      "to": "B", "levels": ["2F", "3F"], "source": _src()})
    columns = [{"label": "KZ1", "kind": "column", "at": [x, y], "from_level": "1F",
                "to_level": "3F", "source": _src()}
               for x in ("1", "2", "3") for y in ("A", "B")]
    return {
        "format": EVIDENCE_FORMAT,
        "images": [{"id": "P1", "name": "平面图.png", "width": 2000, "height": 1000}],
        "units": {"length": {"value": "mm", "text": "尺寸以毫米计", "source": _src()},
                  "elevation": {"value": "m", "text": "标高以米计", "source": _src()}},
        "grids": {
            "x": {"axes": _axes(["1", "2", "3"], [0.1, 0.4, 0.7]),
                  "spacings": [{"from": "1", "to": "2", "text": "6000", "source": _src()},
                               {"from": "2", "to": "3", "text": "6,000", "source": _src()}],
                  "total": {"text": "12000", "source": _src()}},
            "y": {"axes": _axes(["A", "B"], [0.8, 0.08]),
                  "spacings": [{"from": "A", "to": "B", "text": "7200", "source": _src()}]},
        },
        "levels": [{"name": "1F", "text": "±0.000", "source": _src()},
                   {"name": "2F", "text": "3.600", "source": _src()},
                   {"name": "3F", "text": "7.200", "source": _src()}],
        "members": beams + columns,
        "sections": [{"label": "KZ1", "text": "HW300×300×10×15", "source": _src()},
                     {"label": "KL1", "text": "HN400×200×8×13", "source": _src()},
                     {"label": "KL2", "text": "HN350x175x7x11", "source": _src()}],
        "material": {"text": "钢材 Q355", "source": _src()},
        "support": {"kind": "fixed", "text": "柱脚刚接", "source": _src()},
    }


def _row(assembly, row_id):
    return next(row for row in assembly.rows if row.id == row_id)


def _screenshot_evidence():
    """按用户截图的 6000×4000×3500、四柱四梁与统一截面重建证据，不使用像素尺寸。"""
    evidence = _evidence()
    evidence["grids"]["x"]["axes"] = _axes(["1", "2"], [0.1, 0.7])
    evidence["grids"]["x"]["spacings"] = [{"from": "1", "to": "2", "text": "6000"}]
    evidence["grids"]["x"]["total"] = {"text": "6000 mm"}
    evidence["grids"]["y"]["spacings"][0]["text"] = "4000"
    evidence["levels"] = [{"name": "±0.000", "text": "±0.000", "source": _src()},
                          {"name": "屋面", "text": "3.500 m", "source": _src()},
                          {"name": "+3.500 m", "text": "+3.500 m", "source": _src()}]
    evidence["members"] = [
        {"label": label, "kind": "column", "at": at, "from_level": "±0.000",
         "to_level": "屋面", "source": _src()}
        for label, at in (("C1", ["1", "A"]), ("C2", ["2", "A"]),
                          ("C3", ["2", "B"]), ("C4", ["1", "B"]))]
    evidence["members"] += [
        {"label": label, "kind": "beam", "along": along, "from": a, "to": b,
         "levels": ["屋面"], "source": _src()}
        for label, along, a, b in (("B1", "A", "1", "2"), ("B2", "2", "A", "B"),
                                   ("B3", "B", "2", "1"), ("B4", "1", "B", "A"))]
    evidence["sections"] = [{"labels": [item["label"] for item in evidence["members"]],
                             "text": "H300×200×8×12", "source": _src()}]
    return evidence


def test_screenshot_evidence_generates_all_eight_members():
    """截图中的八根杆件不应因带单位标高、跨图别名和统一截面全部变成冲突。"""
    result = assemble(_screenshot_evidence())
    assert not result.blocking
    assert len(result.model["nodes"]) == len(result.model["members"]) == 8
    assert len(result.model["supports"]) == 4
    assert len(result.model["sections"]) == 1
    assert {n["z"] for n in result.model["nodes"]} == {0.0, 3.5}
    assert {n["x"] for n in result.model["nodes"]} == {0.0, 6.0}
    assert {n["y"] for n in result.model["nodes"]} == {0.0, 4.0}
    assert validate_payload(result.model) == []


def test_unit_bearing_levels_and_repeated_view_names_share_one_storey():
    """用户截图：屋面与 +3.500 m 在不同视图重复，旧实现让全部梁柱冲突。"""
    evidence = _evidence()
    evidence["levels"][1]["text"] = "+3.600 m"
    evidence["levels"].append({"name": "屋面", "text": "+3.600 m", "source": _src()})
    for member in evidence["members"]:
        if member["kind"] == "beam":
            member["levels"] = ["屋面", "3F"]
    result = assemble(evidence)
    assert not result.blocking
    assert len(result.model["nodes"]) == 18
    assert len(result.model["members"]) == 26
    assert {node["z"] for node in result.model["nodes"]} == {0.0, 3.6, 7.2}
    assert evidence["levels"][1]["text"] == "+3.600 m"


def test_explicit_length_units_convert_to_declared_units():
    """标注自带单位时换算，不删除单位后把 3600 mm 当作 3600 m。"""
    evidence = _evidence()
    evidence["levels"][1]["text"] = "3600 mm"
    evidence["grids"]["x"]["spacings"][0]["text"] = "6 m"
    result = assemble(evidence)
    assert not result.blocking
    assert {node["z"] for node in result.model["nodes"]} == {0.0, 3.6, 7.2}
    assert {node["x"] for node in result.model["nodes"]} == {0.0, 6.0, 12.0}
    evidence["units"]["elevation"] = None
    missing_unit = assemble(evidence)
    assert _row(missing_unit, "units.elevation").status == MISSING
    assert missing_unit.model is None


@pytest.mark.parametrize("text", ["约3.6m", "3.6m+3.6m", "3.6 ft"])
def test_ambiguous_or_unsupported_length_text_still_blocks(text):
    """支持单位后缀不应变成从不确定说明、算式或不支持单位中随意取数。"""
    evidence = _evidence()
    evidence["levels"][1]["text"] = text
    result = assemble(evidence)
    assert _row(result, "levels.1").status == CONFLICT
    assert result.model is None


def test_conflicting_values_for_same_level_name_still_block():
    """跨图同名且异值的标高不能当别名合并，必须保留冲突。"""
    evidence = _evidence()
    evidence["levels"].append({"name": "2F", "text": "4.200 m", "source": _src()})
    result = assemble(evidence)
    assert _row(result, "levels.3").status == CONFLICT
    assert result.model is None


def test_shared_section_targets_explicit_member_labels():
    """用户截图：全部梁柱同截面不能被当成不存在的单个杆件编号。"""
    evidence = _evidence()
    evidence["sections"] = [{"labels": ["KZ1", "KL1", "KL2"],
                             "text": "H300×200×8×12", "source": _src()}]
    assert validate_evidence(evidence) == []
    result = assemble(evidence)
    assert not result.blocking
    assert len(result.model["sections"]) == 1
    assert {member["section"] for member in result.model["members"]} == {"H300×200×8×12"}
    assert _row(result, "sections.0").source.endswith("原文「H300×200×8×12」")
    corrected = assemble(resolve(evidence, "sections.0", "H400×200×8×13"))
    assert {m["section"] for m in corrected.model["members"]} == {"H400×200×8×13"}


def test_shared_section_does_not_fill_unlisted_members():
    """统一说明只应用于明确列出的编号，不扩散到未指定的杆件。"""
    evidence = _evidence()
    evidence["sections"] = [{"labels": ["KZ1", "KL1"],
                             "text": "H300×200×8×12", "source": _src()}]
    result = assemble(evidence)
    assert _row(result, "sections+KL2").status == MISSING
    assert result.model is None


def test_shared_section_conflicting_assignment_is_reported():
    """统一说明与单独截面表冲突时不能靠读取顺序静默覆盖。"""
    evidence = _evidence()
    evidence["sections"].append({"labels": ["KZ1", "KL1"],
                                 "text": "H300×200×8×12", "source": _src()})
    result = assemble(evidence)
    assert _row(result, "sections.3").status == CONFLICT
    assert result.model is None


def test_frame_assembles_exact_coordinates_from_dimension_text():
    """坐标来自间距文字累加与标高文字，不来自像素；拆跨、拆层后通过模型校验。"""
    result = assemble(_evidence())
    assert result.model is not None and not result.blocking
    model = result.model
    assert validate_payload(model) == []
    assert {n["x"] for n in model["nodes"]} == {0.0, 6.0, 12.0}
    assert {n["y"] for n in model["nodes"]} == {0.0, 7.2}
    assert {n["z"] for n in model["nodes"]} == {0.0, 3.6, 7.2}
    assert len(model["nodes"]) == 18
    # 柱 6 根×2 层；x 向梁 2 条线×2 跨×2 层；y 向梁 3 条线×1 跨×2 层
    assert len(model["members"]) == 12 + 8 + 6
    assert len(model["sets"]["KZ1"]["members"]) == 12
    assert all(m["ref_vector"] == [1.0, 0.0, 0.0]
               for m in model["members"] if m["id"] in model["sets"]["KZ1"]["members"])
    assert len(model["supports"]) == 6
    assert all(s["fix"] == [1] * 6 for s in model["supports"])
    assert {model["nodes"][s["node"] - 1]["z"] for s in model["supports"]} == {0.0}


def test_section_properties_come_from_the_kernel_builder():
    """规格字符串只解析尺寸，特性由 sections 构造器算，与手工调用逐位一致。"""
    model = assemble(_evidence()).model
    by_name = {s["name"]: s for s in model["sections"]}
    expected = sections.i_section("HN400×200×8×13", 0.4, 0.2, 0.008, 0.013)
    assert by_name["HN400×200×8×13"] == expected
    beam = model["sets"]["KL2"]["members"][0]
    assert model["members"][beam - 1]["section"] == "HN350×175×7×11"


def test_material_from_code_table_needs_one_confirmation():
    """弹性模量来自规范表也要用户点一次确认，确认前不能生成模型 JSON。"""
    result = assemble(_evidence())
    row = _row(result, "material")
    assert row.status == CONFIRM and "GB 50017" in row.value
    assert not result.ready
    confirmed = assemble(resolve(_evidence(), "material"))
    assert confirmed.ready
    material = confirmed.model["materials"][0]
    assert (material["name"], material["E"], material["nu"]) == ("Q355", 206.0e9, 0.3)


def test_imported_model_is_accepted_by_session():
    """生成的 JSON 能整体写入 Session，且只占一个撤销步骤。"""
    model = assemble(resolve(_evidence(), "material")).model
    session = Session()
    assert session.set_model(model).ok
    assert len(session.model["members"]) == 26
    assert session.undo()


def test_total_dimension_mismatch_is_a_conflict_until_corrected():
    """6000 被抄成 8000 时分段和与总尺寸不符，必须改正，不能确认放行。"""
    evidence = _evidence()
    evidence["grids"]["x"]["spacings"][1]["text"] = "8000"
    result = assemble(evidence)
    row = _row(result, "grids.x.total")
    assert row.status == CONFLICT and "14000" in row.message
    assert result.model is None
    fixed = assemble(resolve(evidence, "grids.x.spacings.1", "6000"))
    assert _row(fixed, "grids.x.total").status == OK
    assert fixed.model is not None


def test_pixel_ratio_only_raises_an_alarm_never_changes_values():
    """没有总尺寸时，靠同图各段像素比例发现可疑抄写；数值仍取文字。"""
    evidence = _evidence()
    evidence["grids"]["x"].pop("total")
    evidence["grids"]["x"]["spacings"][1]["text"] = "8000"
    result = assemble(evidence)
    row = _row(result, "grids.x.spacings.1")
    assert row.status == CONFIRM and "可能抄错" in row.message
    assert _row(result, "grids.x.spacings.0").status == OK
    assert result.model is not None
    assert max(n["x"] for n in result.model["nodes"]) == pytest.approx(14.0)
    assert PIXEL_SCALE_ALARM < 8000 / 6000 - 1


def test_missing_spacing_is_never_measured_from_pixels():
    """轴线位置都在，但间距文字缺失时拒绝生成，而不是按像素推算。"""
    evidence = _evidence()
    evidence["grids"]["x"]["spacings"].pop(1)
    evidence["grids"]["x"].pop("total")
    result = assemble(evidence)
    row = _row(result, "grids.x.spacings+2:3")
    assert row.status == MISSING and "不从图上量取" in row.message
    assert result.model is None
    filled = assemble(resolve(evidence, "grids.x.spacings+2:3", "6000"))
    assert filled.model is not None
    assert _row(filled, "grids.x.spacings.1").source == "用户填写"


def test_unit_note_missing_is_asked_not_defaulted():
    """图上没有单位说明时不按「毫米」惯例默认，必须由用户选。"""
    evidence = _evidence()
    del evidence["units"]["length"]
    result = assemble(evidence)
    row = _row(result, "units.length")
    assert row.status == MISSING and row.options == ("mm", "m")
    assert result.model is None
    chosen = assemble(resolve(evidence, "units.length", "mm"))
    assert chosen.model is not None


def test_support_without_note_offers_choices_and_pinned_releases_rotations():
    """没有柱脚说明时不设默认；用户选铰接后只约束平动。"""
    evidence = _evidence()
    evidence["support"] = None
    row = _row(assemble(evidence), "support")
    assert row.status == MISSING and row.options == ("固接", "铰接")
    model = assemble(resolve(evidence, "support", "铰接")).model
    assert all(s["fix"] == [1, 1, 1, 0, 0, 0] for s in model["supports"])


def test_missing_and_unsupported_sections_block_generation():
    """构件表缺编号、或写了本版不能换算的矩形管规格，都要用户处理。"""
    evidence = _evidence()
    evidence["sections"] = evidence["sections"][:2]
    evidence["sections"][0]["text"] = "□400×400×16"
    result = assemble(evidence)
    assert _row(result, "sections+KL2").status == MISSING
    unsupported = _row(result, "sections.0")
    assert unsupported.status == CONFLICT and "矩形管" in unsupported.message
    assert result.model is None


def test_rectangle_designation_depends_on_drawing_convention():
    """「300×600」按平法理解为宽×高，是约定而非证据，所以要求确认。"""
    evidence = _evidence()
    evidence["sections"][1]["text"] = "300×600"
    row = _row(assemble(evidence), "sections.1")
    assert row.status == CONFIRM and "b×h" in row.message
    model = assemble(resolve(resolve(evidence, "sections.1"), "material")).model
    rect = next(s for s in model["sections"] if s["name"] == "300×600")
    assert rect["geometry"]["dimensions"] == {"b": 0.3, "h": 0.6}


def test_member_with_unknown_axis_can_be_deleted():
    """杆件引用了图上没有的轴号时报冲突；用户删除这条后可以继续。"""
    evidence = _evidence()
    evidence["members"].append({"label": "KL3", "kind": "beam", "along": "C",
                                "from": "1", "to": "2", "levels": ["2F"]})
    index = len(evidence["members"]) - 1
    row = _row(assemble(evidence), f"members.{index}")
    assert row.status == CONFLICT and "C" in row.message
    resolved = resolve(evidence, f"members.{index}", "删除此条")
    assert resolved["members"][index]["deleted"] is True
    assert assemble(resolved).model is not None


def test_overlapping_members_with_different_labels_conflict():
    """同一位置被读成两个编号时不静默取其一。"""
    evidence = _evidence()
    evidence["members"].append({"label": "KL9", "kind": "beam", "along": "A",
                                "from": "1", "to": "2", "levels": ["2F"]})
    row = _row(assemble(evidence), f"members.{len(evidence['members']) - 1}")
    assert row.status == CONFLICT and "KL1" in row.message


def test_low_confidence_reading_needs_confirmation():
    """置信度低于 0.75 的读数进「需确认」，确认后保留原文和原出处。"""
    evidence = _evidence()
    evidence["levels"][1]["source"]["confidence"] = 0.6
    row = _row(assemble(evidence), "levels.1")
    assert row.status == CONFIRM and row.image == "P1"
    resolved = resolve(evidence, "levels.1")
    assert resolved["levels"][1]["confirmed"] is True
    edited = resolve(evidence, "levels.1", "3.900")
    assert edited["levels"][1]["source"] == {"user": True}
    assert edited["levels"][1]["source_before"]["confidence"] == 0.6


def test_planar_elevation_without_letter_axes():
    """只有立面图（一组数字轴）时建平面框架，y 取 0，不要求虚构字母轴。"""
    evidence = _evidence()
    evidence["grids"]["y"] = {"axes": [], "spacings": []}
    evidence["members"] = [
        {"label": "KZ1", "kind": "column", "at": [x], "from_level": "1F", "to_level": "2F"}
        for x in ("1", "2")] + [
        {"label": "KL1", "kind": "beam", "from": "1", "to": "2", "levels": ["2F"]}]
    evidence["sections"] = evidence["sections"][:2]
    model = assemble(evidence).model
    assert model is not None
    assert len(model["nodes"]) == 4 and len(model["members"]) == 3
    assert {n["y"] for n in model["nodes"]} == {0.0}


def test_missing_levels_can_be_filled_as_a_list():
    """只上传平面图时标高缺失；用户按逗号填写后生成。"""
    evidence = _evidence()
    evidence["levels"] = []
    del evidence["units"]["elevation"]
    result = assemble(evidence)
    assert _row(result, "levels+").status == MISSING
    filled = resolve(resolve(evidence, "levels+", "±0.000, 3.600, 7.200"),
                     "units.elevation", "m")
    for member in filled["members"]:
        if member["kind"] == "beam":
            member["levels"] = ["3.600", "7.200"]
        else:
            member["from_level"], member["to_level"] = "±0.000", "7.200"
    assert assemble(filled).model is not None


@pytest.mark.parametrize(("text", "value"), [
    ("6000", 6000.0), ("6,000", 6000.0), ("6 000", 6000.0), ("±0.000", 0.0),
    ("-0.450", -0.45), ("+3.600", 3.6), ("%%p0.000", 0.0), ("３６００", 3600.0),
])
def test_parse_number_accepts_drawing_notations(text, value):
    """图纸常见写法都能读，结果是精确数值。"""
    assert parse_number(text) == value


@pytest.mark.parametrize("text", ["6000(估)", "约6m", "", "6.0.0", None, True])
def test_parse_number_rejects_annotated_or_vague_text(text):
    """带注释或含糊的文字不当作数字，交给用户。"""
    assert parse_number(text) is None


@pytest.mark.parametrize(("text", "kind", "dims"), [
    ("HN400×200×8×13", "i", {"h": 400, "b": 200, "tw": 8, "tf": 13}),
    ("BH500x250x10x16", "i", {"h": 500, "b": 250, "tw": 10, "tf": 16}),
    ("H-300*150*6.5*9", "i", {"h": 300, "b": 150, "tw": 6.5, "tf": 9}),
    ("Φ219×8", "tube", {"d": 219, "t": 8}),
    ("D168X6", "tube", {"d": 168, "t": 6}),
    ("φ50", "circle", {"d": 50}),
])
def test_parse_designation(text, kind, dims):
    """常见规格写法解析成与 section_geometry 一致的轮廓尺寸。"""
    shape = parse_designation(text)
    assert shape["type"] == kind and shape["dimensions"] == dims
    assert shape["convention"] == ""


@pytest.mark.parametrize("text", ["I20a", "HN400×200", "L100×8", ""])
def test_parse_designation_rejects_unknown_formats(text):
    """查表型号或不完整规格不按相近规格猜。"""
    with pytest.raises(ValueError, match="规格"):
        parse_designation(text)


def test_parse_material_requires_grade_or_explicit_modulus():
    """只写「钢材」不估弹性模量；直接写 E 和 ν 时按单位换算。"""
    with pytest.raises(ValueError):
        parse_material("钢材")
    with pytest.raises(ValueError, match="Q550"):
        parse_material("Q550")
    assert parse_material("C30")["E"] == 3.00e10
    explicit = parse_material("E=206000MPa, ν=0.3")
    assert (explicit["E"], explicit["nu"]) == (206.0e9, 0.3)


def test_validate_evidence_reports_shape_errors_for_repair():
    """大模型漏字段时给出可回喂的错误，而不是在组装时抛异常。"""
    assert validate_evidence(_evidence()) == []
    broken = deepcopy(_evidence())
    broken["members"][0].pop("levels")
    broken["members"][-1]["kind"] = "brace"
    broken["grids"]["x"]["axes"].append({"label": "1"})
    errors = validate_evidence(broken)
    assert any("levels" in e for e in errors)
    assert any("beam 或 column" in e for e in errors)
    assert any("轴号重复" in e for e in errors)
