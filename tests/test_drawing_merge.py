"""混合输入的图纸识别：补充说明作为带出处的证据，图片、几个 DXF、说明合成一个模型。

防的问题：

* 把用户写的文字直接拼进提示词让模型自由发挥，会断掉出处链——核对表里的数说不清
  来自图还是来自话，模型还可能借说明「补」出图上没有的数。这里说明逐句编号，模型
  只能原样抄，代码逐字核对，抄不出原文的条目不采用并要用户确认。
* 几份证据各说各的：同一段间距图上写 6000、说明写 6600，任何一边自动胜出都会悄悄
  改掉模型；这里要报冲突、列出两边来源、由用户选。值相同（6000 与 6,000）则合并。
* 立面 DXF 只有一个方向的轴线：直接合并会把几榀框架叠在 y=0 一个平面上；这里要用户
  选所在轴线，没有平面轴网时问是不是同一榀。
* 重读、加图后用户已处理的行丢失，要从头再点一遍。
"""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from drawing_assemble import CONFIRM, CONFLICT, INFO, MISSING, OK, assemble, resolve
from drawing_evidence import (EVIDENCE_FORMAT, bind_statement_sources, split_sentences,
                              validate_evidence)
from drawing_merge import merge_evidence, merge_order
from test_drawing_assemble import _evidence

SENTENCES = ["1~2 轴间距应为 6600", "柱脚铰接", "尺寸单位为毫米", "KL1 采用 HN400×200×8×13"]


def _statement(raw: dict, sentences=SENTENCES, known=()) -> dict:
    evidence = bind_statement_sources(raw, "T1", list(sentences), known)
    evidence.update(format=EVIDENCE_FORMAT, images=[], files=[], statements=[
        {"id": "T1", "name": "补充说明", "sentences": list(sentences)}])
    return evidence


def _row(assembly, row_id):
    return next(row for row in assembly.rows if row.id == row_id)


def _rows(assembly, status):
    return [row for row in assembly.rows if row.status == status]


def _option(row, needle):
    return next(option for option in row.options if needle in option)


# ---- 补充说明：逐句编号、原样抄写、逐字核对 -----------------------------------------

def test_sentences_split_on_full_stops_but_keep_decimals():
    """句子按句号、分号、换行切分；「3.6 m」里的小数点不切。"""
    text = "层高 3.6 m；柱脚铰接。\n\n钢材 Q355!  "
    assert split_sentences(text) == ["层高 3.6 m", "柱脚铰接", "钢材 Q355"]
    assert split_sentences("  ") == [] and split_sentences(None) == []


def test_statement_items_must_quote_their_sentence_verbatim():
    """抄出的原文必须逐字在所引句子里；改写过的数、编造的轴号、错句号都不采用。"""
    raw = {"grids": {"x": {"axes": [], "spacings": [
        {"from": "1", "to": "2", "text": "6600", "source": {"sentence": 1}},     # 原样
        {"from": "2", "to": "3", "text": "6000", "source": {"sentence": 1}},     # 句里没有 6000
        {"from": "7", "to": "8", "text": "6600", "source": {"sentence": 1}}]}},  # 轴号是编的
        "support": {"kind": "pinned", "text": "柱脚 铰接", "source": {"sentence": 2}},
        "material": {"text": "Q355", "source": {"sentence": 9}}}
    evidence = _statement(raw)
    spacings = evidence["grids"]["x"]["spacings"]
    assert spacings == [{"from": "1", "to": "2", "text": "6600",
                         "source": {"statement": "T1", "sentence": 1}}]
    assert evidence["support"]["source"] == {"statement": "T1", "sentence": 2}
    assert evidence["material"] is None
    unused = [note for note in evidence["notes"] if note.get("unused")]
    assert len(unused) == 3
    assert any("6000" in note["text"] for note in unused)
    assert any("「7」" in note["text"] for note in unused)
    assert any("没有指明出自哪一句" in note["text"] for note in unused)
    assert validate_evidence(evidence) == []


def test_statement_names_known_only_from_drawings_need_confirmation():
    """「所有梁」关联到图上读到的编号是模型做的关联，原句没写出编号，要用户确认。"""
    raw = {"sections": [{"labels": ["KL1", "KL2"], "text": "HN400×200×8×13",
                         "source": {"sentence": 1}}]}
    evidence = _statement(raw, ["所有梁采用 HN400×200×8×13"], known=["KL1", "KL2", "KZ1"])
    assert "请对照说明第 1 句确认" in evidence["sections"][0]["check"]
    evidence = _statement(raw, ["所有梁采用 HN400×200×8×13"], known=["KZ1"])
    assert evidence["sections"] == []


def test_statement_unit_must_match_the_unit_word_in_its_text():
    """单位说明的 value 是模型的理解，必须与原文里的单位词一致；「毫米」不能被填成 m。"""
    good = {"units": {"length": {"value": "mm", "text": "尺寸单位为毫米",
                                 "source": {"sentence": 3}}}}
    bad = deepcopy(good)
    bad["units"]["length"]["value"] = "m"
    assert _statement(good)["units"]["length"]["value"] == "mm"
    evidence = _statement(bad)
    assert evidence["units"] == {} and evidence["notes"][0]["unused"]


def test_statement_conflicting_with_drawing_is_listed_with_both_sources():
    """图上 6000、说明 6600：报冲突并列出两边来源，选了说明那一项后以它为准并记作用户选定。"""
    image = _evidence()
    image["material"]["confirmed"] = True
    statement = _statement({"grids": {"x": {"axes": [], "spacings": [
        {"from": "1", "to": "2", "text": "6600", "source": {"sentence": 1}}]}}})
    merged = merge_evidence([image, statement])
    result = assemble(merged)
    row = _row(result, "grids.x.spacings.0")
    assert row.status == CONFLICT and result.model is None
    assert "图 1「平面图.png」" in row.source and "用户说明第 1 句「1~2 轴间距应为 6600」" in row.source
    assert "不自动以哪一份为准" in row.message
    assert row.options == ("6000（图 1）", "6600（用户说明第 1 句）")
    with pytest.raises(ValueError, match="请填写"):
        resolve(merged, row.id, "")
    chosen = resolve(merged, row.id, _option(row, "用户说明"))
    result = assemble(chosen)
    row = _row(result, "grids.x.spacings.0")
    assert row.status == OK and "用户在 2 份读数中选定" in row.source
    # 总尺寸 12000 与 6600 + 6000 不符：选了说明之后，图上的总尺寸照常核对出来
    assert _row(result, "grids.x.total").status == CONFLICT


def test_equal_readings_from_two_sources_merge_and_list_both_origins():
    """值相同（6000 与 6000 mm）就合并为一行，来源两边都写；不另报冲突。"""
    image = _evidence()
    statement = _statement({"grids": {"x": {"axes": [], "spacings": [
        {"from": "1", "to": "2", "text": "6000 mm", "source": {"sentence": 1}}]}}},
        ["1~2 轴间距 6000 mm"])
    result = assemble(merge_evidence([image, statement]))
    row = _row(result, "grids.x.spacings.0")
    assert row.status == OK and "图 1" in row.source and "用户说明第 1 句" in row.source
    image["material"]["confirmed"] = True
    merged = merge_evidence([image, statement])
    merged["material"]["confirmed"] = True
    assert assemble(merged).ready


def test_unused_statement_needs_acknowledgement_then_stops_blocking():
    """没采用的说明不悄悄丢掉：列为需确认，确认后改为提示，不再阻塞。"""
    image = _evidence()
    image["material"]["confirmed"] = True
    statement = _statement({"material": {"text": "Q420", "source": {"sentence": 2}}})
    merged = merge_evidence([image, statement])
    result = assemble(merged)
    note = next(row for row in result.rows if row.item == "未采用的说明")
    assert note.status == CONFIRM and not result.ready
    result = assemble(resolve(merged, note.id))
    assert next(row for row in result.rows if row.item == "未采用的说明").status == INFO
    assert result.ready


def test_material_grades_with_equal_modulus_still_conflict():
    """Q235 与 Q355 的弹性模量相同，但不是同一种材料：两份证据牌号不同要报冲突。"""
    image = _evidence()
    image["material"]["text"] = "钢材 Q235"
    statement = _statement({"material": {"text": "Q355", "source": {"sentence": 1}}},
                           ["钢材 Q355"])
    row = _row(assemble(merge_evidence([image, statement])), "material")
    assert row.status == CONFLICT and len(row.options) == 2


def test_statement_transcription_sends_only_sentences_and_names(tmp_path):
    """说明抄写只给模型编号后的句子和已读到的名称——图上的数值不进提示词；出处逐句。"""
    from test_drawing_extract import FakeOpenAI, _extractor

    from drawing_extract import STATEMENT_PROMPT, known_names
    reply = {"grids": {"x": {"axes": [], "spacings": [
        {"from": "1", "to": "2", "text": "6600", "source": {"sentence": 1}}]}},
        "support": {"kind": "pinned", "text": "柱脚铰接", "source": {"sentence": 2}}}
    client = FakeOpenAI(["不是 JSON", json.dumps(reply)])
    image = _evidence()
    names = known_names([image])
    result = _extractor(tmp_path, client).transcribe_statements(
        "1~2 轴间距应为 6600。柱脚铰接。", names)
    assert result.success and result.calls == 2
    request = client.chat.requests[0]
    assert request["messages"][0]["content"] == STATEMENT_PROMPT
    sent = request["messages"][1]["content"][0]["text"]
    assert "第 1 句：1~2 轴间距应为 6600" in sent and "第 2 句：柱脚铰接" in sent
    assert "KL1" in sent and "2F" in sent
    assert "7200" not in sent and "3.600" not in sent       # 图上读到的数不给说明抄写
    named_by_value = _evidence()
    named_by_value["levels"][1]["name"] = "3.600"            # DXF 把标高原文当层名
    assert "3.600" not in known_names([named_by_value]) and "1F" in known_names([named_by_value])
    assert "不是 JSON" not in json.dumps(client.chat.requests[1]["messages"])
    evidence = result.evidence
    assert evidence["statements"][0]["sentences"] == ["1~2 轴间距应为 6600", "柱脚铰接"]
    assert evidence["support"]["source"] == {"statement": "T1", "sentence": 2}
    row = _row(assemble(merge_evidence([image, evidence])), "support")
    assert row.status == CONFLICT and "用户说明第 2 句" in row.source


# ---- 多份证据合并 -------------------------------------------------------------------

def test_merge_order_reports_ambiguous_and_contradictory_sequences():
    """轴号先后只按各份证据内部的顺序合并；合不出唯一顺序时如实报出，不按名字猜。"""
    assert merge_order([["1", "2", "3"], ["1", "3"]]) == (["1", "2", "3"], "")
    assert merge_order([["1", "2"], ["3", "4"]])[1] == "ambiguous"
    assert merge_order([["1", "2", "3"], ["3", "1"]])[1] == "cycle"


def _elevation(labels, *, file_id="D1", name="立面.dxf"):
    """只有一组轴线的立面证据（DXF 读取器的形状）。"""
    src = {"file": file_id, "handles": ["2A"], "layer": "轴线"}
    return {"format": EVIDENCE_FORMAT, "images": [],
            "files": [{"id": file_id, "name": name, "sha256": "0", "kind": "dxf"}],
            "grids": {"x": {"axes": [{"label": label, "source": src} for label in labels],
                            "spacings": []}},
            "levels": [], "members": [], "notes": []}


def test_single_axis_view_follows_the_plan_family_and_orientation():
    """立面上的字母轴也是竖线：轴号全属于平面图 y 向时改归 y 向；反向视图按反向合并。"""
    plan = _evidence()
    merged = merge_evidence([plan, _elevation(["B", "A"])])
    assert [axis["label"] for axis in merged["grids"]["y"]["axes"]] == ["A", "B"]
    assert [axis["label"] for axis in merged["grids"]["x"]["axes"]] == ["1", "2", "3"]
    hints = " ".join(note["text"] for note in merged["notes"])
    assert "已按y 向合并" in hints and "反向视图" in hints
    assert "order_problem" not in merged["grids"]["y"]


def test_axes_that_cannot_be_ordered_go_to_the_checklist():
    """两份证据的轴号互不重叠又没有平面图：顺序无法确定，列为冲突。"""
    merged = merge_evidence([_elevation(["1", "2"]), _elevation(["5", "6"], file_id="D2")])
    row = _row(assemble(merged), "grids.x.order")
    assert row.status == CONFLICT and "相对顺序无法确定" in row.message


def test_unnumbered_member_at_a_numbered_position_is_the_same_member():
    """另一视图没写编号的梁与 KL1 位置相同：按同一根合并，不报重叠。"""
    evidence = _evidence()
    evidence["material"]["confirmed"] = True
    before = assemble(evidence).model
    evidence["members"].append({"label": None, "kind": "beam", "along": "A", "from": "1",
                                "to": "2", "levels": ["2F"], "source": evidence["members"][0][
                                    "source"]})
    result = assemble(evidence)
    row = _row(result, f"members.{len(evidence['members']) - 1}")
    assert row.status == OK and "按同一杆件合并" in row.message
    assert result.model == before


def test_user_resolutions_survive_remerge_until_a_new_reading_appears():
    """重读后用户选过的冲突、确认过的材料保留；新证据带来没见过的读数时才重新报冲突。"""
    image = _evidence()
    statement = _statement({"grids": {"x": {"axes": [], "spacings": [
        {"from": "1", "to": "2", "text": "6600", "source": {"sentence": 1}}]}}})
    merged = merge_evidence([image, statement])
    row = _row(assemble(merged), "grids.x.spacings.0")
    merged = resolve(merged, row.id, _option(row, "图 1"))
    merged = resolve(merged, "material")
    again = merge_evidence([image, statement], previous=merged)
    result = assemble(again)
    assert _row(result, "grids.x.spacings.0").status == OK
    assert _row(result, "material").status == OK
    assert again["grids"]["x"]["spacings"][0]["text"] == "6000"
    other = _statement({"grids": {"x": {"axes": [], "spacings": [
        {"from": "1", "to": "2", "text": "6300", "source": {"sentence": 1}}]}}},
        ["1~2 轴间距 6300"])
    other["statements"][0]["id"] = "T2"
    other["grids"]["x"]["spacings"][0]["source"]["statement"] = "T2"
    third = assemble(merge_evidence([image, statement, other], previous=merged))
    row = _row(third, "grids.x.spacings.0")
    assert row.status == CONFLICT and len(row.options) == 2
    assert "6000（图 1）" in row.options and "6300（用户说明第 1 句）" in row.options


# ---- DXF：多个文件、与图片混用 ----------------------------------------------------------

@pytest.fixture
def frames(tmp_path):
    pytest.importorskip("ezdxf", reason="未安装 ezdxf，跳过 DXF 合并测试")
    from test_drawing_dxf import ROLES, _frame_dxf

    from drawing_dxf import read_dxf
    paths = [_frame_dxf(tmp_path / name) for name in ("A 轴立面.dxf", "B 轴立面.dxf")]
    return [read_dxf(path, ROLES, file_id=f"D{index}") for index, path in enumerate(paths, 1)]


def _plan_grid_only():
    """只有轴网、规格、材料与支座的平面图证据（杆件都来自立面 DXF）。"""
    plan = _evidence()
    plan["members"] = []
    plan["material"]["confirmed"] = True
    return plan


def _settle_dxf_units(evidence):
    for row_id in ("units.length",):
        row = next(r for r in assemble(evidence).rows if r.id == row_id)
        if row.status == CONFIRM:
            evidence = resolve(evidence, row_id)
    return evidence


def test_two_elevation_dxfs_and_a_plan_assemble_a_3d_model(frames):
    """A、B 两榀立面 DXF 与平面图合并：各选所在轴线后得到两榀三维框架，出处仍是各自的实体。"""
    merged = merge_evidence([_plan_grid_only(), *frames])
    assert [record["id"] for record in merged["files"]] == ["D1", "D2"]
    result = assemble(merged)
    planes = [row for row in result.rows if row.category == "视图"]
    assert [row.status for row in planes] == [MISSING, MISSING]
    assert planes[0].options == ("A", "B") and "A 轴立面.dxf" in planes[0].item
    assert result.model is None
    merged = resolve(merged, "files.0.plane", "A")
    merged = resolve(merged, "files.1.plane", "B")
    result = assemble(_settle_dxf_units(merged))
    assert result.ready, [(r.id, r.status, r.message) for r in result.pending]
    nodes = {(n["x"], n["y"], n["z"]) for n in result.model["nodes"]}
    assert nodes == {(x, y, z) for x in (0.0, 6.0, 12.0) for y in (0.0, 7.2)
                     for z in (0.0, 3.6, 7.2)}
    assert len(result.model["members"]) == 2 * (6 + 4)
    sources = {row.source for row in result.rows if row.category == "杆件"}
    assert any("A 轴立面.dxf" in text for text in sources)
    assert any("B 轴立面.dxf" in text for text in sources)


def test_dxf_elevation_and_plan_image_with_same_members_merge_into_one(frames):
    """平面图片已有 A 轴框架，立面 DXF 选在 A 轴：同位置同编号的杆件合并，模型与只用图片一样。"""
    image = _evidence()
    image["material"]["confirmed"] = True
    alone = assemble(image).model
    merged = resolve(merge_evidence([image, frames[0]]), "files.0.plane", "A")
    result = assemble(_settle_dxf_units(merged))
    assert result.ready, [(r.id, r.status, r.message) for r in result.pending]
    assert result.model == alone
    units = _row(result, "units.length")
    assert "图 1" in units.source and "DXF「A 轴立面.dxf」" in units.source


def test_dxf_reading_that_differs_from_the_image_is_a_conflict(frames):
    """图上 1→2 写 6600、DXF 标注实测 6000：冲突行写明 DXF 的实体句柄与图片原文，由用户选。"""
    image = _evidence()
    image["grids"]["x"]["spacings"][0]["text"] = "6600"
    merged = resolve(merge_evidence([image, frames[0]]), "files.0.plane", "A")
    row = _row(assemble(merged), "grids.x.spacings.0")
    assert row.status == CONFLICT
    assert "原文「6600」" in row.source and "DXF「A 轴立面.dxf」 实体 #" in row.source
    merged = resolve(merged, row.id, _option(row, "DXF"))
    assert _row(assemble(merged), row.id).status == OK
    assert merged["grids"]["x"]["spacings"][0]["value"] == 6000.0


def test_elevations_without_a_plan_ask_whether_they_are_one_frame(frames):
    """两份立面都没有另一方向轴线：不默认叠在一个平面，问用户；确认同一榀后按同一平面合并。"""
    merged = merge_evidence(frames)
    result = assemble(merged)
    row = _row(result, "views")
    assert row.status == CONFIRM and not result.ready
    assert "A 轴立面.dxf" in row.value and "B 轴立面.dxf" in row.value
    merged = resolve(merged, "views")
    for row_id, value in (("units.elevation", "m"), ("support", "固接"), ("material", None)):
        merged = resolve(merged, row_id, value)
    result = assemble(_settle_dxf_units(merged))
    assert _row(result, "views").status == OK
    assert result.ready, [(r.id, r.status, r.message) for r in result.pending]
    assert len(result.model["members"]) == 6 + 4
    assert _row(assemble(merge_evidence(frames, previous=merged)), "views").status == OK
    # 再加入第三份视图时，之前的确认不再算数
    third = deepcopy(frames[0])
    for key in ("files",):
        third[key][0]["id"] = "D3"
    third = json.loads(json.dumps(third).replace('"file": "D1"', '"file": "D3"'))
    assert _row(assemble(merge_evidence([*frames, third], previous=merged)),
                "views").status == CONFIRM
