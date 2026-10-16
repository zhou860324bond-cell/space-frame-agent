"""图纸图片 → 文字证据：两遍调用、坐标换回原图、校验失败回喂重读。

全部用假客户端回放，不调用真实模型。防的问题：整图缩小后小字读错（所以要
裁剪原分辨率区域）、裁剪后出处坐标仍是区域坐标导致核对表高亮错位、抄错的
间距被直接放行。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from drawing_extract import DrawingExtractor, image_records
from sketch_parser import SketchParser

PIL = pytest.importorskip("PIL.Image")


class FakeChat:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.completions = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        text = self.replies.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class FakeOpenAI:
    def __init__(self, replies):
        self.chat = FakeChat(replies)


class FakeAnthropic:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.messages = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(text=self.replies.pop(0))])


def _drawing(tmp_path, name="平面图.png", size=(2000, 1000)):
    path = tmp_path / name
    PIL.new("RGB", size, "white").save(path)
    return path


def _src(region, bbox=(0.1, 0.1, 0.2, 0.2), conf=0.95):
    return {"region": region, "bbox": list(bbox), "confidence": conf}


def _raw(second_spacing="6000"):
    """区域 1 是左半张平面图，区域 2 是右上角的构件表与说明。"""
    return {
        "units": {"length": {"value": "mm", "text": "尺寸以毫米计", "source": _src(2)},
                  "elevation": {"value": "m", "text": "标高以米计", "source": _src(2)}},
        "grids": {"x": {"axes": [{"label": "1", "pos": 0.2, "source": _src(1)},
                                 {"label": "2", "pos": 0.5, "source": _src(1)},
                                 {"label": "3", "pos": 0.8, "source": _src(1)}],
                        "spacings": [{"from": "1", "to": "2", "text": "6000",
                                      "source": _src(1)},
                                     {"from": "2", "to": "3", "text": second_spacing,
                                      "source": _src(1)}],
                        "total": {"text": "12000", "source": _src(1)}},
                  "y": {"axes": [], "spacings": []}},
        "levels": [{"name": "±0.000", "text": "±0.000", "source": _src(1)},
                   {"name": "3.600", "text": "3.600", "source": _src(1)}],
        "members": [{"label": "KZ1", "kind": "column", "at": [x], "from_level": "±0.000",
                     "to_level": "3.600", "source": _src(1)} for x in ("1", "2", "3")]
        + [{"label": "KL1", "kind": "beam", "from": "1", "to": "3", "levels": ["3.600"],
            "source": _src(1)}],
        "sections": [{"label": "KZ1", "text": "HW300×300×10×15", "source": _src(2)},
                     {"label": "KL1", "text": "HN400×200×8×13", "source": _src(2)}],
        "material": {"text": "Q355", "source": _src(2)},
        "support": {"kind": "fixed", "text": "柱脚刚接", "source": _src(2)},
    }


LAYOUT = json.dumps({"regions": [
    {"image": 1, "kind": "plan", "bbox": [0.0, 0.0, 0.5, 1.0]},
    {"image": 1, "kind": "section_table", "bbox": [0.5, 0.0, 1.0, 0.5]}]})


def _extractor(tmp_path, client, provider="openai"):
    parser = SketchParser(provider=provider, api_key="test")
    parser._client = client
    return DrawingExtractor(parser, tmp_path / "work")


def test_two_pass_crops_regions_and_maps_sources_back(tmp_path):
    """第二遍送的是原分辨率裁剪区域；出处与轴线位置换回原图坐标。"""
    image = _drawing(tmp_path)
    client = FakeOpenAI([LAYOUT, json.dumps(_raw())])
    result = _extractor(tmp_path, client).extract([image])
    assert result.success and result.calls == 2
    assert [r.kind for r in result.regions] == ["plan", "section_table"]
    # 左半张外扩 2%：宽 0~1040 像素，裁剪图按原分辨率保存
    assert result.regions[0].origin == (0, 0) and result.regions[0].size == (1040, 1000)
    with PIL.open(result.regions[0].path) as crop:
        assert crop.size == (1040, 1000)
    second = client.chat.requests[1]["messages"][1]["content"]
    assert sum(block["type"] == "image_url" for block in second) == 2
    ev = result.evidence
    assert ev["images"][0]["name"] == "平面图.png" and ev["images"][0]["width"] == 2000
    axis = ev["grids"]["x"]["axes"][1]
    assert axis["pos"] == pytest.approx(0.5 * 1040 / 2000)
    assert axis["source"]["image"] == "P1" and "region" not in axis["source"]
    material_bbox = ev["material"]["source"]["bbox"]
    assert material_bbox[0] == pytest.approx((960 + 0.1 * 1040) / 2000)


def test_dimension_chain_conflict_is_sent_back_for_rereading(tmp_path):
    """分段和与总尺寸不符时回喂重读，第二次读对后采用第二次结果。"""
    image = _drawing(tmp_path)
    client = FakeOpenAI([LAYOUT, json.dumps(_raw("8000")), json.dumps(_raw("6000"))])
    result = _extractor(tmp_path, client).extract([image], max_repairs=1)
    assert result.calls == 3
    correction = client.chat.requests[2]["messages"][1]["content"][-2]["text"]
    assert "14000" in correction and "不要编造" in correction
    assert result.evidence["grids"]["x"]["spacings"][1]["text"] == "6000"
    assert any("需要复核" in e for e in result.errors)


def test_unrepaired_evidence_is_still_returned_for_user_review(tmp_path):
    """重读后仍不闭合时不丢弃结果，交给核对表由用户改正。"""
    image = _drawing(tmp_path)
    client = FakeOpenAI([LAYOUT, json.dumps(_raw("8000")), json.dumps(_raw("8000"))])
    result = _extractor(tmp_path, client).extract([image], max_repairs=1)
    assert result.success
    assert result.evidence["grids"]["x"]["spacings"][1]["text"] == "8000"


def test_layout_failure_falls_back_to_whole_image(tmp_path):
    """定位那一遍没给出 JSON 时整图读取，并把原因记下。"""
    image = _drawing(tmp_path)
    raw = _raw()
    for item in raw["sections"] + [raw["material"], raw["support"]]:
        item["source"]["region"] = 1
    raw["units"]["length"]["source"]["region"] = 1
    raw["units"]["elevation"]["source"]["region"] = 1
    client = FakeOpenAI(["无法判断", json.dumps(raw)])
    result = _extractor(tmp_path, client).extract([image])
    assert result.success
    assert [r.kind for r in result.regions] == ["whole"]
    assert any("整图读取" in e for e in result.errors)


def test_anthropic_provider_sends_image_blocks(tmp_path):
    """Anthropic 接口用 base64 图片块，同一套两遍流程。"""
    image = _drawing(tmp_path)
    client = FakeAnthropic([LAYOUT, json.dumps(_raw())])
    result = _extractor(tmp_path, client, provider="anthropic").extract([image])
    assert result.success
    content = client.requests[1]["messages"][0]["content"]
    assert sum(block["type"] == "image" for block in content) == 2


def test_provider_errors_are_reported_not_raised(tmp_path):
    """网络或密钥错误原样报给用户，不让后台线程抛出。"""
    image = _drawing(tmp_path)
    parser = SketchParser(provider="openai", api_key="")
    result = DrawingExtractor(parser, tmp_path / "work").extract([image])
    assert not result.success
    assert "缺少 API 密钥" in result.errors[-1]


def test_image_records_hash_and_size(tmp_path):
    """出处锚点记录文件名、哈希与像素尺寸，换了图能被发现。"""
    image = _drawing(tmp_path, size=(640, 480))
    record = image_records([image])[0]
    assert (record["id"], record["width"], record["height"]) == ("P1", 640, 480)
    assert len(record["sha256"]) == 64
