"""手绘草图解析（sketch_parser）的单元测试。

由于不能真实调用多模态 LLM API，这里用 mock 替换 LLM 调用，测试：
1. 图片编码、JSON 提取、模型校验等辅助函数
2. parse 单次调用成功/失败
3. parse_with_retry 自修复闭环（第一次失败、第二次成功）
4. 全部重试失败时返回失败结果
5. 配置校验（不支持的提供商、缺密钥等）
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from recognition_draft import DRAFT_FORMAT
from sketch_parser import SKETCH_SYSTEM_PROMPT, SketchParser, ParseResult


# 一个合法的模型 JSON
VALID_MODEL = {
    "units": "N-m-Pa",
    "materials": [{"name": "Q355", "E": 2.06e11, "nu": 0.3, "density": 7850.0}],
    "sections": [
        {"name": "COL", "A": 0.0147, "Iy": 4.2e-5, "Iz": 1.18e-3, "J": 9e-7},
        {"name": "BEAM", "A": 0.010, "Iy": 4.0e-5, "Iz": 3.0e-4, "J": 8e-7},
    ],
    "nodes": [
        {"id": 1, "x": 0, "y": 0, "z": 0},
        {"id": 2, "x": 0, "y": 0, "z": 3.6},
        {"id": 3, "x": 6.0, "y": 0, "z": 3.6},
        {"id": 4, "x": 6.0, "y": 0, "z": 0},
    ],
    "members": [
        {"id": 1, "i": 1, "j": 2, "section": "COL", "material": "Q355"},
        {"id": 2, "i": 2, "j": 3, "section": "BEAM", "material": "Q355"},
        {"id": 3, "i": 3, "j": 4, "section": "COL", "material": "Q355"},
    ],
    "supports": [
        {"node": 1, "fix": [1, 1, 1, 1, 1, 1]},
        {"node": 4, "fix": [1, 1, 1, 1, 1, 1]},
    ],
    "load_cases": [{"name": "DL", "member_loads": [
        {"member": 2, "w": [0, 0, -10e3]}]}],
}


@pytest.fixture
def temp_image(tmp_path):
    """创建一个临时的小 PNG 图片（1x1 像素）。"""
    # 最小的有效 PNG（1x1 红色像素）
    png_bytes = bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753"
        "de0000000c4944415408d763f8cfc00000000300015c37155c0000000049454e44ae426082"
    )
    path = tmp_path / "sketch.png"
    path.write_bytes(png_bytes)
    return str(path)


# --------------------------------------------------------- 配置校验

def test_create_parser():
    parser = SketchParser(provider="openai", api_key="test-key")
    assert parser._provider == "openai"
    assert parser._model == "gpt-4o"


def test_unsupported_provider_raises():
    with pytest.raises(ValueError, match="不支持的提供商"):
        SketchParser(provider="nonexistent", api_key="test-key")


def test_missing_api_key_raises_on_call(temp_image):
    parser = SketchParser(provider="openai", api_key="")
    with pytest.raises(ValueError, match="缺少 API 密钥"):
        parser.parse(temp_image)


# --------------------------------------------------------- 辅助函数

def test_encode_image(temp_image):
    data_url = SketchParser._encode_image(temp_image)
    assert data_url.startswith("data:image/png;base64,")
    # base64 部分非空
    assert len(data_url.split(",", 1)[1]) > 0


def test_encode_nonexistent_image_raises():
    with pytest.raises(FileNotFoundError):
        SketchParser._encode_image("/nonexistent/image.png")


def test_encode_unsupported_image_raises(tmp_path):
    path = tmp_path / "sketch.tiff"
    path.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="不支持的图片格式"):
        SketchParser._encode_image(path)


def test_anthropic_image_block_uses_its_native_format(temp_image):
    data_url = SketchParser._encode_image(temp_image)
    block = SketchParser._anthropic_image(data_url)
    assert block["type"] == "image"
    assert block["source"]["type"] == "base64"
    assert block["source"]["media_type"] == "image/png"


def test_anthropic_call_sends_a_native_image_block(temp_image):
    """Anthropic 不接受 OpenAI 的 image_url 结构。"""
    captured = {}

    class Messages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return type("Response", (), {
                "content": [type("Block", (), {"text": "{}"})()]})()

    parser = SketchParser(provider="anthropic", api_key="test-key")
    parser._client = type("Client", (), {"messages": Messages()})()
    parser._call_llm(SketchParser._encode_image(temp_image), "缺少支座")
    content = captured["messages"][0]["content"]
    image = next(part for part in content if part["type"] == "image")
    assert image["source"]["type"] == "base64"
    assert all(part["type"] != "image_url" for part in content)


def test_extract_json_direct():
    text = json.dumps(VALID_MODEL)
    result = SketchParser._extract_json(text)
    assert result["nodes"][0]["id"] == 1


def test_extract_json_from_markdown_block():
    text = f"```json\n{json.dumps(VALID_MODEL)}\n```"
    result = SketchParser._extract_json(text)
    assert len(result["members"]) == 3


def test_extract_json_with_surrounding_text():
    text = f"这是识别结果：\n{json.dumps(VALID_MODEL)}\n希望对你有帮助。"
    result = SketchParser._extract_json(text)
    assert result["units"] == "N-m-Pa"


def test_extract_json_invalid_raises():
    with pytest.raises(ValueError, match="无法从响应中提取 JSON"):
        SketchParser._extract_json("这不是 JSON")


def test_validate_valid_model():
    errors = SketchParser._validate_model(VALID_MODEL)
    assert errors == [] or isinstance(errors, list)


def test_validate_invalid_model():
    errors = SketchParser._validate_model({"nodes": []})
    assert len(errors) > 0


def test_prompt_forbids_silent_engineering_defaults():
    assert "不得补默认材料" in SKETCH_SYSTEM_PROMPT
    assert "默认支座" in SKETCH_SYSTEM_PROMPT
    assert "绝不能把相对坐标假装成米" in SKETCH_SYSTEM_PROMPT
    assert DRAFT_FORMAT in SKETCH_SYSTEM_PROMPT


# --------------------------------------------------------- parse（单次调用）

def test_parse_success(temp_image, monkeypatch):
    parser = SketchParser(provider="openai", api_key="test-key")
    # mock LLM 调用
    response = {
        "format": DRAFT_FORMAT,
        "scale": {"status": "confirmed", "evidence": "图中标注尺寸"},
        "model": VALID_MODEL,
        "entities": [], "questions": [], "warnings": [],
    }
    monkeypatch.setattr(parser, "_call_llm", lambda img, corr="": json.dumps(response))
    model = parser.parse(temp_image)
    assert model["units"] == "N-m-Pa"
    assert len(model["nodes"]) == 4


def test_parse_legacy_response_cannot_bypass_scale_confirmation(temp_image, monkeypatch):
    parser = SketchParser(provider="openai", api_key="test-key")
    monkeypatch.setattr(parser, "_call_llm", lambda img, corr="": json.dumps(VALID_MODEL))

    with pytest.raises(ValueError, match="尺度尚未确认"):
        parser.parse(temp_image)


def test_parse_validation_failure_raises(temp_image, monkeypatch):
    parser = SketchParser(provider="openai", api_key="test-key")
    invalid = {"nodes": [], "members": [], "supports": []}
    monkeypatch.setattr(parser, "_call_llm", lambda img, corr="": json.dumps(invalid))
    with pytest.raises(ValueError, match="识别草稿无效"):
        parser.parse(temp_image)


# --------------------------------------------------------- parse_with_retry（自修复）

def test_parse_with_retry_first_try_success(temp_image, monkeypatch):
    parser = SketchParser(provider="openai", api_key="test-key")
    call_count = [0]

    def mock_call(img, correction=""):
        call_count[0] += 1
        return json.dumps(VALID_MODEL)

    monkeypatch.setattr(parser, "_call_llm", mock_call)
    result = parser.parse_with_retry(temp_image, max_retries=3)
    assert result.success is True
    assert result.attempts == 1
    assert call_count[0] == 1
    assert result.model is not None
    assert result.draft is not None
    assert result.model["units"] == "N-m-Pa"


def test_parse_with_retry_self_healing(temp_image, monkeypatch):
    """第一次返回不合法模型，第二次返回合法模型——验证自修复闭环。"""
    parser = SketchParser(provider="openai", api_key="test-key")
    call_count = [0]

    def mock_call(img, correction=""):
        call_count[0] += 1
        if call_count[0] == 1:
            # 第一次：杆件引用不存在的节点，属于坏拓扑，必须重试
            invalid = json.loads(json.dumps(VALID_MODEL))
            invalid["members"][0]["j"] = 99
            return json.dumps(invalid)
        else:
            # 第二次：修正后的合法模型
            assert correction, "第二次调用应该带上第一次的错误信息"
            return json.dumps(VALID_MODEL)

    monkeypatch.setattr(parser, "_call_llm", mock_call)
    result = parser.parse_with_retry(temp_image, max_retries=3)
    assert result.success is True
    assert result.attempts == 2
    assert call_count[0] == 2
    assert len(result.errors) == 1  # 第一次的错误


def test_parse_with_retry_all_fail(temp_image, monkeypatch):
    parser = SketchParser(provider="openai", api_key="test-key")

    def mock_call(img, correction=""):
        return "这不是 JSON"

    monkeypatch.setattr(parser, "_call_llm", mock_call)
    result = parser.parse_with_retry(temp_image, max_retries=2)
    assert result.success is False
    assert result.attempts == 2
    assert result.model is None
    assert len(result.errors) == 2
    assert result.duration_ms >= 0


def test_parse_with_retry_json_then_validation_fail(temp_image, monkeypatch):
    """JSON 能解析但模型校验一直失败。"""
    parser = SketchParser(provider="openai", api_key="test-key")
    invalid = {"nodes": [], "members": [], "supports": []}

    monkeypatch.setattr(parser, "_call_llm", lambda img, corr="": json.dumps(invalid))
    result = parser.parse_with_retry(temp_image, max_retries=2)
    assert result.success is False
    assert result.attempts == 2


# --------------------------------------------------------- ParseResult

def test_parse_result_defaults():
    r = ParseResult()
    assert r.success is False
    assert r.attempts == 0
    assert r.errors == []
    assert r.model is None
    assert r.draft is None


def test_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-env-key")
    parser = SketchParser.from_env("openai")
    assert parser._api_key == "test-env-key"
    assert parser._provider == "openai"


def test_from_env_deepseek(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    parser = SketchParser.from_env("deepseek")
    assert parser._api_key == "deepseek-key"
    # DeepSeek 平台上没有 deepseek-vl（那是开源权重名）。官方文档里可接收
    # 图片的模型是 deepseek-v4-flash-vision-exp；用错名字报的是 model not
    # found，很容易被误读成"多模态没跑通"。
    assert parser._model == "deepseek-v4-flash-vision-exp"


def test_from_env_deepseek_falls_back_to_the_key_file(monkeypatch, tmp_path):
    """环境变量没有时要读仓库根目录的 deepseek.key。

    此前只看环境变量，本地明明放了密钥文件，冒烟测试仍报无凭据。
    """
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    import credentials
    monkeypatch.setattr(credentials, "load_api_key", lambda *a, **k: "from-file")
    assert SketchParser.from_env("deepseek")._api_key == "from-file"


def test_from_env_keeps_the_requested_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-env-key")
    parser = SketchParser.from_env("openai", model="gpt-4.1-mini")
    assert parser._model == "gpt-4.1-mini"


# ------------------------- 真实模型响应回归（2026-09-09 实测）

def _real_deepseek_response() -> dict:
    """一次真实的 deepseek-v4-flash-vision-exp 响应。

    它把柱、梁、荷载和尺寸标注都认出来了，却因为几个**簿记字段**被判不合法：
    format 没给、source 写成了 title/type/author、work_plane 和 scale 被压成
    字符串。这些全是代码自己知道的东西，不该让模型去凑。
    """
    import json
    from pathlib import Path
    path = Path(__file__).parent / "fixtures" / "deepseek_v2_real_response.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_bookkeeping_fields_are_filled_by_code_not_the_model():
    """簿记字段由代码补齐后，这几条格式错误必须消失。"""
    from sketch_parser import _fill_bookkeeping
    from multimodal_workflow import V2_DRAFT_FORMAT, validate_v2_draft

    raw = _real_deepseek_response()
    assert "format" in str(validate_v2_draft(dict(raw)))      # 修复前确实报这个

    filled = _fill_bookkeeping(raw, "the-caller-hash", "a.png")
    errors = " ".join(validate_v2_draft(filled))
    assert "format" not in errors
    assert "work_plane" not in errors
    assert "scale" not in errors
    assert filled["format"] == V2_DRAFT_FORMAT
    assert filled["revision"] == 0 and filled["confirmation"] is None


def test_image_hash_is_written_by_the_caller_never_echoed_by_the_model():
    """哈希绑定必须由代码写入。

    让模型把调用方给的哈希抄回来，既没有增加任何保证，还留了抄错或自己编一个
    的余地——真实响应里它填的就是一个对不上的哈希。代码写入是更强的绑定。
    """
    from sketch_parser import _fill_bookkeeping
    raw = _real_deepseek_response()
    filled = _fill_bookkeeping(raw, "authoritative-hash", "a.png")
    assert filled["source"]["image_hash"] == "authoritative-hash"
    # 模型对图纸的描述是有用线索，不该连带丢掉
    assert filled["source"]["title"] == "两层两跨框架 立面图"


def test_v2_prompt_shows_the_shape_it_demands():
    """提示词必须给出显式骨架。

    上一版只罗列了十二个字段名、不给结构，模型只能猜——实测它把 image_model
    理解成了图片元信息（哈希、尺寸、分辨率），结构全塞进了 entities。
    字段名歧义要靠提示词消除，不能靠事后纠错。
    """
    from sketch_parser import V2_SKETCH_SYSTEM_PROMPT as prompt
    assert '"u":' in prompt and '"v":' in prompt, "必须写明归一化 u/v 坐标"
    assert "不是图片的元信息" in prompt
    assert "最小单元" in prompt, "必须说明跨层的柱要拆成多根杆件"


def test_issues_are_normalised_into_objects():
    """issues 必须规整成对象，模型给字符串也不能让界面崩。

    v2 的 issue 带 category/severity/entity_refs/status，界面会去读这些字段。
    模型很容易只吐一句话——那样 item.get(...) 直接
    AttributeError: 'str' object has no attribute 'get'，整个面板崩掉。
    **模型输出的形状永远不可信，消费前必须规整。**
    """
    from sketch_parser import _fill_bookkeeping
    out = _fill_bookkeeping(
        {"issues": ["支座看不清", {"message": "x", "status": "weird"}]},
        "h", "a.png")
    assert all(isinstance(item, dict) for item in out["issues"])
    first, second = out["issues"]
    assert first["message"] == "支座看不清"
    assert first["status"] == "open" and first["entity_refs"] == []
    assert second["status"] == "open", "非法状态要落回 open，不能原样带进界面"
    # 界面就是这样读的，规整之后不能再抛
    assert {(i.get("category"), tuple(i.get("entity_refs") or []))
            for i in out["issues"] if i.get("status") == "open"}


def test_prompt_shows_issues_as_objects_not_strings():
    """提示词里的 issues 示例必须是对象。

    上一版写成了字符串数组，模型照做，界面立刻崩——提示词里的示例就是契约。
    """
    from sketch_parser import V2_SKETCH_SYSTEM_PROMPT as prompt
    assert '"category"' in prompt and '"entity_refs"' in prompt
    assert '"issues":    ["' not in prompt, "不能再是字符串数组"


def test_supports_kind_is_translated_into_a_fix_mask():
    """模型给 kind，代码给掩码。

    认符号是视觉活（"三角形=铰接"），把它翻成 [1,1,1,0,0,0] 是编码活。
    项目里所有消费方读的都是 fix；缺了它 model_tree 会 KeyError: 'fix'，
    并把整棵树的重建、连带整个界面刷新一起带崩——实测就是这么崩的。
    """
    from sketch_parser import _fill_bookkeeping
    out = _fill_bookkeeping({"image_model": {"supports": [
        {"node": 1, "kind": "fixed"}, {"node": 2, "kind": "pinned"},
        {"node": 3, "kind": "roller"}, {"node": 4, "kind": "看不清"},
        {"node": 5, "fix": [1, 0, 1, 0, 0, 0]},
    ]}}, "h", "a.png")
    masks = [s["fix"] for s in out["image_model"]["supports"]]
    assert masks[0] == [1, 1, 1, 1, 1, 1]
    assert masks[1] == [1, 1, 1, 0, 0, 0]
    assert masks[2] == [0, 1, 1, 0, 0, 0]
    assert masks[3] == [1, 1, 1, 0, 0, 0], "认不出类型时按铰接保守处理"
    assert out["image_model"]["supports"][3]["name"] == "待确认支座"
    assert masks[4] == [1, 0, 1, 0, 0, 0], "已经给了 fix 的不许被覆盖"
    # 每一条都要能被界面直接消费：六个 0/1，classify_support 就是这么读的
    for mask in masks:
        assert len(mask) == 6 and all(v in (0, 1) for v in mask)
