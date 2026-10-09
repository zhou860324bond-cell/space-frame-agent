"""防止像素梁轴候选误改无证据几何或绕过人工审核。"""

from copy import deepcopy

import pytest
from PIL import Image, ImageDraw

from sketch_axis_refinement import refine_horizontal_axis
from sketch_parser import _fill_bookkeeping


def _draft():
    return _fill_bookkeeping({
        "image_model": {"nodes": [{"id": 1, "u": 0.12, "v": 0.5},
                                    {"id": 2, "u": 0.88, "v": 0.5}],
                        "members": [{"id": 1, "i": 1, "j": 2}], "supports": [], "load_cases": []},
        "entities": [{"id": "n1", "kind": "node", "target": {"node": 1}, "confidence": 0.9,
                       "image_geometry": {"point": [0.12, 0.5]}},
                      {"id": "m1", "kind": "member", "target": {"member": 1}, "confidence": 0.9}],
    }, "a" * 64, "drawing.png")


def test_black_solid_band_proposes_axis_without_claiming_verified_geometry(tmp_path):
    """混合力/矩公开图的黑色实心梁曾漏过灰色检测，厚带候选须保留原观察和审核阻断。"""
    path = tmp_path / "black.png"
    image = Image.new("RGB", (400, 200), "white")
    ImageDraw.Draw(image).rectangle((60, 84, 340, 92), fill="black")
    image.save(path)
    result = refine_horizontal_axis(_draft(), path)
    assert result["image_model"]["nodes"][0]["v"] * 199 == pytest.approx(88)
    assert result["entities"][0]["position_refinement"]["method"] == "solid-black-horizontal-band/v1"
    assert not result["entities"][0]["verified"]


@pytest.mark.parametrize("drawing", ["thin", "two_black", "sloping_black"])
def test_black_dimension_line_multiple_bands_and_sloping_stroke_are_not_beams(tmp_path, drawing):
    """黑色细尺寸线、多条厚带及倾斜笔画不能强制修正为唯一水平梁。"""
    path = tmp_path / "ambiguous-black.png"
    image = Image.new("RGB", (400, 200), "white")
    paint = ImageDraw.Draw(image)
    if drawing == "thin":
        paint.line((60, 88, 340, 88), fill="black", width=1)
    elif drawing == "two_black":
        paint.rectangle((60, 80, 340, 88), fill="black")
        paint.rectangle((60, 108, 340, 116), fill="black")
    else:
        paint.line((60, 120, 340, 70), fill="black", width=8)
    image.save(path)
    value = _draft()
    assert refine_horizontal_axis(value, path) is value


@pytest.mark.parametrize("size", [(400, 200), (640, 320)])
def test_filled_band_corrects_axis_and_preserves_raw_observation_for_review(tmp_path, size):
    """公开图粗坐标偏离厚梁轴；像素候选应接近真实中心，保留原坐标且不得自动确认。"""
    width, height = size
    image = Image.new("RGB", size, "white")
    bounds = (int(width * 0.15), int(height * 0.42), int(width * 0.85), int(height * 0.48))
    ImageDraw.Draw(image).rectangle(bounds, fill="#999999", outline="black")
    path = tmp_path / "beam.png"
    image.save(path)
    draft = _draft()
    original = deepcopy(draft)
    refined = refine_horizontal_axis(draft, path)
    assert draft == original
    assert abs(refined["image_model"]["nodes"][0]["v"] * (height - 1) - (bounds[1] + bounds[3]) / 2) <= 1
    assert abs(refined["image_model"]["nodes"][0]["u"] * (width - 1) - bounds[0]) <= 2
    assert refined["image_model"]["members"] == original["image_model"]["members"]
    assert refined["image_model"]["supports"] == [] and refined["image_model"]["load_cases"] == []
    observed = next(e for e in refined["entities"] if e["id"] == "n1")
    assert observed["original_observation"] == {"image_geometry": {"point": [0.12, 0.5]}, "confidence": 0.9}
    assert observed["source"] == "derived" and observed["confidence"] is None and not observed["verified"]
    assert refined["issues"][-1]["category"] == "low_confidence"
    assert refined["issues"][-1]["status"] == "open" and refined["issues"][-1]["severity"] == "blocking"


@pytest.mark.parametrize("drawing", ["blank", "open_outline", "diagonal", "two_bands"])
def test_ambiguous_or_unsupported_pixels_never_change_geometry(tmp_path, drawing):
    """空图、未闭合轮廓、斜杆或两条候选梁不能被强制贴到某一条像素轴。"""
    image = Image.new("RGB", (400, 200), "white")
    draw = ImageDraw.Draw(image)
    if drawing == "open_outline":
        draw.line((60, 84, 340, 84), fill="#999999")
        draw.line((60, 96, 340, 96), fill="#999999")
    elif drawing == "diagonal":
        draw.line((60, 120, 340, 70), fill="#999999", width=8)
    elif drawing == "two_bands":
        draw.rectangle((60, 80, 340, 88), fill="#999999")
        draw.rectangle((60, 108, 340, 116), fill="#999999")
    path = tmp_path / "ambiguous.png"
    image.save(path)
    draft = _draft()
    assert refine_horizontal_axis(draft, path) is draft


@pytest.mark.parametrize("review", ["confirmed_scale", "verified_entity", "non_chain"])
def test_confirmed_or_non_chain_geometry_is_never_overwritten(tmp_path, review):
    """像素辅助只处理新识别水平链，不能改人工已确认比例、已审核对象或其他拓扑。"""
    path = tmp_path / "beam.png"
    image = Image.new("RGB", (400, 200), "white")
    ImageDraw.Draw(image).rectangle((60, 84, 340, 96), fill="#999999")
    image.save(path)
    draft = _draft()
    if review == "confirmed_scale":
        draft["scale"].update(status="confirmed", length_per_pixel=0.1)
    elif review == "verified_entity":
        draft["entities"][0]["verified"] = True
    else:
        draft["image_model"]["members"] = []
    assert refine_horizontal_axis(draft, path) is draft


def test_unique_full_image_band_can_propose_a_candidate_for_inverted_vertical_coordinates(tmp_path):
    """真实接口曾把上方梁写到 v=0.7；唯一填充梁带可提出正确候选，但保留原坐标和审核阻断。"""
    path = tmp_path / "inverted.png"
    image = Image.new("RGB", (400, 200), "white")
    ImageDraw.Draw(image).rectangle((60, 40, 340, 52), fill="#999999")
    image.save(path)
    draft = _draft()
    for node in draft["image_model"]["nodes"]:
        node["v"] = 0.7
    refined = refine_horizontal_axis(draft, path)
    assert abs(refined["image_model"]["nodes"][0]["v"] * 199 - 46) <= 1
    assert refined["entities"][0]["position_refinement"]["original_nodes"][0]["v"] == 0.7
    assert not refined["entities"][0]["verified"] and refined["issues"][-1]["status"] == "open"


@pytest.mark.parametrize("size", [(400, 200), (640, 320)])
def test_closed_outline_uses_initial_axis_and_not_a_deformed_curve(tmp_path, size):
    """悬臂图初始梁是空心矩形，黑色变形曲线不能取代灰色初始梁轴。"""
    width, height = size
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    bounds = (int(width*.15), int(height*.4), int(width*.85), int(height*.55))
    draw.rectangle(bounds, outline="#bbbbbb", width=2)
    draw.line((bounds[0], bounds[1], int(width*.5), int(height*.45),
               bounds[2], int(height*.65)), fill="black", width=4)
    path = tmp_path / "initial.png"
    image.save(path)
    draft = _draft()
    refined = refine_horizontal_axis(draft, path)
    assert refined is not draft
    assert abs(refined["image_model"]["nodes"][1]["v"]*(height-1)-(bounds[1]+bounds[3])/2) <= 2
    assert refined["entities"][0]["position_refinement"]["method"] == "closed-horizontal-outline/v1"
    assert refined["entities"][0]["confidence"] is None and refined["issues"][-1]["severity"] == "blocking"


@pytest.mark.parametrize("drawing", ["two_outlines", "distant_outline", "sloping_outline"])
def test_multiple_distant_or_sloping_outlines_do_not_choose_a_beam(tmp_path, drawing):
    """辅助图矩形、离原观察很远的轮廓和倾斜图不能被当成唯一初始梁。"""
    image = Image.new("RGB", (400, 200), "white")
    draw = ImageDraw.Draw(image)
    if drawing == "two_outlines":
        draw.rectangle((60, 75, 340, 90), outline="#bbbbbb")
        draw.rectangle((60, 110, 340, 125), outline="#bbbbbb")
    elif drawing == "distant_outline":
        draw.rectangle((60, 20, 340, 35), outline="#bbbbbb")
    else:
        draw.polygon(((60, 80), (340, 100), (340, 115), (60, 95)), outline="#bbbbbb")
    path = tmp_path / "auxiliary.png"
    image.save(path)
    draft = _draft()
    assert refine_horizontal_axis(draft, path) is draft


@pytest.mark.parametrize("name", ["cantilever", "udl", "point_load"])
def test_public_outline_and_small_jpeg_supports_keep_original_objects(name):
    """已有真实失败图必须校正初始轮廓或 JPEG 支座顶端，并保留观察与计算对象不变。"""
    import json
    from pathlib import Path
    from multimodal_contract import point_match_threshold
    root = Path(__file__).resolve().parents[1] / "multimodal_eval/public_cases"
    response = json.loads((root / "round_03/responses" / f"{name}.json").read_text(encoding="utf-8"))
    truth = json.loads((root / "ground_truth" / f"{name}.json").read_text(encoding="utf-8"))
    image = next((root / "images").glob(name + ".*"))
    draft = _fill_bookkeeping(json.loads(response["raw_responses"][0]), truth["image_hash"], str(image))
    original = deepcopy(draft)
    refined = refine_horizontal_axis(draft, image)
    tolerance = point_match_threshold(truth["width_px"], truth["height_px"])
    for expected in truth["nodes"]:
        got = next(n for n in refined["image_model"]["nodes"] if n["id"] == expected["id"])
        error = ((got["u"]-expected["point"][0])*truth["width_px"])**2 + ((got["v"]-expected["point"][1])*truth["height_px"])**2
        assert error**.5 <= tolerance
    assert draft == original
    assert refined["image_model"]["supports"] == original["image_model"]["supports"]
    assert refined["image_model"]["load_cases"] == original["image_model"]["load_cases"]
    for entity in refined["entities"]:
        if entity["kind"] == "support":
            node = next(n for n in refined["image_model"]["nodes"] if n["id"] == entity["target"]["support"]["node"])
            assert entity["image_geometry"]["point"] == [node["u"], node["v"]]
            assert entity["original_observation"] and not entity["verified"]


@pytest.mark.parametrize("symbol", ["triangle", "down_arrow", "line", "two_triangles"])
def test_antialiased_support_apex_requires_a_unique_upward_expanding_contour(tmp_path, symbol):
    """灰色支座边缘需保留顶端，但细线、反向箭头或两处候选不能拉偏附着节点。"""
    image = Image.new("RGB", (400, 200), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((60, 84, 340, 96), fill="#999999", outline="black")
    if symbol in ("triangle", "two_triangles"):
        draw.polygon(((335, 97), (329, 109), (341, 109)), outline="#aaaaaa")
        if symbol == "two_triangles":
            draw.polygon(((348, 97), (344, 106), (352, 106)), outline="#aaaaaa")
    elif symbol == "down_arrow":
        draw.polygon(((329, 97), (341, 97), (335, 109)), outline="#aaaaaa")
    else:
        draw.line((335, 97, 335, 109), fill="#aaaaaa", width=2)
    path = tmp_path / "symbol.png"
    image.save(path)
    draft = _draft()
    draft["image_model"]["supports"] = [{"node": 2, "fix": [1, 1, 1, 0, 0, 0]}]
    refined = refine_horizontal_axis(draft, path)
    endpoint = refined["image_model"]["nodes"][1]["u"] * 399
    assert abs(endpoint - (335 if symbol == "triangle" else 339)) <= 2
