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


@pytest.mark.parametrize("drawing", ["blank", "outline", "diagonal", "two_bands"])
def test_ambiguous_or_unsupported_pixels_never_change_geometry(tmp_path, drawing):
    """空图、细轮廓、斜杆或两条候选梁不能被强制贴到某一条像素轴。"""
    image = Image.new("RGB", (400, 200), "white")
    draw = ImageDraw.Draw(image)
    if drawing == "outline":
        draw.rectangle((60, 84, 340, 96), outline="#999999")
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
