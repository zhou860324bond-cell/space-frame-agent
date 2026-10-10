"""圆弧中心候选不能取箭头尖，或从灰度、短弧和多个圆中强选位置。"""

from copy import deepcopy

import pytest
from PIL import Image, ImageDraw

from sketch_moment_refinement import moment_centres
from sketch_action_review import apply_action_review
from test_sketch_action_review import draft, review


@pytest.mark.parametrize("colour", ["#2277dd", "#dd2244", "#229955"])
def test_circle_with_arrowhead_uses_centre_independent_of_ink_colour(tmp_path, colour):
    """已观察为力矩的圆弧要取中心；填充箭头头部和颜色不能决定作用位置。"""
    path = tmp_path / "moment.png"
    image = Image.new("RGB", (700, 400), "white")
    paint = ImageDraw.Draw(image)
    paint.arc((448, 168, 512, 232), start=210, end=500, fill=colour, width=3)
    paint.polygon([(454, 220), (468, 228), (463, 213)], fill=colour)
    image.save(path)
    actions = [{"id": "M", "kind": "moment", "point": [512 / 699, 200 / 399]}]
    original = deepcopy(actions)
    evidence = moment_centres(actions, path)["M"]
    assert evidence["centre_px"] == pytest.approx([480, 200], abs=2)
    assert evidence["angular_bins"] >= 14
    assert actions == original


@pytest.mark.parametrize("shape", ["grey", "arrow", "short_arc", "two_circles", "ellipse", "cropped"])
def test_unsupported_or_ambiguous_marks_keep_original_position(tmp_path, monkeypatch, shape):
    """不能用灰度圆、直箭头、短弧、双圆、扁椭圆或被局部窗截断的笔画自动改力矩位置。"""
    path = tmp_path / "unknown.png"
    image = Image.new("RGB", (700, 400), "white")
    paint = ImageDraw.Draw(image)
    if shape == "grey":
        paint.ellipse((448, 168, 512, 232), outline="black", width=3)
    elif shape == "arrow":
        paint.line((480, 150, 480, 240), fill="blue", width=3)
        paint.polygon([(470, 225), (490, 225), (480, 240)], fill="blue")
    elif shape == "short_arc":
        paint.arc((448, 168, 512, 232), start=10, end=80, fill="blue", width=3)
    elif shape == "two_circles":
        paint.ellipse((468, 182, 504, 218), outline="blue", width=2)
        paint.ellipse((514, 182, 550, 218), outline="red", width=2)
    elif shape == "ellipse":
        paint.ellipse((450, 180, 550, 220), outline="blue", width=3)
    else:
        paint.ellipse((420, 110, 600, 290), outline="blue", width=3)
    image.save(path)
    candidates = []
    if shape == "two_circles":
        import sketch_moment_refinement
        original = sketch_moment_refinement._fit_circle
        def fit(*args):
            result = original(*args)
            if result is not None:
                candidates.append(result)
            return result
        monkeypatch.setattr(sketch_moment_refinement, "_fit_circle", fit)
    assert moment_centres([{"id": "M", "kind": "moment", "point": [510 / 699, 200 / 399]}], path) == {}
    if shape == "two_circles":
        assert len(candidates) == 2


def test_force_and_transparent_background_do_not_become_moment_evidence(tmp_path):
    """只处理力矩观察，透明像素中的隐藏颜色不能被识别成圆弧。"""
    path = tmp_path / "transparent.png"
    image = Image.new("RGBA", (700, 400), (0, 0, 255, 0))
    image.save(path)
    assert moment_centres([{"id": "M", "kind": "moment", "point": [.7, .5]}], path) == {}
    assert moment_centres([{"id": "F", "kind": "force", "point": [.7, .5]}], path) == {}


def test_pixel_circle_proposal_preserves_raw_observation_and_verified_protection(tmp_path):
    """像素圆心经梁轴投影提出候选，模型原点不改写，已人工确认的草稿不能重拟合。"""
    path = tmp_path / "beam.png"
    image = Image.new("RGB", (1000, 200), "white")
    ImageDraw.Draw(image).ellipse((650, 80, 690, 120), outline="blue", width=2)
    image.save(path)
    value = draft()
    observation = review((.690, .500))
    observation["actions"][0]["kind"] = "moment"
    result = apply_action_review(value, observation, image_path=path)
    record = result["action_review"]["records"][0]
    assert record["observation"]["point"] == [.690, .500]
    assert record["pixel_refinement"]["centre_px"][0] == pytest.approx(670, abs=2)
    assert result["image_model"]["nodes"][-1]["u"] == pytest.approx(670 / 999, abs=.002)
    value["entities"][0]["verified"] = True
    assert apply_action_review(value, observation, image_path=path)["image_model"] == value["image_model"]
