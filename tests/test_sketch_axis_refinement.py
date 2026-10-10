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



def _joint_picture(tmp_path, scale=1):
    """独立绘制三个灰色杆身/节点圆点，不使用公开图坐标调测试图。"""
    image = Image.new("RGB", (360 * scale, 220 * scale), "white")
    paint = ImageDraw.Draw(image)
    points = [(50 * scale, 170 * scale), (180 * scale, 65 * scale), (310 * scale, 170 * scale)]
    for a, b in ((0, 1), (1, 2), (0, 2)):
        paint.line((*points[a], *points[b]), fill=(210, 210, 210), width=12 * scale)
    for x, y in points:
        paint.ellipse((x-7*scale, y-7*scale, x+7*scale, y+7*scale), fill=(220, 220, 220), outline="black")
        paint.ellipse((x-2*scale, y-2*scale, x+2*scale, y+2*scale), fill="black")
    path = tmp_path / "joints.png"
    image.save(path)
    w, h = image.size
    value = _fill_bookkeeping({"image_model": {
        "nodes": [{"id": i, "u": (x + 5*scale)/(w-1), "v": (y + 8*scale)/(h-1)}
                  for i, (x, y) in enumerate(points, 1)],
        "members": [{"id": 1, "i": 1, "j": 2}, {"id": 2, "i": 2, "j": 3}, {"id": 3, "i": 1, "j": 3}],
        "supports": [{"node": 1, "kind": "pinned"}],
        "load_cases": [{"name": "D", "nodal_loads": [{"node": 2, "name": "P1", "load": [0, -6000, 0, 0, 0, 0]}]}]},
        "entities": [{"id": "load", "kind": "load", "target": {"load": {"case": "D", "collection": "nodal_loads", "name": "P1"}},
                      "confidence": .9, "image_geometry": {"point": [.5, .5]}}]}, "a" * 64, str(path))
    return path, image, value, points


@pytest.mark.parametrize("scale", [1, 2])
def test_joint_dots_and_member_strokes_propose_positions_without_numeric_changes(tmp_path, scale):
    """多杆图不能跳过位置候选，候选须有圆点/连线证据，荷载大小、单位、拓扑和人工审核边界保持。"""
    from sketch_axis_refinement import refine_joint_nodes
    path, image, draft, points = _joint_picture(tmp_path, scale)
    before = deepcopy(draft)
    result = refine_joint_nodes(draft, path)
    assert result is not draft and draft == before
    w, h = image.size
    for node, (x, y) in zip(result["image_model"]["nodes"], points, strict=True):
        assert node["u"] * (w - 1) == pytest.approx(x, abs=.5)
        assert node["v"] * (h - 1) == pytest.approx(y, abs=.5)
    for field in ("members", "supports", "load_cases"):
        assert result["image_model"][field] == before["image_model"][field]
    assert result["scale"] == before["scale"] and result["dimensions"] == before["dimensions"]
    load = next(e for e in result["entities"] if e["kind"] == "load")
    node = result["image_model"]["nodes"][1]
    assert load["image_geometry"] == {"point": [node["u"], node["v"]]}
    assert load["original_observation"] == {"image_geometry": {"point": [.5, .5]}, "confidence": .9}
    assert load["position_refinement"]["original_nodes"] == before["image_model"]["nodes"]
    assert load["confidence"] is None and not load["verified"]
    assert result["issues"][-1]["severity"] == "blocking" and result["issues"][-1]["status"] == "open"
    assert refine_joint_nodes(result, path) is result


@pytest.mark.parametrize("protection", ["scale", "verified", "entity", "dimension", "confirmed_dimension", "intersection", "node", "member"])
def test_joint_candidate_does_not_move_user_or_confirmed_geometry(tmp_path, protection):
    """像素位置校正不能覆盖已确认尺度、人工对象/尺寸/交点或已审核实体。"""
    from sketch_axis_refinement import refine_joint_nodes
    _, _, draft, _ = _joint_picture(tmp_path)
    if protection == "scale":
        draft["scale"]["status"] = "confirmed"
    elif protection == "verified":
        draft["entities"][0]["verified"] = True
    elif protection == "entity":
        draft["entities"][0]["source"] = "user"
    elif protection in ("dimension", "confirmed_dimension"):
        draft["dimensions"] = [{"source": "user"}] if protection == "dimension" else [{"status": "confirmed"}]
    elif protection == "intersection":
        draft["intersections"] = [{"source": "user"}]
    else:
        draft["image_model"]["nodes" if protection == "node" else "members"][0]["source"] = "user"
    assert refine_joint_nodes(draft, tmp_path / "not-read.png") is draft


@pytest.mark.parametrize("ambiguity", ["missing_dot", "no_link", "duplicate_dot", "shared_dot", "far_node", "colour_dot", "thin_link", "line_chain", "tiny_image"])
def test_joint_ambiguity_keeps_all_original_positions(tmp_path, ambiguity):
    """缺圆点/杆线、重叠候选、彩色符号、细尺寸线及远点不能部分移动图纸节点或猜连接。"""
    from sketch_axis_refinement import refine_joint_nodes
    path, image, draft, points = _joint_picture(tmp_path)
    paint = ImageDraw.Draw(image)
    x, y = points[0]
    if ambiguity in ("missing_dot", "colour_dot"):
        paint.ellipse((x-2, y-2, x+2, y+2), fill="white" if ambiguity == "missing_dot" else "blue")
    elif ambiguity == "duplicate_dot":
        paint.ellipse((x+4, y-13, x+18, y+1), fill=(220, 220, 220))
        paint.ellipse((x+9, y-8, x+13, y-4), fill="black")
    elif ambiguity in ("no_link", "thin_link"):
        paint.rectangle((85, 160, 140, 181), fill="white")
        if ambiguity == "thin_link":
            paint.line((85, 170, 140, 170), fill=(210, 210, 210), width=1)
    elif ambiguity == "shared_dot":
        draft["image_model"]["nodes"][1].update(u=draft["image_model"]["nodes"][0]["u"], v=draft["image_model"]["nodes"][0]["v"])
    elif ambiguity == "far_node":
        draft["image_model"]["nodes"][0]["u"] += .3
    elif ambiguity == "line_chain":
        draft["image_model"]["members"].pop()
    else:
        image = image.resize((10, 10))
    image.save(path)
    original = deepcopy(draft)
    assert refine_joint_nodes(draft, path) is draft and draft == original


@pytest.mark.parametrize("image_id", ["moj_labels", "moj_bridge"])
def test_frozen_truss_replay_matches_positions_and_preserves_raw_values(image_id):
    """真实桁架首测位置偏差须由像素证据修正，运行时不读取参考；旧标注/响应及原工程向量不改。"""
    import json
    from pathlib import Path
    from sketch_axis_refinement import refine_joint_nodes
    from sketch_parser import SketchParser
    root = Path(__file__).resolve().parents[1] / "multimodal_eval/public_cases/numeric_01"
    response = json.loads((root / "responses" / f"{image_id}.json").read_text(encoding="utf-8"))
    draft = _fill_bookkeeping(SketchParser._extract_json(response["raw_responses"][0]), "a"*64, "")
    original = deepcopy(draft)
    path = root / "images" / f"{image_id}.png"
    refined = refine_joint_nodes(draft, path)
    assert refined is not draft and draft == original
    truth = json.loads((root / "ground_truth" / f"{image_id}.json").read_text(encoding="utf-8"))
    from multimodal_contract import match_points
    predicted = [{"id": n["id"], "point": [n["u"], n["v"]]} for n in refined["image_model"]["nodes"]]
    matched = match_points(predicted, truth["nodes"], truth["width_px"], truth["height_px"])
    assert not matched.unmatched_truths and not matched.unmatched_predictions
    assert refined["image_model"]["load_cases"] == original["image_model"]["load_cases"]
    for e in refined["entities"]:
        if e["kind"] in ("node", "member", "support", "load"):
            assert e["confidence"] is None and not e["verified"]



@pytest.mark.parametrize("image_id", ["moj_labels", "moj_bridge"])
def test_parser_joint_refinement_uses_one_frozen_response_without_extra_request(monkeypatch, image_id):
    """节点位置候选接入真实解析路径，不增加付费请求，结果与直接像素校正保持一致。"""
    import json
    from pathlib import Path
    from multimodal_workflow import MultimodalControllerState, validate_v2_draft
    from sketch_axis_refinement import refine_joint_nodes
    from sketch_parser import SketchParser
    from image_preprocess import work_plane_payload
    root = Path(__file__).resolve().parents[1] / "multimodal_eval/public_cases/numeric_01"
    raw = json.loads((root / "responses" / f"{image_id}.json").read_text(encoding="utf-8"))["raw_responses"][0]
    parser = SketchParser(api_key="offline")
    calls = []
    def frozen_response(*args, **kwargs):
        calls.append(1)
        return raw
    monkeypatch.setattr(parser, "_call_llm", frozen_response)
    state = MultimodalControllerState()
    state.load_image("a" * 64)
    path = root / "images" / f"{image_id}.png"
    result = parser.parse_v2_with_retry(str(path), state, state.start_recognition("offline"),
                                       max_repairs=0, work_plane=work_plane_payload("XY", confirmed=True))
    direct = refine_joint_nodes(_fill_bookkeeping(SketchParser._extract_json(raw), "a"*64, str(path)), path)
    assert result.success and calls == [1] and result.attempts == 1
    assert validate_v2_draft(result.draft) == []
    assert result.draft["image_model"] == direct["image_model"]
    assert any(i["id"] == "pixel-joint-review" and i["status"] == "open" for i in result.draft["issues"])


def test_joint_review_appends_without_resolving_existing_issue(tmp_path):
    """已有同名或全局问题不能被像素候选覆盖或解除，重复回放不能叠加候选历史。"""
    from sketch_axis_refinement import refine_joint_nodes
    path, _, draft, _ = _joint_picture(tmp_path)
    original_issue = {"id": "pixel-joint-review", "category": "load_incomplete", "severity": "blocking", "status": "open"}
    draft["issues"].append(original_issue)
    result = refine_joint_nodes(draft, path)
    assert original_issue in result["issues"]
    assert result["issues"][-1]["id"] == "pixel-joint-review-new"
    assert refine_joint_nodes(result, path) is result
