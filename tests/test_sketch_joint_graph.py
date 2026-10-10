"""像素拓扑候选必须独立、可追溯，并在人工编辑后重新核对。"""

from copy import deepcopy
import json
from pathlib import Path

import pytest
from PIL import ImageDraw

from multimodal_contract import file_digest, match_members, match_points
from sketch_axis_refinement import (dismiss_joint_graph_review, review_joint_graph,
                                    sync_joint_graph_review)
from sketch_parser import SketchParser, _fill_bookkeeping
from test_sketch_axis_refinement import _joint_picture


def _issue(draft):
    return next(i for i in draft["issues"] if i["id"] == draft["joint_graph_review"]["issue_id"])


@pytest.mark.parametrize("missing", ["node", "member", "position"])
def test_missing_geometry_gets_independent_candidates_without_changing_model(tmp_path, missing):
    """漏点、漏杆及偏移不能使整张图跳过核对，也不能靠像素候选重绑荷载和支座。"""
    path, _, draft, _ = _joint_picture(tmp_path)
    if missing == "node":
        draft["image_model"]["nodes"].pop()
        draft["image_model"]["members"] = draft["image_model"]["members"][:1]
    elif missing == "member":
        draft["image_model"]["members"].pop()
    before = deepcopy(draft)
    result = review_joint_graph(draft, path)
    review = result["joint_graph_review"]
    assert draft == before and result["image_model"] == before["image_model"]
    for key in ("entities", "dimensions", "scale", "work_plane"):
        assert result[key] == before[key]
    assert len(review["nodes"]) == len(review["members"]) == 3
    assert review["file_sha256"] == file_digest(path)
    assert review["confidence"] is None and review["source"] == "derived"
    assert review["original_geometry"]["nodes"] == before["image_model"]["nodes"]
    assert _issue(result)["status"] == "open" and _issue(result)["severity"] == "blocking"
    assert any(ref.startswith("pixel-node:") for ref in _issue(result)["entity_refs"])
    assert review_joint_graph(result, tmp_path / "not-read.png") is result


@pytest.mark.parametrize("protection", ["scale", "verified", "entity", "dimension", "confirmed_dimension", "intersection", "node", "member"])
def test_existing_manual_geometry_is_not_reinterpreted(tmp_path, protection):
    """已确认或人工编辑的草稿不能被新一次像素探测重新解释。"""
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
    assert review_joint_graph(draft, tmp_path / "not-read.png") is draft


@pytest.mark.parametrize("unsafe", ["missing_dot", "thin_line", "isolated_dot", "disconnected"])
def test_insufficient_or_multiple_pixel_graphs_do_not_propose_topology(tmp_path, unsafe):
    """缺接头、细尺寸线、孤点和多个图样不能强行选取主体或补出连接。"""
    path, image, draft, points = _joint_picture(tmp_path)
    paint = ImageDraw.Draw(image)
    if unsafe == "missing_dot":
        x, y = points[0]
        paint.ellipse((x-2, y-2, x+2, y+2), fill="white")
    elif unsafe == "thin_line":
        paint.rectangle((85, 160, 140, 181), fill="white")
        paint.line((85, 170, 140, 170), fill=(210, 210, 210), width=1)
    else:
        extra = [(25, 25)] if unsafe == "isolated_dot" else [(20, 20), (60, 20), (40, 45)]
        if len(extra) > 1:
            for a, b in ((0, 1), (1, 2), (0, 2)):
                paint.line((*extra[a], *extra[b]), fill=(210, 210, 210), width=12)
        for x, y in extra:
            paint.ellipse((x-7, y-7, x+7, y+7), fill=(220, 220, 220), outline="black")
            paint.ellipse((x-2, y-2, x+2, y+2), fill="black")
    image.save(path)
    assert review_joint_graph(draft, path) is draft


def test_intermediate_joint_prevents_long_bar_skipping_a_node(tmp_path):
    """原图长线中间已有黑色接头时，候选须分成两段，不能另外生成跨接头杆件。"""
    path, image, draft, _ = _joint_picture(tmp_path)
    paint = ImageDraw.Draw(image)
    paint.ellipse((173, 163, 187, 177), fill=(220, 220, 220), outline="black")
    paint.ellipse((178, 168, 182, 172), fill="black")
    image.save(path)
    review = review_joint_graph(draft, path)["joint_graph_review"]
    assert len(review["nodes"]) == len(review["members"]) == 4
    by_id = {n["id"]: n["point"] for n in review["nodes"]}
    assert all(abs(by_id[m["i"]][0] - by_id[m["j"]][0]) < .7 for m in review["members"])


def test_manual_retention_is_local_and_geometry_changes_reopen_it(tmp_path):
    """保留当前拓扑只能解除本问题；工程值编辑不撤销核对，节点改动必须重新核对。"""
    path, _, draft, _ = _joint_picture(tmp_path)
    draft["issues"].append({"id": "global", "category": "load_incomplete", "severity": "blocking", "status": "open"})
    result = review_joint_graph(draft, path)
    initial = deepcopy(result["joint_graph_review"])
    kept = dismiss_joint_graph_review(result, initial["issue_id"])
    assert _issue(kept)["resolved_by"] == "user"
    assert kept["image_model"] == draft["image_model"]
    assert next(i for i in kept["issues"] if i["id"] == "global")["status"] == "open"
    assert kept["edit_history"][-1]["previous"]["review"] == initial
    kept["image_model"]["load_cases"][0]["nodal_loads"][0]["load"][1] = -7000
    assert sync_joint_graph_review(kept) is kept
    kept["image_model"]["nodes"][0]["u"] += .1
    reopened = sync_joint_graph_review(kept)
    assert _issue(reopened)["status"] == "open" and _issue(reopened)["resolved_by"] is None
    assert reopened["joint_graph_review"]["initial_comparison"] == initial["initial_comparison"]
    assert reopened["joint_graph_review"]["nodes"] == initial["nodes"]
    with pytest.raises(ValueError, match="已改变"):
        dismiss_joint_graph_review(reopened, "global")


def test_corrected_positions_resolve_only_pixel_review_without_reading_image(tmp_path):
    """人工移到像素位置后只解除候选差异；原图不可再读时仍须正确同步冻结证据。"""
    path, image, draft, points = _joint_picture(tmp_path)
    result = review_joint_graph(draft, path)
    path.unlink()
    for node, (x, y) in zip(result["image_model"]["nodes"], points, strict=True):
        node.update(u=x/(image.width-1), v=y/(image.height-1), source="user")
    result["model"] = result["merge_plan"] = result["confirmation"] = {"stale": True}
    synced = sync_joint_graph_review(result)
    assert synced["joint_graph_review"]["status"] == "matched"
    assert _issue(synced)["status"] == "resolved"
    assert synced["model"] is synced["merge_plan"] is synced["confirmation"] is None
    assert synced["joint_graph_review"]["comparison"]["matched_members"] == 3


def test_candidate_image_binding_and_colliding_issue_are_preserved(tmp_path):
    """换图不能沿用旧候选；同名工程阻断不能被候选覆盖。"""
    path, _, draft, _ = _joint_picture(tmp_path)
    old = {"id": "pixel-topology-review", "category": "load_incomplete", "status": "open", "severity": "blocking"}
    draft["issues"].append(old)
    result = review_joint_graph(draft, path)
    assert old in result["issues"]
    assert result["joint_graph_review"]["issue_id"] == "pixel-topology-review-new"
    result["source"]["image_hash"] = "b"*64
    with pytest.raises(ValueError, match="当前图片不一致"):
        sync_joint_graph_review(result)


def test_visual_response_cannot_forge_local_pixel_evidence(tmp_path):
    """视觉输出自报已核对候选不能绕过本地读图或阻断审核。"""
    _, _, draft, _ = _joint_picture(tmp_path)
    draft["joint_graph_review"] = {"status": "dismissed", "source": "user"}
    result = _fill_bookkeeping(draft, "a"*64, "")
    assert "joint_graph_review" not in result


@pytest.mark.parametrize(("image_id", "nodes", "members"), [("tower_truss", 12, 21), ("wall_truss", 5, 7)])
def test_frozen_compact_graph_candidates_are_separate_from_primary_scores(image_id, nodes, members):
    """冻结开发图的漏点候选单独评分，原节点、荷载和错误支座不能改写成已修复。"""
    root = Path(__file__).resolve().parents[1] / "multimodal_eval/public_cases/compact_01"
    response = json.loads((root / "responses" / f"{image_id}.json").read_text(encoding="utf-8"))
    draft = _fill_bookkeeping(SketchParser._extract_json(response["raw_responses"][0]), "a"*64, "")
    result = review_joint_graph(draft, root / "images" / f"{image_id}.png")
    review = result["joint_graph_review"]
    assert len(review["nodes"]) == nodes and len(review["members"]) == members
    assert result["image_model"] == draft["image_model"]
    truth = json.loads((root / "ground_truth" / f"{image_id}.json").read_text(encoding="utf-8"))
    matched = match_points(review["nodes"], truth["nodes"], truth["width_px"], truth["height_px"])
    edges = match_members(review["members"], truth["members"], matched.pairs)
    assert not matched.unmatched_predictions and not matched.unmatched_truths
    assert not edges.unmatched_predictions and not edges.unmatched_truths
    assert _issue(result)["status"] == "open"


def test_optional_action_review_resynchronizes_candidate_comparison(monkeypatch):
    """作用点复核增删几何之后不能向控制器提交过期的像素对应与核对状态。"""
    from image_preprocess import work_plane_payload
    from multimodal_workflow import MultimodalControllerState
    root = Path(__file__).resolve().parents[1] / "multimodal_eval/public_cases/compact_01"
    raw = json.loads((root / "responses/tower_truss.json").read_text(encoding="utf-8"))["raw_responses"][0]
    parser = SketchParser(api_key="offline")
    calls = []
    def response(*args, **kwargs):
        calls.append(1)
        return raw if len(calls) == 1 else "{}"
    def reviewed(draft, *_args, **_kwargs):
        result = deepcopy(draft)
        graph = result["joint_graph_review"]
        result["image_model"]["nodes"] = [{"id": n["id"], "u": n["point"][0], "v": n["point"][1]} for n in graph["nodes"]]
        result["image_model"]["members"] = [{k: m[k] for k in ("id", "i", "j")} for m in graph["members"]]
        return result
    monkeypatch.setattr(parser, "_call_llm", response)
    monkeypatch.setattr("sketch_action_review.apply_action_review", reviewed)
    state = MultimodalControllerState()
    state.load_image("a"*64)
    result = parser.parse_v2_with_retry(str(root / "images/tower_truss.png"), state,
        state.start_recognition("offline"), max_repairs=0, review_actions=True,
        work_plane=work_plane_payload("XY", confirmed=True))
    assert result.success and calls == [1, 1]
    assert result.draft["joint_graph_review"]["status"] == "matched"
    assert _issue(result.draft)["status"] == "resolved"


def test_development_evidence_preserves_raw_bindings_and_separate_denominators():
    """候选成绩不能偷换主模型成绩，公开开发记录必须绑定原响应/参考及完整源码。"""
    from multimodal_contract import canonical_digest
    from multimodal_eval.real_world_eval import evaluate_real_world
    root = Path(__file__).resolve().parents[1] / "multimodal_eval/public_cases"
    report = json.loads((root / "joint_graph_01/report.json").read_text(encoding="utf-8"))
    snapshot = json.loads((root / "joint_graph_01/pipeline_snapshot.json").read_text(encoding="utf-8"))
    assert len(snapshot) == 8 and canonical_digest(snapshot) == report["pipeline_hash"]
    assert report["new_api_calls"] == report["primary_replay"]["new_api_calls"] == 0
    assert not report["primary_replay"]["passed"]
    original = evaluate_real_world(root / "compact_01")
    assert report["original_report_sha256"] == file_digest(root / "compact_01/report.json")
    assert report["primary_replay"]["metrics"]["issue_false_positives"] == original["metrics"]["issue_false_positives"] + 2
    for case in report["cases"]:
        name = case["image_id"]
        response_path = root / "compact_01/responses" / f"{name}.json"
        response = json.loads(response_path.read_text(encoding="utf-8"))
        assert case["response_sha256"] == file_digest(response_path)
        assert case["reference_sha256"] == file_digest(root / "compact_01/ground_truth" / f"{name}.json")
        assert case["input_sha256"] == file_digest(root / "compact_01/images" / f"{name}.png")
        assert case["first_raw_response_sha256"] == canonical_digest(response["raw_responses"][0])
        assert case["reference_read_after_extraction"] and case["primary_image_model_unchanged"]
        for key in ("nodes", "members"):
            assert case["candidate_only_scores"][key]["tp"] == len(case["candidates"][key])
            assert case["candidate_only_scores"][key]["fp"] == case["candidate_only_scores"][key]["fn"] == 0
