"""集中作用点复核只提出未审核几何，不能补造荷载或迁移已绑定工程引用。"""

from copy import deepcopy

import pytest

from sketch_action_review import apply_action_review
from sketch_parser import _fill_bookkeeping


def draft():
    value = _fill_bookkeeping({
        "image_model": {"nodes": [{"id": 1, "u": .1, "v": .5},
                                   {"id": 2, "u": .9, "v": .5}],
                        "members": [{"id": 1, "i": 1, "j": 2}],
                        "supports": [], "load_cases": []},
        "entities": [],
    }, "a" * 64, "drawing.png")
    value["source"].update(width_px=1000, height_px=200)
    return value


def review(*points):
    return {"actions": [{"id": f"A{i}", "kind": "force" if i == 0 else "moment",
                         "member": 1, "point": list(point), "text": "可见集中作用符号"}
                        for i, point in enumerate(points)], "warnings": []}


def test_two_actions_split_original_member_without_automatic_verification():
    """同一原杆件上的集中力和力矩要连续分段，第二次分段不能把第一段重新标成人工已确认。"""
    original = draft()
    before = deepcopy(original)
    result = apply_action_review(original, review((.4, .5), (.7, .5)))
    assert original == before
    assert len(result["image_model"]["nodes"]) == 4
    assert len(result["image_model"]["members"]) == 3
    assert [r["node"] for r in result["action_review"]["records"]] == [3, 4]
    assert all(not e["verified"] and e["source"] == "derived" for e in result["entities"])
    assert result["image_model"]["load_cases"] == []
    assert result["model"] is None and result["confirmation"] is None
    assert all(i["status"] == "open" and i["severity"] == "blocking" for i in result["issues"]
               if i["id"].startswith("action-review-"))


def test_reversed_member_and_nearby_symbols_reuse_the_same_node():
    """杆件方向反转及同一点的力/矩不能导致分段错误或两个重合节点。"""
    value = draft()
    value["image_model"]["members"][0].update(i=2, j=1)
    result = apply_action_review(value, review((.4, .5), (.401, .5), (.1, .5)))
    assert [r["node"] for r in result["action_review"]["records"]] == [3, 3, 1]
    assert len(result["image_model"]["members"]) == 2


@pytest.mark.parametrize("protection", ["verified", "scale", "dimension", "span", "release"])
def test_existing_review_or_engineering_references_prevent_automatic_splitting(protection):
    """自动作用点不能改已审核几何或静默重挂尺寸、局部荷载、端释放。"""
    value = draft()
    if protection == "verified":
        value["entities"] = [{"kind": "node", "target": {"node": 1}, "verified": True}]
    elif protection == "scale":
        value["scale"]["status"] = "confirmed"
    elif protection == "dimension":
        value["dimensions"] = [{"target": {"member": 1}}]
    elif protection == "span":
        value["image_model"]["load_cases"] = [{"name": "LC", "member_spans": [{"member": 1}]}]
    else:
        value["image_model"]["members"][0]["releases"] = [True] * 6
    result = apply_action_review(value, review((.4, .5)))
    assert result["image_model"] == value["image_model"]
    assert result["action_review"]["records"][0]["reason"]
    assert result["action_review"]["records"][0]["node"] is None


@pytest.mark.parametrize("point,member", [((.5, .9), 1), ((.99, .5), 1), ((.5, .5), 99)])
def test_outside_or_unknown_member_observation_stays_unbound(point, member):
    """远离梁轴、超出杆端或错误引用的观察要保留疑问，不能强行吸附到梁。"""
    value, observation = draft(), review(point)
    observation["actions"][0]["member"] = member
    result = apply_action_review(value, observation)
    assert result["image_model"] == value["image_model"]
    assert result["action_review"]["records"][0]["node"] is None


def test_projection_uses_actual_pixel_aspect_ratio():
    """宽幅图片上的斜梁投影要使用像素宽高，不能把归一化坐标当作正方形距离。"""
    value = draft()
    value["image_model"]["nodes"][0]["v"] = .1
    value["image_model"]["nodes"][1]["v"] = .9
    result = apply_action_review(value, review((.5, .52)))
    node = result["image_model"]["nodes"][-1]
    assert .500 < node["u"] < .502
    assert node["u"] == pytest.approx(node["v"])


@pytest.mark.parametrize("invalid", ["force_value", "nan", "duplicate", "boolean", "extra_top"])
def test_invalid_review_rejects_the_whole_phase_without_partial_mutation(invalid):
    """错误的第二条观察或数值荷载字段不能让第一条分段先写入草稿。"""
    value, observation = draft(), review((.4, .5), (.7, .5))
    before = deepcopy(value)
    if invalid == "force_value":
        observation["actions"][1]["force"] = [0, 0, -100]
    elif invalid == "nan":
        observation["actions"][1]["point"][0] = float("nan")
    elif invalid == "duplicate":
        observation["actions"][1]["id"] = "A0"
    elif invalid == "boolean":
        observation["actions"][1]["member"] = True
    else:
        observation["load_cases"] = []
    with pytest.raises(ValueError):
        apply_action_review(value, observation)
    assert value == before


def test_old_observations_and_nodal_values_are_preserved_with_new_warnings():
    """复核不能覆盖既有支座、节点荷载数值及前一阶段未解决的符号疑问。"""
    value = draft()
    value["image_model"]["supports"] = [{"node": 1, "fix": [1, 1, 1, 0, 0, 0]}]
    value["image_model"]["load_cases"] = [{"name": "LC", "nodal_loads": [{"node": 1, "force": [1, 2, 3]}]}]
    old = {"id": "old", "category": "load_incomplete", "status": "open", "entity_refs": []}
    value["issues"].append(old)
    observation = review((.4, .5))
    observation["warnings"] = ["右端符号可能为反力，请核对。"]
    result = apply_action_review(value, observation)
    assert result["image_model"]["supports"] == value["image_model"]["supports"]
    assert result["image_model"]["load_cases"] == value["image_model"]["load_cases"]
    assert old in result["issues"]
    assert result["action_review"]["warnings"] == observation["warnings"]
