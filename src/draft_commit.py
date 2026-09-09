"""RecognitionDraft v2 的确定性预演、原子提交与审计侧车。"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from change_preview import model_digest
from multimodal_contract import MODEL_COINCIDENCE_M, canonical_digest
from multimodal_workflow import (MultimodalControllerState, draft_digest,
                                 validate_v2_draft)

SIDECAR_FORMAT = "space-frame-multimodal-provenance/v1"


class DraftCommitError(ValueError):
    pass


def build_merge_plan(draft: Mapping[str, Any], baseline: Mapping[str, Any], *,
                     node_reuse: Mapping[int, int] | None = None) -> dict[str, Any]:
    """生成完整、稳定的 replace-empty/add-only ID 映射。"""
    errors = validate_v2_draft(dict(draft))
    if errors:
        raise DraftCommitError("；".join(errors))
    model = draft.get("model")
    if not isinstance(model, dict):
        raise DraftCommitError("草稿尚未物化为模型")
    if any(item.get("severity") == "blocking" and item.get("status") == "open"
           for item in draft.get("issues", ())):
        raise DraftCommitError("仍有未解决的阻断问题")

    empty = not (baseline.get("nodes") or baseline.get("members"))
    mode = "replace_empty" if empty else "add_only"
    existing_nodes = list(baseline.get("nodes") or [])
    next_node = max((int(item["id"]) for item in existing_nodes), default=0) + 1
    node_actions = []
    create_count = 0
    reuse = {int(key): int(value) for key, value in (node_reuse or {}).items()}
    for node in sorted(model.get("nodes") or [], key=lambda item: int(item["id"])):
        draft_id = int(node["id"])
        point = (float(node["x"]), float(node["y"]), float(node["z"]))
        if draft_id in reuse:
            target = next((item for item in existing_nodes
                           if int(item["id"]) == reuse[draft_id]), None)
            if target is None or math.dist(
                    point, (float(target["x"]), float(target["y"]),
                            float(target["z"]))) > MODEL_COINCIDENCE_M:
                raise DraftCommitError(f"草稿节点 {draft_id} 的复用目标无效或不重合")
            node_actions.append({"draft_id": draft_id, "action": "reuse",
                                 "target_id": reuse[draft_id]})
            continue
        if not empty:
            if any(math.dist(point, (float(old["x"]), float(old["y"]),
                                     float(old["z"]))) <= MODEL_COINCIDENCE_M
                   for old in existing_nodes):
                raise DraftCommitError(
                    f"草稿节点 {draft_id} 与现有节点重合，需先明确复用或平移")
        target_id = draft_id if empty else next_node + create_count
        create_count += 1
        node_actions.append({"draft_id": draft_id, "action": "create",
                             "target_id": target_id})

    next_member = max((int(item["id"]) for item in baseline.get("members") or []),
                      default=0) + 1
    member_id_map = {
        str(int(member["id"])): (int(member["id"]) if empty else next_member + index)
        for index, member in enumerate(sorted(model.get("members") or [],
                                              key=lambda item: int(item["id"])))
    }
    support_name_map = {}
    for index, support in enumerate(model.get("supports") or [], start=1):
        name = str(support.get("name") or f"S{support['node']}-{index}")
        support_name_map[f"{int(support['node'])}:{name}"] = name
    load_name_map = {}
    containers = [("__top__", model)] + [
        (str(case.get("name", "default")), case)
        for case in model.get("load_cases") or []]
    for case_name, container in containers:
        for collection in ("nodal_loads", "member_loads", "member_spans", "settlements"):
            for index, load in enumerate(container.get(collection) or [], start=1):
                name = str(load.get("name") or f"{collection}-{index}")
                load_name_map[f"{case_name}:{collection}:{name}"] = name
    return {
        "mode": mode,
        "baseline_model_hash": model_digest(dict(baseline)),
        "translation_ab_m": [0.0, 0.0],
        "node_actions": node_actions,
        "topology_actions": [],
        "member_id_map": member_id_map,
        "support_name_map": support_name_map,
        "load_name_map": load_name_map,
    }


def materialize_candidate(draft: Mapping[str, Any],
                          baseline: Mapping[str, Any]) -> dict[str, Any]:
    """严格按 merge_plan 生成候选模型，不改变任一输入。"""
    plan = draft.get("merge_plan")
    model = draft.get("model")
    if not isinstance(plan, dict) or not isinstance(model, dict):
        raise DraftCommitError("缺少 model 或 merge_plan")
    if plan.get("baseline_model_hash") != model_digest(dict(baseline)):
        raise DraftCommitError("提交基线已变化，请重新预演")
    mode = plan.get("mode")
    if mode == "replace_empty":
        if baseline.get("nodes") or baseline.get("members"):
            raise DraftCommitError("replace_empty 只能用于空模型")
        return deepcopy(model)
    if mode != "add_only" or not baseline.get("nodes"):
        raise DraftCommitError("非空模型必须使用 add_only")

    candidate = deepcopy(dict(baseline))
    node_actions = {int(item["draft_id"]): item
                    for item in plan.get("node_actions") or []}
    node_map = {draft_id: int(item["target_id"])
                for draft_id, item in node_actions.items()}
    member_map = {int(key): int(value)
                  for key, value in (plan.get("member_id_map") or {}).items()}
    draft_node_ids = {int(item["id"]) for item in model.get("nodes") or []}
    draft_member_ids = {int(item["id"]) for item in model.get("members") or []}
    if set(node_map) != draft_node_ids or set(member_map) != draft_member_ids:
        raise DraftCommitError("merge_plan 未覆盖全部节点或杆件")

    existing_node_ids = {int(item["id"]) for item in candidate.get("nodes") or []}
    for item in model.get("nodes") or []:
        action = node_actions[int(item["id"])]
        target_id = node_map[int(item["id"])]
        if action.get("action") == "reuse":
            if target_id not in existing_node_ids:
                raise DraftCommitError(f"复用节点 {target_id} 不存在")
            continue
        if action.get("action") != "create" or target_id in existing_node_ids:
            raise DraftCommitError(f"节点 {target_id} 的创建/复用计划无效")
        mapped = deepcopy(item)
        mapped["id"] = target_id
        candidate.setdefault("nodes", []).append(mapped)
        existing_node_ids.add(target_id)
    existing_edges = {tuple(sorted((int(item["i"]), int(item["j"]))))
                      for item in candidate.get("members") or []}
    for item in model.get("members") or []:
        mapped = deepcopy(item)
        mapped["id"] = member_map[int(item["id"])]
        mapped["i"], mapped["j"] = node_map[int(item["i"])], node_map[int(item["j"])]
        edge = tuple(sorted((mapped["i"], mapped["j"])))
        if mapped["i"] == mapped["j"] or edge in existing_edges:
            raise DraftCommitError(f"合并后杆件 {item['id']} 自连接或重复")
        candidate.setdefault("members", []).append(mapped)
        existing_edges.add(edge)
    for item in model.get("supports") or []:
        mapped = deepcopy(item)
        mapped["node"] = node_map[int(item["node"])]
        source_name = str(item.get("name") or "")
        mapped_name = (plan.get("support_name_map") or {}).get(
            f"{int(item['node'])}:{source_name}", source_name)
        if mapped_name:
            mapped["name"] = mapped_name
        candidate.setdefault("supports", []).append(mapped)

    for collection in ("materials", "sections", "combos", "sets"):
        key = "name"
        existing = {str(item[key]): item for item in candidate.get(collection) or []}
        for item in model.get(collection) or []:
            name = str(item[key])
            if name in existing and existing[name] != item:
                raise DraftCommitError(f"{collection} 名称冲突: {name}")
            if name not in existing:
                candidate.setdefault(collection, []).append(deepcopy(item))
                existing[name] = item

    def mapped_load(load: Mapping[str, Any], case_name: str,
                    collection: str, index: int) -> dict[str, Any]:
        result = deepcopy(dict(load))
        source_name = str(load.get("name") or f"{collection}-{index}")
        result["name"] = (plan.get("load_name_map") or {}).get(
            f"{case_name}:{collection}:{source_name}", source_name)
        if "node" in result:
            result["node"] = node_map[int(result["node"])]
        if "member" in result:
            result["member"] = member_map[int(result["member"])]
        return result

    load_collections = ("nodal_loads", "member_loads", "member_spans", "settlements")
    for collection in load_collections:
        existing_names = {str(item.get("name", ""))
                          for item in candidate.get(collection) or []}
        for index, item in enumerate(model.get(collection) or [], start=1):
            mapped = mapped_load(item, "__top__", collection, index)
            if mapped["name"] in existing_names:
                raise DraftCommitError(f"顶层 {collection} 名称冲突: {mapped['name']}")
            candidate.setdefault(collection, []).append(mapped)
            existing_names.add(mapped["name"])

    cases = {str(item["name"]): item for item in candidate.get("load_cases") or []}
    for source_case in model.get("load_cases") or []:
        case_name = str(source_case["name"])
        target_case = cases.get(case_name)
        if target_case is None:
            target_case = {key: deepcopy(value) for key, value in source_case.items()
                           if key not in load_collections}
            target_case["name"] = case_name
            candidate.setdefault("load_cases", []).append(target_case)
            cases[case_name] = target_case
        else:
            source_meta = {key: value for key, value in source_case.items()
                           if key not in load_collections}
            target_meta = {key: value for key, value in target_case.items()
                           if key not in load_collections}
            if source_meta != target_meta:
                raise DraftCommitError(f"荷载工况名称冲突: {case_name}")
        for collection in load_collections:
            names = {str(item.get("name", ""))
                     for item in target_case.get(collection) or []}
            for index, item in enumerate(source_case.get(collection) or [], start=1):
                mapped = mapped_load(item, case_name, collection, index)
                if mapped["name"] in names:
                    raise DraftCommitError(
                        f"工况 {case_name} 的 {collection} 名称冲突: {mapped['name']}")
                target_case.setdefault(collection, []).append(mapped)
                names.add(mapped["name"])
    return candidate


def _operations(before: Mapping[str, Any], after: Mapping[str, Any],
                plan: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    specs = (("node", "nodes", "id"), ("member", "members", "id"),
             ("material", "materials", "name"),
             ("section", "sections", "name"), ("load_case", "load_cases", "name"))
    operations = []
    for kind, collection, identity in specs:
        old = {str(item[identity]): item for item in before.get(collection) or []}
        new = {str(item[identity]): item for item in after.get(collection) or []}
        for key in sorted(new.keys() - old.keys()):
            operations.append({"op": "add", "kind": kind, "key": key,
                               "before": None, "after": deepcopy(new[key])})
    old_supports = {f"{item['node']}:{item.get('name', '')}": item
                    for item in before.get("supports") or []}
    new_supports = {f"{item['node']}:{item.get('name', '')}": item
                    for item in after.get("supports") or []}
    for key in sorted(new_supports.keys() - old_supports.keys()):
        operations.append({"op": "add", "kind": "support", "key": key,
                           "before": None, "after": deepcopy(new_supports[key])})

    collections = ("nodal_loads", "member_loads", "member_spans", "settlements")

    def loads(model: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        indexed = {}
        containers = [("__top__", model)] + [
            (str(case.get("name", "default")), case)
            for case in model.get("load_cases") or []]
        for case_name, container in containers:
            for collection in collections:
                for index, item in enumerate(container.get(collection) or [], start=1):
                    name = str(item.get("name") or f"{collection}-{index}")
                    indexed[f"{case_name}:{collection}:{name}"] = item
        return indexed

    old_loads, new_loads = loads(before), loads(after)
    for key in sorted(new_loads.keys() - old_loads.keys()):
        operations.append({"op": "add", "kind": "load", "key": key,
                           "before": None, "after": deepcopy(new_loads[key])})
    if plan:
        old_nodes = {int(item["id"]): item for item in before.get("nodes") or []}
        for action in plan.get("node_actions") or []:
            if action.get("action") == "reuse":
                target = int(action["target_id"])
                operations.append({
                    "op": "reuse", "kind": "node", "key": str(action["draft_id"]),
                    "before": deepcopy(old_nodes.get(target)),
                    "after": {"target_id": target},
                })
        da, db = plan.get("translation_ab_m") or [0.0, 0.0]
        if da or db:
            operations.append({"op": "translate", "kind": "geometry", "key": "all",
                               "before": [0.0, 0.0], "after": [da, db]})
        for action in plan.get("topology_actions") or []:
            operations.append({
                "op": action["decision"], "kind": action["kind"],
                "key": f"{action['candidate_id']}:{action['existing_id']}",
                "before": None, "after": deepcopy(action),
            })
    return sorted(operations, key=lambda item: (item["kind"], item["key"], item["op"]))


def prepare_commit(draft: Mapping[str, Any], baseline: Mapping[str, Any], *,
                   node_reuse: Mapping[int, int] | None = None) \
        -> tuple[dict[str, Any], dict[str, Any]]:
    """返回带 merge_plan 的新草稿及不可变预演；Session 保持不变。"""
    prepared = deepcopy(dict(draft))
    prepared["merge_plan"] = build_merge_plan(
        prepared, baseline, node_reuse=node_reuse)
    prepared["confirmation"] = None
    candidate = materialize_candidate(prepared, baseline)
    operations = _operations(baseline, candidate, prepared["merge_plan"])
    preview = {
        "preview_id": "mm_" + canonical_digest({
            "draft": draft_digest(prepared), "operations": operations})[:16],
        "draft_hash": draft_digest(prepared),
        "merge_plan_hash": canonical_digest(prepared["merge_plan"]),
        "baseline_model_hash": model_digest(dict(baseline)),
        "operations": operations,
        "diff_hash": canonical_digest(operations),
    }
    return prepared, preview


def commit_prepared(session: Any, state: MultimodalControllerState, *,
                    confirmed_at: str) -> Any:
    """核对全部哈希后，提交模型和 provenance，并生成 controller receipt。"""
    preview = state.commit_preview
    draft = state.draft
    if not isinstance(preview, dict) or not isinstance(draft, dict):
        raise DraftCommitError("没有可提交的预演")
    if not state.start_commit(session.model, confirmed_at=confirmed_at):
        raise DraftCommitError("草稿未达到可提交状态或预演已陈旧")
    try:
        candidate = materialize_candidate(draft, session.model)
        operations = _operations(session.model, candidate, draft["merge_plan"])
        checks = (
            preview.get("draft_hash") == draft_digest(draft),
            preview.get("merge_plan_hash") == canonical_digest(draft["merge_plan"]),
            preview.get("baseline_model_hash") == model_digest(session.model),
            preview.get("operations") == operations,
            preview.get("diff_hash") == canonical_digest(operations),
        )
        if not all(checks):
            raise DraftCommitError("预演哈希已陈旧，请重新预演")
        provenance = {
            "format": SIDECAR_FORMAT,
            "workflow_instance_id": state.workflow_instance_id,
            "image_hash": state.image_hash,
            "draft_hash": draft_digest(draft),
            "revision": draft["revision"],
            "source": deepcopy(draft["source"]),
            "work_plane": deepcopy(draft["work_plane"]),
            "scale": deepcopy(draft["scale"]),
            "entities": deepcopy(draft["entities"]),
            "dimensions": deepcopy(draft["dimensions"]),
            "intersections": deepcopy(draft["intersections"]),
            "issues": deepcopy(draft["issues"]),
            "merge_plan": deepcopy(draft["merge_plan"]),
            "commit_preview": deepcopy(preview),
        }
        result = session.apply_multimodal_draft(
            candidate, provenance,
            expected_baseline_hash=preview["baseline_model_hash"])
        if not result.ok:
            raise DraftCommitError(result.payload.get("error") or "提交失败")
        history_id = str(session.history[session.history.cursor].index)
        state.finish_commit(success=True, model_hash=model_digest(session.model),
                            committed_at=confirmed_at, history_step_id=history_id)
        return result
    except Exception:
        state.finish_commit(success=False)
        raise


def sidecar_path(model_path: str | Path) -> Path:
    return Path(model_path).with_suffix(".multimodal.json")


def save_sidecar(model_path: str | Path, model: Mapping[str, Any],
                 provenance: Mapping[str, Any] | None) -> Path | None:
    if provenance is None:
        return None
    target = sidecar_path(model_path)
    payload = {"format": SIDECAR_FORMAT, "model_hash": model_digest(dict(model)),
               "provenance": deepcopy(dict(provenance))}
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def load_sidecar(model_path: str | Path, model: Mapping[str, Any]) \
        -> tuple[dict[str, Any] | None, str | None]:
    target = sidecar_path(model_path)
    if not target.is_file():
        return None, None
    try:
        payload = json.loads(target.read_text(encoding="utf-8-sig"))
        if payload.get("format") != SIDECAR_FORMAT or not isinstance(
                payload.get("provenance"), dict):
            raise ValueError("侧车格式无效")
        if payload.get("model_hash") != model_digest(dict(model)):
            return None, "多模态侧车与模型哈希不匹配，已忽略追溯数据"
        return deepcopy(payload["provenance"]), None
    except Exception as exc:  # noqa: BLE001 - 损坏侧车不能阻止打开主模型
        return None, f"多模态侧车读取失败，已忽略：{exc}"
