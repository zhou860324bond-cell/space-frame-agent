"""由确定性模型状态推导 Agent 当前所处的工作流阶段。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

from model_io import validate_payload

WORKFLOW_MESSAGE_NAME = "workflow_state"


class WorkflowPhase(str, Enum):
    EMPTY = "empty"
    DRAFT = "draft"
    READY = "ready"
    SOLVED = "solved"


@dataclass(frozen=True)
class WorkflowStatus:
    phase: WorkflowPhase
    counts: dict[str, int]
    validation_errors: tuple[str, ...]
    recommended_tools: tuple[str, ...]
    pending_change: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "agent-workflow/v1",
            "phase": self.phase.value,
            "counts": dict(self.counts),
            "validation_errors": list(self.validation_errors),
            "recommended_tools": list(self.recommended_tools),
            "pending_change": self.pending_change,
        }

    def system_text(self) -> str:
        payload = json.dumps(self.to_dict(), ensure_ascii=False)
        return (
            "以下是代码根据当前 Session 推导的权威工作流状态，不是聊天推测。\n"
            f"{payload}\n"
            "按 recommended_tools 优先推进；draft 状态先修正 validation_errors，"
            "ready 状态才求解，solved 状态才解释或导出结果。用户要求修改模型时可以"
            "从 solved 返回建模阶段，但修改后必须重新校验和求解。"
        )


def inspect_workflow(session) -> WorkflowStatus:
    model = session.model or {}
    counts = {
        "nodes": len(model.get("nodes") or []),
        "members": len(model.get("members") or []),
        "materials": len(model.get("materials") or []),
        "sections": len(model.get("sections") or []),
        "supports": len(model.get("supports") or []),
        "load_cases": len(model.get("load_cases") or []),
    }
    pending = None
    if session.pending_change is not None:
        pending = {
            "preview_id": session.pending_change["preview_id"],
            "tool": session.pending_change["tool"],
            "authorized": (session.authorized_preview_id
                           == session.pending_change["preview_id"]),
        }
    if not counts["nodes"] and not counts["members"]:
        return WorkflowStatus(
            WorkflowPhase.EMPTY, counts, (),
            ("generate_frame", "generate_portal_frame", "generate_bent",
             "add_nodes", "define_materials_and_sections"), pending,
        )

    errors = tuple(str(item)[:240] for item in validate_payload(model)[:5])
    if errors:
        return WorkflowStatus(
            WorkflowPhase.DRAFT, counts, errors,
            ("validate_model", "add_nodes", "add_members",
             "define_materials_and_sections", "assign_properties", "set_supports"),
            pending,
        )

    if session.solution is not None and session.result_db is not None:
        return WorkflowStatus(
            WorkflowPhase.SOLVED, counts, (),
            ("query_results", "query_diagram", "query_envelope",
             "plot_results", "write_report"), pending,
        )

    return WorkflowStatus(
        WorkflowPhase.READY, counts, (),
        ("preview_analysis_mesh", "solve_model"), pending,
    )


def workflow_message(session) -> dict[str, str]:
    return {"role": "system", "name": WORKFLOW_MESSAGE_NAME,
            "content": inspect_workflow(session).system_text()}


def refresh_workflow_message(messages: list[dict], session) -> None:
    """原位替换状态消息，避免旧状态与新状态同时留在上下文中。"""
    messages[:] = [message for message in messages
                   if not (message.get("role") == "system"
                           and message.get("name") == WORKFLOW_MESSAGE_NAME)]
    insert_at = 1 if messages and messages[0].get("role") == "system" else 0
    messages.insert(insert_at, workflow_message(session))
