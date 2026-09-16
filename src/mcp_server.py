"""空间刚架求解器 MCP Server — 让 Codex / Cursor / Claude 直接调用求解器。

这是一个无状态的 MCP Server：每次调用独立，不维护会话状态。
所有工具接收 JSON 字符串，返回 JSON 字符串。

启动方式：
    # stdio（默认，MCP 客户端直接 spawn）
    python -m mcp_server

    # SSE（HTTP 流，适合远程访问）
    python -m mcp_server --transport sse --host 0.0.0.0 --port 8765

MCP 客户端配置示例（Claude Desktop / Cursor）：
    {
      "mcpServers": {
        "space-frame": {
          "command": "python",
          "args": ["-m", "mcp_server"],
          "cwd": "/path/to/agent开发"
        }
      }
    }

工具清单：
    solve_frame          验证+求解+静默失败检测+保存胶囊（一站式）
    validate_frame       只验证模型格式
    query_result         查询特定工况的结果（位移/反力/内力）
    diagnose_supports    支座约束诊断
    modal_analysis       模态分析（自振频率+振型）
    buckling_analysis    线性屈曲分析（临界荷载因子）
    detect_silent_failures  静默失败检测（8 类）
    list_capsules        列出历史运行胶囊
    get_capsule          获取胶囊详情
    diff_capsules        对比两个胶囊
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# 确保 src 在路径中
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from frame3d import Frame, Solution, check_equilibrium, solve
from model_io import from_dict, validate_payload
from silent_failures import detect_silent_failures, format_findings, summarize_findings
from capsule import (
    diff_capsules,
    find_capsule,
    list_capsules,
    load_capsule,
    save_capsule,
)

from mcp.server.mcpserver import MCPServer

mcp = MCPServer(
    "space-frame-solver",
    instructions="空间刚架智能计算求解器。输入 JSON 模型，返回求解结果、"
    "静默失败检测和可复现胶囊。所有数值由确定性求解器计算，大模型只负责产出结构。",
)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _json_response(data: Any) -> str:
    """把数据转成 JSON 字符串（处理 numpy 类型）。"""
    def _default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")
    return json.dumps(data, ensure_ascii=False, indent=2, default=_default)


def _parse_model(model_json: str) -> tuple[dict, Frame, list[str]]:
    """解析并验证模型，返回 (payload, model, errors)。"""
    payload = json.loads(model_json)
    errors = validate_payload(payload)
    if errors:
        return payload, None, errors
    model = from_dict(payload)
    return payload, model, []


def _results_summary(model: Frame, sol: Solution) -> dict[str, Any]:
    """提取所有工况的结果摘要。"""
    out = {}
    for name, result in sol.all_results().items():
        # 最大位移
        best_node, best_disp = None, -1.0
        for nid in model.order():
            dofs = model.node_dofs(nid)
            mag = float(np.linalg.norm(result.U[dofs[:3]]))
            if mag > best_disp:
                best_disp, best_node = mag, nid
        # 最大反力
        best_reaction = 0.0
        for nid in model.supports:
            dofs = model.node_dofs(nid)
            best_reaction = max(best_reaction, float(np.linalg.norm(result.R[dofs[:3]])))
        # 控制内力
        max_axial, max_moment, ctrl_member = 0.0, 0.0, None
        for mid, f in result.member_forces.items():
            moment = float(max(abs(f[5]), abs(f[11])))
            if moment > max_moment:
                max_moment = moment
                max_axial = float(max(abs(f[0]), abs(f[6])))
                ctrl_member = mid
        # 平衡校核
        eq = check_equilibrium(model, sol, name)
        out[name] = {
            "max_displacement_mm": best_disp * 1000.0,
            "max_displacement_node": best_node,
            "max_reaction_kN": best_reaction / 1000.0,
            "controlling_member": ctrl_member,
            "controlling_axial_kN": max_axial / 1000.0,
            "controlling_moment_kNm": max_moment / 1000.0,
            "equilibrium_ok": bool(eq.get("ok", False)),
            "equilibrium_relative_residual": float(eq.get("relative", 0.0)),
        }
    return out


# ---------------------------------------------------------------------------
# 工具 1: solve_frame（一站式）
# ---------------------------------------------------------------------------

@mcp.tool()
def solve_frame(model_json: str, label: str = "", save_capsule_flag: bool = True) -> str:
    """验证模型、求解、静默失败检测，可选保存胶囊。一站式求解入口。

    Args:
        model_json: 空间刚架模型 JSON 字符串（包含 nodes/members/sections/materials/supports/load_cases）
        label: 可选标签，用于胶囊存档命名
        save_capsule_flag: 是否保存可复现胶囊（默认 True）

    Returns:
        JSON 字符串，包含模型统计、各工况结果摘要、静默失败检测结果、胶囊路径（如保存）
    """
    try:
        payload, model, errors = _parse_model(model_json)
        if errors:
            return _json_response({"ok": False, "error": "validation_failed", "errors": errors})

        sol = solve(model)
        results = _results_summary(model, sol)

        # 静默失败检测
        findings = detect_silent_failures(model, sol)
        silent_summary = summarize_findings(findings)

        response = {
            "ok": True,
            "model_stats": {
                "nodes": len(model.nodes),
                "members": len(model.members),
                "sections": len(model.sections),
                "load_cases": len(model.load_cases),
                "combos": len(model.combos),
                "units": model.units,
            },
            "results": results,
            "silent_failure_check": {
                "overall": silent_summary["overall"],
                "passed": silent_summary["passed"],
                "critical_issues": silent_summary["critical_issues"],
                "warnings": silent_summary["warnings"],
                "details": findings,
            },
        }

        # 保存胶囊
        if save_capsule_flag:
            cap_path = save_capsule(
                payload, model, sol,
                label=label or None,
                source="mcp",
            )
            response["capsule"] = {
                "saved": True,
                "path": str(cap_path),
                "id": cap_path.stem,
            }

        return _json_response(response)
    except Exception as e:
        return _json_response({"ok": False, "error": type(e).__name__, "message": str(e)})


# ---------------------------------------------------------------------------
# 工具 2: validate_frame
# ---------------------------------------------------------------------------

@mcp.tool()
def validate_frame(model_json: str) -> str:
    """只验证模型格式，不求解。

    Args:
        model_json: 空间刚架模型 JSON 字符串

    Returns:
        JSON 字符串，包含验证结果（通过/错误列表）
    """
    try:
        payload = json.loads(model_json)
        errors = validate_payload(payload)
        if errors:
            return _json_response({"ok": False, "valid": False, "errors": errors})
        model = from_dict(payload)
        return _json_response({
            "ok": True,
            "valid": True,
            "model_stats": {
                "nodes": len(model.nodes),
                "members": len(model.members),
                "sections": len(model.sections),
                "load_cases": len(model.load_cases),
                "combos": len(model.combos),
            },
        })
    except Exception as e:
        return _json_response({"ok": False, "error": type(e).__name__, "message": str(e)})


# ---------------------------------------------------------------------------
# 工具 3: query_result
# ---------------------------------------------------------------------------

@mcp.tool()
def query_result(model_json: str, case: str = "", what: str = "all") -> str:
    """查询特定工况的详细结果。

    Args:
        model_json: 空间刚架模型 JSON 字符串
        case: 工况名称（空则返回所有工况）
        what: 查询内容："all"（全部摘要）、"displacement"（各节点位移）、
              "reaction"（支座反力）、"member_forces"（各杆内力）、"max_displacement"

    Returns:
        JSON 字符串，包含查询结果
    """
    try:
        payload, model, errors = _parse_model(model_json)
        if errors:
            return _json_response({"ok": False, "error": "validation_failed", "errors": errors})

        sol = solve(model)
        cases = [case] if case else list(sol.all_results().keys())
        out = {}

        for cname in cases:
            if cname not in sol.all_results():
                out[cname] = {"error": f"工况 {cname} 不存在"}
                continue
            result = sol[cname]

            if what in ("all", "max_displacement"):
                best_node, best_disp = None, -1.0
                for nid in model.order():
                    dofs = model.node_dofs(nid)
                    mag = float(np.linalg.norm(result.U[dofs[:3]]))
                    if mag > best_disp:
                        best_disp, best_node = mag, nid
                out.setdefault(cname, {})["max_displacement"] = {
                    "value_mm": best_disp * 1000.0, "node": best_node,
                }

            if what in ("all", "displacement"):
                disp = {}
                for nid in model.order():
                    dofs = model.node_dofs(nid)
                    u = result.U[dofs]
                    disp[nid] = {
                        "ux_mm": float(u[0] * 1000), "uy_mm": float(u[1] * 1000),
                        "uz_mm": float(u[2] * 1000), "rx_rad": float(u[3]),
                        "ry_rad": float(u[4]), "rz_rad": float(u[5]),
                    }
                out.setdefault(cname, {})["displacements"] = disp

            if what in ("all", "reaction"):
                reactions = {}
                for nid in model.supports:
                    dofs = model.node_dofs(nid)
                    r = result.R[dofs]
                    reactions[nid] = {
                        "Fx_kN": float(r[0] / 1000), "Fy_kN": float(r[1] / 1000),
                        "Fz_kN": float(r[2] / 1000), "Mx_kNm": float(r[3] / 1000),
                        "My_kNm": float(r[4] / 1000), "Mz_kNm": float(r[5] / 1000),
                    }
                out.setdefault(cname, {})["reactions"] = reactions

            if what in ("all", "member_forces"):
                forces = {}
                for mid, f in result.member_forces.items():
                    forces[mid] = {
                        "Ni_kN": float(f[0] / 1000), "Vyi_kN": float(f[1] / 1000),
                        "Vzi_kN": float(f[2] / 1000), "Mxi_kNm": float(f[3] / 1000),
                        "Myi_kNm": float(f[4] / 1000), "Mzi_kNm": float(f[5] / 1000),
                        "Nj_kN": float(f[6] / 1000), "Vyj_kN": float(f[7] / 1000),
                        "Vzj_kN": float(f[8] / 1000), "Mxj_kNm": float(f[9] / 1000),
                        "Myj_kNm": float(f[10] / 1000), "Mzj_kNm": float(f[11] / 1000),
                    }
                out.setdefault(cname, {})["member_forces"] = forces

        return _json_response({"ok": True, "case": case or "all", "what": what, "data": out})
    except Exception as e:
        return _json_response({"ok": False, "error": type(e).__name__, "message": str(e)})


# ---------------------------------------------------------------------------
# 工具 4: diagnose_supports
# ---------------------------------------------------------------------------

@mcp.tool()
def diagnose_supports(model_json: str) -> str:
    """支座约束诊断 — 检查约束是否足够、是否有机构运动风险。

    Args:
        model_json: 空间刚架模型 JSON 字符串

    Returns:
        JSON 字符串，包含支座约束诊断结果
    """
    try:
        payload, model, errors = _parse_model(model_json)
        if errors:
            return _json_response({"ok": False, "error": "validation_failed", "errors": errors})

        # 统计每个支座的约束情况
        supports_detail = {}
        trans_constrained = [False, False, False]
        rot_constrained = [False, False, False]
        total_dofs = 0

        for nid, sup in model.supports.items():
            constrained = [bool(x) for x in sup]
            total_dofs += sum(constrained)
            for i in range(3):
                if constrained[i]:
                    trans_constrained[i] = True
            for i in range(3, 6):
                if constrained[i]:
                    rot_constrained[i - 3] = True
            supports_detail[nid] = {
                "constrained_dofs": constrained,
                "constrained_count": sum(constrained),
                "type": (
                    "固定端" if sum(constrained) == 6
                    else "铰支座" if sum(constrained[:3]) == 3 and sum(constrained[3:]) == 0
                    else "滚动支座" if sum(constrained) <= 2
                    else "混合约束"
                ),
            }

        missing_trans = [["x", "y", "z"][i] for i in range(3) if not trans_constrained[i]]
        missing_rot = [["rx", "ry", "rz"][i] for i in range(3) if not rot_constrained[i]]

        issues = []
        if total_dofs < 6:
            issues.append(f"约束自由度总数 {total_dofs} < 6，空间结构可能有刚体运动")
        if missing_trans:
            issues.append(f"平动方向 {', '.join(missing_trans)} 未约束")
        if missing_rot and not missing_trans:
            issues.append(f"转动方向 {', '.join(missing_rot)} 未约束（平面结构可能正常）")

        status = "ok" if not issues else ("warning" if len(issues) == 1 and missing_rot else "fail")

        return _json_response({
            "ok": True,
            "status": status,
            "support_count": len(model.supports),
            "total_constrained_dofs": total_dofs,
            "supports": supports_detail,
            "translation_constrained": trans_constrained,
            "rotation_constrained": rot_constrained,
            "missing_translation": missing_trans,
            "missing_rotation": missing_rot,
            "issues": issues,
        })
    except Exception as e:
        return _json_response({"ok": False, "error": type(e).__name__, "message": str(e)})


# ---------------------------------------------------------------------------
# 工具 5: modal_analysis
# ---------------------------------------------------------------------------

@mcp.tool()
def modal_analysis(model_json: str, num_modes: int = 6) -> str:
    """模态分析 — 计算自振频率和振型。

    Args:
        model_json: 空间刚架模型 JSON 字符串
        num_modes: 计算模态数（默认 6）

    Returns:
        JSON 字符串，包含自振频率（Hz）、周期、有效质量
    """
    try:
        from modal import modal as modal_solve

        payload, model, errors = _parse_model(model_json)
        if errors:
            return _json_response({"ok": False, "error": "validation_failed", "errors": errors})

        result = modal_solve(model, num_modes=num_modes)
        freqs = result.frequencies.tolist()
        periods = [1.0 / f if f > 0 else None for f in freqs]
        eff_mass = result.effective_mass.tolist() if result.effective_mass is not None else []

        return _json_response({
            "ok": True,
            "num_modes": len(freqs),
            "total_mass_kg": float(result.total_mass),
            "modes": [
                {
                    "mode": i + 1,
                    "frequency_hz": float(f),
                    "period_s": float(p) if p else None,
                    "omega_rad_s": float(result.omega[i]),
                    "effective_mass_x": float(eff_mass[i][0]) if eff_mass else None,
                    "effective_mass_y": float(eff_mass[i][1]) if eff_mass else None,
                    "effective_mass_z": float(eff_mass[i][2]) if eff_mass else None,
                }
                for i, (f, p) in enumerate(zip(freqs, periods))
            ],
        })
    except Exception as e:
        return _json_response({"ok": False, "error": type(e).__name__, "message": str(e)})


# ---------------------------------------------------------------------------
# 工具 6: buckling_analysis
# ---------------------------------------------------------------------------

@mcp.tool()
def buckling_analysis(model_json: str, case: str = "", num_modes: int = 4) -> str:
    """线性屈曲分析 — 计算临界荷载因子和失稳模态。

    Args:
        model_json: 空间刚架模型 JSON 字符串
        case: 参考荷载工况（空则用主工况）
        num_modes: 计算屈曲模态数（默认 4）

    Returns:
        JSON 字符串，包含临界荷载因子 λ、控制压杆
    """
    try:
        from buckling import buckling as buckling_solve

        payload, model, errors = _parse_model(model_json)
        if errors:
            return _json_response({"ok": False, "error": "validation_failed", "errors": errors})

        result = buckling_solve(model, case=case or None, num_modes=num_modes)
        factors = result.factors.tolist()

        # 找控制压杆（轴力最负的）
        controlling_member = None
        max_compression = 0.0
        if result.axial:
            for mid, n in result.axial.items():
                if n < max_compression:
                    max_compression = n
                    controlling_member = mid

        return _json_response({
            "ok": True,
            "reference_case": result.case,
            "num_modes": len(factors),
            "critical_load_factors": [
                {"mode": i + 1, "lambda": float(f)}
                for i, f in enumerate(factors)
            ],
            "first_critical_lambda": float(factors[0]) if factors else None,
            "controlling_compression_member": controlling_member,
            "controlling_compression_force_kN": float(max_compression / 1000.0) if controlling_member else None,
        })
    except Exception as e:
        return _json_response({"ok": False, "error": type(e).__name__, "message": str(e)})


# ---------------------------------------------------------------------------
# 工具 7: detect_silent_failures
# ---------------------------------------------------------------------------

@mcp.tool()
def detect_silent_failures_tool(model_json: str) -> str:
    """静默失败检测 — 求解后扫描 8 类"跑完了但结果可能是错的"的情况。

    Args:
        model_json: 空间刚架模型 JSON 字符串

    Returns:
        JSON 字符串，包含 8 项检测结果和总体状态
    """
    try:
        payload, model, errors = _parse_model(model_json)
        if errors:
            return _json_response({"ok": False, "error": "validation_failed", "errors": errors})

        sol = solve(model)
        findings = detect_silent_failures(model, sol)
        summary = summarize_findings(findings)

        return _json_response({
            "ok": True,
            "overall": summary["overall"],
            "summary": {
                "total_checks": summary["total_checks"],
                "passed": summary["passed"],
                "critical_issues": summary["critical_issues"],
                "warnings": summary["warnings"],
            },
            "findings": findings,
            "text_report": format_findings(findings, verbose=True),
        })
    except Exception as e:
        return _json_response({"ok": False, "error": type(e).__name__, "message": str(e)})


# ---------------------------------------------------------------------------
# 工具 8-10: 胶囊管理
# ---------------------------------------------------------------------------

@mcp.tool()
def list_capsules_tool(limit: int = 20) -> str:
    """列出历史运行胶囊（按时间倒序）。

    Args:
        limit: 返回数量上限（默认 20）

    Returns:
        JSON 字符串，包含胶囊摘要列表
    """
    try:
        capsules = list_capsules()[:limit]
        return _json_response({"ok": True, "count": len(capsules), "capsules": capsules})
    except Exception as e:
        return _json_response({"ok": False, "error": type(e).__name__, "message": str(e)})


@mcp.tool()
def get_capsule_tool(capsule_id: str) -> str:
    """获取胶囊详情（完整输入模型和结果）。

    Args:
        capsule_id: 胶囊 ID（或前缀）

    Returns:
        JSON 字符串，包含胶囊完整内容
    """
    try:
        path = find_capsule(capsule_id)
        if not path:
            return _json_response({"ok": False, "error": "not_found", "message": f"未找到胶囊: {capsule_id}"})
        cap = load_capsule(path)
        return _json_response({"ok": True, "capsule": cap.to_dict()})
    except Exception as e:
        return _json_response({"ok": False, "error": type(e).__name__, "message": str(e)})


@mcp.tool()
def diff_capsules_tool(capsule_id1: str, capsule_id2: str) -> str:
    """对比两个胶囊的输入和结果差异。

    Args:
        capsule_id1: 第一个胶囊 ID
        capsule_id2: 第二个胶囊 ID

    Returns:
        JSON 字符串，包含结构化差异报告
    """
    try:
        p1 = find_capsule(capsule_id1)
        p2 = find_capsule(capsule_id2)
        if not p1 or not p2:
            missing = []
            if not p1:
                missing.append(capsule_id1)
            if not p2:
                missing.append(capsule_id2)
            return _json_response({"ok": False, "error": "not_found", "message": f"未找到胶囊: {', '.join(missing)}"})
        report = diff_capsules(p1, p2)
        return _json_response({"ok": True, "diff": report})
    except Exception as e:
        return _json_response({"ok": False, "error": type(e).__name__, "message": str(e)})


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="空间刚架求解器 MCP Server")
    parser.add_argument(
        "--transport", choices=["stdio", "sse", "streamable-http"],
        default="stdio", help="传输方式（默认 stdio）",
    )
    parser.add_argument("--host", default="127.0.0.1", help="SSE/HTTP 监听地址")
    parser.add_argument("--port", type=int, default=8765, help="SSE/HTTP 监听端口")
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run()
    elif args.transport == "sse":
        mcp.run_sse_async(host=args.host, port=args.port)
    elif args.transport == "streamable-http":
        mcp.run_streamable_http_async(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
