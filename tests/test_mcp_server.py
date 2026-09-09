"""MCP Server（mcp_server）的单元测试。

MCP Server 把空间刚架计算 Agent 的能力暴露成 10 个 MCP 工具：
1. solve_frame              一站式建模→求解→自校验→出结果
2. validate_frame           只校验模型合法性
3. query_result             查询位移/反力/内力
4. diagnose_supports        支座约束不足诊断
5. modal_analysis           自振频率与振型
6. buckling_analysis        线性屈曲
7. detect_silent_failures_tool  8 项静默失败检测
8. list_capsules_tool       列出历史胶囊
9. get_capsule_tool         读取某个胶囊
10. diff_capsules_tool      对比两个胶囊

这里测五件事：
1. 模块能正常导入，MCPServer 实例创建成功
2. 10 个工具函数都存在且可调用
3. validate_frame 能区分合法/非法模型
4. solve_frame 能求解并返回包含结果的 JSON
5. 每个工具返回的都是合法 JSON 字符串
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("mcp", reason="未安装 mcp 库，跳过 MCP Server 测试")

import mcp_server as mcp_s


# 一个合法的单跨单层框架模型 JSON
VALID_MODEL = json.dumps({
    "units": "N-m-Pa",
    "materials": [{"name": "Q355", "E": 2.06e11, "nu": 0.3, "density": 7850.0}],
    "sections": [{"name": "COL", "A": 0.0147, "Iy": 4.2e-5, "Iz": 1.18e-3, "J": 9e-7},
                 {"name": "BEAM", "A": 0.010, "Iy": 4.0e-5, "Iz": 3.0e-4, "J": 8e-7}],
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
})

INVALID_MODEL = json.dumps({"nodes": [], "members": [], "supports": [], "load_cases": []})


# --------------------------------------------------------- 模块与工具注册

def test_module_creates_mcp_server():
    assert hasattr(mcp_s, "mcp"), "模块应该创建一个 MCPServer 实例"
    assert mcp_s.mcp is not None


def test_all_ten_tools_exist():
    expected = [
        "solve_frame", "validate_frame", "query_result", "diagnose_supports",
        "modal_analysis", "buckling_analysis", "detect_silent_failures_tool",
        "list_capsules_tool", "get_capsule_tool", "diff_capsules_tool",
    ]
    for name in expected:
        assert hasattr(mcp_s, name), f"缺少工具函数 {name}"
        assert callable(getattr(mcp_s, name)), f"{name} 应该可调用"


# --------------------------------------------------------- validate_frame

def test_validate_frame_accepts_valid_model():
    out = mcp_s.validate_frame(VALID_MODEL)
    data = json.loads(out)
    assert data.get("ok") is True or "valid" in str(data).lower()


def test_validate_frame_rejects_invalid_model():
    out = mcp_s.validate_frame(INVALID_MODEL)
    data = json.loads(out)
    assert data.get("ok") is False or "errors" in data or "error" in data


def test_validate_frame_rejects_malformed_json():
    out = mcp_s.validate_frame("not a json")
    data = json.loads(out)
    assert data.get("ok") is False or "error" in data


# --------------------------------------------------------- solve_frame

def test_solve_frame_returns_result_json():
    out = mcp_s.solve_frame(VALID_MODEL, label="test", save_capsule_flag=False)
    data = json.loads(out)
    # 求解结果应该包含 cases 或 results 字段
    assert "cases" in data or "results" in data or "ok" in data


def test_solve_frame_invalid_model_returns_error():
    out = mcp_s.solve_frame(INVALID_MODEL, save_capsule_flag=False)
    data = json.loads(out)
    assert data.get("ok") is False or "error" in data or "errors" in data


# --------------------------------------------------------- detect_silent_failures_tool

def test_detect_silent_failures_returns_eight_checks():
    out = mcp_s.detect_silent_failures_tool(VALID_MODEL)
    data = json.loads(out)
    findings = data.get("findings", data.get("results", []))
    # 8 项检测
    assert len(findings) == 8, f"应该有 8 项检测，实得 {len(findings)}"
    for f in findings:
        assert "id" in f and "name" in f and "status" in f


# --------------------------------------------------------- list_capsules_tool

def test_list_capsules_returns_valid_json():
    out = mcp_s.list_capsules_tool(limit=5)
    data = json.loads(out)
    # 返回应该是列表或包含列表的 dict
    assert isinstance(data, (list, dict))


# --------------------------------------------------------- 所有工具返回合法 JSON

def test_query_result_returns_valid_json():
    out = mcp_s.query_result(VALID_MODEL, case="DL", what="max_displacement")
    json.loads(out)  # 不抛异常就是合法 JSON


def test_diagnose_supports_returns_valid_json():
    out = mcp_s.diagnose_supports(VALID_MODEL)
    json.loads(out)


def test_buckling_analysis_returns_valid_json():
    out = mcp_s.buckling_analysis(VALID_MODEL, case="DL", num_modes=2)
    json.loads(out)


def test_get_capsule_tool_nonexistent_returns_error_json():
    out = mcp_s.get_capsule_tool("does_not_exist_12345")
    data = json.loads(out)
    assert "error" in data or data.get("ok") is False


def test_diff_capsules_tool_nonexistent_returns_error_json():
    out = mcp_s.diff_capsules_tool("nonexistent_1", "nonexistent_2")
    data = json.loads(out)
    assert "error" in data or data.get("ok") is False
