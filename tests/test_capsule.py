"""实验胶囊（capsule）的单元测试。

胶囊是"求解运行的完整快照"：输入模型 + 各工况结果摘要 + 环境信息 + 输入哈希。
它的价值在于**可复现**和**可对比**——两次运行的输入差异和结果变化都能从胶囊里读出来。

这里测六件事：
1. save 之后文件存在且内容是合法 JSON
2. load 能还原全部字段，且与 save 时传入的一致
3. list 能列出目录里的全部胶囊
4. find 能按 ID 精确找到，找不到返回 None
5. diff 能指出两个胶囊的输入差异和结果变化
6. 相同输入产生相同哈希（这是"可复现"的基础）
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent import Session
import capsule as cap_mod


MATERIALS = [{"name": "Q355", "E": 2.06e11, "nu": 0.3, "density": 7850.0}]
SECTIONS = [{"name": "COL", "A": 0.0147, "Iy": 4.2e-5, "Iz": 1.18e-3, "J": 9e-7},
            {"name": "BEAM", "A": 0.010, "Iy": 4.0e-5, "Iz": 3.0e-4, "J": 8e-7}]


def _solved_session(load: float = -10e3) -> Session:
    """造一个已求解的单跨单层框架。"""
    s = Session()
    s.define_materials_and_sections(MATERIALS, SECTIONS)
    g = s.generate_frame(spans=[6.0], storeys=[3.6],
                         column_section="COL", beam_section="BEAM", material="Q355")
    s.set_load_cases(cases=[{"name": "DL", "member_loads":
        [{"member": m, "w": [0, 0, load]} for m in g.payload["beam_member_ids"]]}])
    s.solve_model()
    return s


# --------------------------------------------------------- save / load

def test_save_creates_a_json_file(tmp_path):
    s = _solved_session()
    path = cap_mod.save_capsule(s.model, s.frame, s.solution,
                                 label="test", directory=tmp_path)
    assert path.exists()
    assert path.suffix == ".json"
    data = json.loads(path.read_text(encoding="utf-8"))
    # 顶层字段必须齐全
    for key in ("id", "timestamp", "input_hash", "input_model",
                "model_stats", "results", "environment", "metadata"):
        assert key in data, f"胶囊缺少顶层字段 {key}"


def test_load_restores_all_fields(tmp_path):
    s = _solved_session()
    path = cap_mod.save_capsule(s.model, s.frame, s.solution,
                                 label="restore-test", note="验证加载",
                                 source="test", directory=tmp_path)
    cap = cap_mod.load_capsule(path)

    assert cap.metadata["label"] == "restore-test"
    assert cap.metadata["note"] == "验证加载"
    assert cap.metadata["source"] == "test"
    # 输入模型完整还原
    assert cap.input_model == s.model
    # 模型统计正确
    assert cap.model_stats["nodes"] == len(s.frame.nodes)
    assert cap.model_stats["members"] == len(s.frame.members)
    # 结果摘要直接以工况名为 key（没有 "cases" 包装层）
    assert "DL" in cap.results


def test_capsule_filename_contains_timestamp_and_hash(tmp_path):
    s = _solved_session()
    path = cap_mod.save_capsule(s.model, s.frame, s.solution, directory=tmp_path)
    # 文件名格式：YYYYMMDD_HHMMSS_<hash前8位>.json
    stem = path.stem
    parts = stem.split("_")
    assert len(parts) == 3, f"文件名格式不对：{stem}"
    assert len(parts[0]) == 8 and parts[0].isdigit()      # 日期
    assert len(parts[1]) == 6 and parts[1].isdigit()      # 时间
    assert len(parts[2]) == 8                               # 哈希前8位


# --------------------------------------------------------- list / find

def test_list_capsules_returns_all_in_directory(tmp_path):
    s1 = _solved_session(load=-10e3)
    s2 = _solved_session(load=-20e3)  # 不同输入 → 不同哈希 → 不同文件名
    cap_mod.save_capsule(s1.model, s1.frame, s1.solution, directory=tmp_path)
    cap_mod.save_capsule(s2.model, s2.frame, s2.solution, directory=tmp_path)
    listed = cap_mod.list_capsules(tmp_path)
    assert len(listed) == 2
    for item in listed:
        assert "id" in item and "timestamp" in item


def test_list_empty_directory_returns_empty(tmp_path):
    assert cap_mod.list_capsules(tmp_path) == []


def test_find_capsule_by_id(tmp_path):
    s = _solved_session()
    path = cap_mod.save_capsule(s.model, s.frame, s.solution, directory=tmp_path)
    cap_id = path.stem
    found = cap_mod.find_capsule(cap_id, tmp_path)
    assert found is not None
    assert Path(found).stem == cap_id


def test_find_nonexistent_returns_none(tmp_path):
    assert cap_mod.find_capsule("does_not_exist", tmp_path) is None


# --------------------------------------------------------- diff

def test_diff_detects_input_changes(tmp_path):
    s1 = _solved_session(load=-10e3)
    s2 = _solved_session(load=-20e3)
    p1 = cap_mod.save_capsule(s1.model, s1.frame, s1.solution, directory=tmp_path)
    p2 = cap_mod.save_capsule(s2.model, s2.frame, s2.solution, directory=tmp_path)

    diff = cap_mod.diff_capsules(p1, p2)
    assert diff["input_identical"] is False
    # 荷载变了，结果也应该变
    assert diff.get("results_diff") or diff.get("results_changed")


def test_diff_identical_capsules_shows_no_change(tmp_path):
    s = _solved_session()
    p1 = cap_mod.save_capsule(s.model, s.frame, s.solution, directory=tmp_path)
    # 相同输入在同一秒内会产生相同文件名，用不同目录或加延迟
    import time
    time.sleep(1.1)
    p2 = cap_mod.save_capsule(s.model, s.frame, s.solution, directory=tmp_path)

    diff = cap_mod.diff_capsules(p1, p2)
    assert diff["input_identical"] is True


# --------------------------------------------------------- 哈希一致性

def test_same_input_produces_same_hash(tmp_path):
    """相同输入模型必须产生相同输入哈希——这是胶囊可复现的基础。

    如果哈希随时间或随机数变化，就没法用哈希判断"两次运行是不是同一个输入"。
    """
    s = _solved_session()
    h1 = cap_mod._input_hash(s.model)
    h2 = cap_mod._input_hash(s.model)
    assert h1 == h2
    assert len(h1) == 64, "SHA256 哈希应该是 64 个十六进制字符"


def test_different_input_produces_different_hash(tmp_path):
    s1 = _solved_session(load=-10e3)
    s2 = _solved_session(load=-20e3)
    assert cap_mod._input_hash(s1.model) != cap_mod._input_hash(s2.model)
