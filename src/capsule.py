"""实验胶囊 — 把每次求解变成可复现、可对比、可回归的存档。

一个胶囊就是一个 JSON 文件，记录：
  - 输入模型（完整 payload，可直接重跑）
  - 结果摘要（各工况最大位移 / 反力 / 控制内力 / 平衡校核）
  - 环境信息（Python / OS / numpy 版本，时间戳）
  - 输入哈希（用于去重和 diff）

用法：
    from capsule import save_capsule, load_capsule, list_capsules, diff_capsules

    # 求解后自动存档
    model = from_dict(payload)
    sol = solve(model)
    path = save_capsule(payload, model, sol, label="门式刚架试算")

    # 列出历史
    for c in list_capsules():
        print(c["id"], c["metadata"]["label"], c["timestamp"])

    # 对比两次运行
    report = diff_capsules("capsules/aaa.json", "capsules/bbb.json")

CLI：
    python -m capsule save <model.json> [--label "..."]
    python -m capsule list
    python -m capsule show <capsule_id>
    python -m capsule diff <capsule_id1> <capsule_id2>
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from frame3d import Frame, Solution, check_equilibrium

DEFAULT_CAPSULE_DIR = Path("capsules")


# ---------------------------------------------------------------------------
# 结果摘要提取
# ---------------------------------------------------------------------------

def _model_stats(model: Frame) -> dict[str, int]:
    return {
        "nodes": len(model.nodes),
        "members": len(model.members),
        "sections": len(model.sections),
        "materials": len(model.materials),
        "supports": len(model.supports),
        "load_cases": len(model.load_cases),
        "combos": len(model.combos),
        "units": model.units,
    }


def _case_summary(model: Frame, sol: Solution, name: str) -> dict[str, Any]:
    """提取单个工况/组合的结果摘要。"""
    result = sol[name]

    # 最大合位移（平动 3 自由度）
    best_node, best_disp = None, -1.0
    for nid in model.order():
        dofs = model.node_dofs(nid)
        mag = float(np.linalg.norm(result.U[dofs[:3]]))
        if mag > best_disp:
            best_disp, best_node = mag, nid

    # 最大反力（支座节点）
    best_reaction = 0.0
    for nid in model.supports:
        dofs = model.node_dofs(nid)
        mag = float(np.linalg.norm(result.R[dofs[:3]]))
        best_reaction = max(best_reaction, mag)

    # 控制内力（所有杆件中最大的轴力和弯矩）
    max_axial, max_moment, ctrl_member = 0.0, 0.0, None
    for mid, f in result.member_forces.items():
        axial = float(max(abs(f[0]), abs(f[6])))
        moment = float(max(abs(f[5]), abs(f[11])))
        if moment > max_moment:
            max_moment = moment
            max_axial = axial
            ctrl_member = mid

    # 平衡校核
    eq = check_equilibrium(model, sol, name)

    return {
        "max_displacement": {
            "value_m": best_disp,
            "value_mm": best_disp * 1000.0,
            "node": best_node,
        },
        "max_reaction_force": {
            "value_N": best_reaction,
            "value_kN": best_reaction / 1000.0,
        },
        "controlling_member": {
            "member": ctrl_member,
            "axial_force_kN": max_axial / 1000.0,
            "moment_z_kNm": max_moment / 1000.0,
        },
        "equilibrium": {
            "ok": bool(eq.get("ok", False)),
            "relative_residual": float(eq.get("relative", 0.0)),
        },
    }


def _results_summary(model: Frame, sol: Solution) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in sol.all_results():
        out[name] = _case_summary(model, sol, name)
    return out


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "os": f"{platform.system()} {platform.release()}",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "script": sys.argv[0] if sys.argv else "",
    }


def _input_hash(payload: dict[str, Any]) -> str:
    """对输入模型做稳定哈希（key 排序，确保同模型同哈希）。"""
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 胶囊数据类
# ---------------------------------------------------------------------------

@dataclass
class Capsule:
    id: str
    timestamp: str
    input_hash: str
    input_model: dict[str, Any]
    model_stats: dict[str, int]
    results: dict[str, Any]
    environment: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "input_hash": self.input_hash,
            "input_model": self.input_model,
            "model_stats": self.model_stats,
            "results": self.results,
            "environment": self.environment,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Capsule":
        return cls(
            id=data["id"],
            timestamp=data["timestamp"],
            input_hash=data["input_hash"],
            input_model=data["input_model"],
            model_stats=data.get("model_stats", {}),
            results=data.get("results", {}),
            environment=data.get("environment", {}),
            metadata=data.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# 核心 API
# ---------------------------------------------------------------------------

def save_capsule(
    payload: dict[str, Any],
    model: Frame,
    sol: Solution,
    label: str | None = None,
    note: str | None = None,
    source: str = "api",
    directory: Path | str | None = None,
) -> Path:
    """求解后调用，把本次运行存成一个胶囊 JSON。

    返回保存的文件路径。
    """
    cap_dir = Path(directory) if directory else DEFAULT_CAPSULE_DIR
    cap_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    full_hash = _input_hash(payload)
    short_hash = full_hash[:8]
    cap_id = f"{ts}_{short_hash}"

    metadata: dict[str, Any] = {"source": source}
    if label:
        metadata["label"] = label
    if note:
        metadata["note"] = note

    capsule = Capsule(
        id=cap_id,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        input_hash=full_hash,
        input_model=payload,
        model_stats=_model_stats(model),
        results=_results_summary(model, sol),
        environment=_environment(),
        metadata=metadata,
    )

    path = cap_dir / f"{cap_id}.json"
    path.write_text(
        json.dumps(capsule.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def load_capsule(path: Path | str) -> Capsule:
    """加载一个胶囊文件。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Capsule.from_dict(data)


def list_capsules(directory: Path | str | None = None) -> list[dict[str, Any]]:
    """列出所有胶囊的摘要（不加载完整 input_model，省内存）。

    按时间倒序排列。
    """
    cap_dir = Path(directory) if directory else DEFAULT_CAPSULE_DIR
    if not cap_dir.exists():
        return []

    summaries: list[dict[str, Any]] = []
    for p in sorted(cap_dir.glob("*.json"), reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            summaries.append({
                "id": data.get("id", p.stem),
                "path": str(p),
                "timestamp": data.get("timestamp", ""),
                "input_hash": data.get("input_hash", "")[:12],
                "model_stats": data.get("model_stats", {}),
                "metadata": data.get("metadata", {}),
                "case_count": len(data.get("results", {})),
            })
        except (json.JSONDecodeError, OSError):
            continue
    return summaries


def find_capsule(capsule_id: str, directory: Path | str | None = None) -> Path | None:
    """按 ID 前缀查找胶囊文件。"""
    cap_dir = Path(directory) if directory else DEFAULT_CAPSULE_DIR
    if not cap_dir.exists():
        return None
    matches = list(cap_dir.glob(f"{capsule_id}*.json"))
    return matches[0] if matches else None


def diff_capsules(
    path1: Path | str,
    path2: Path | str,
) -> dict[str, Any]:
    """对比两个胶囊，返回结构化差异报告。

    对比维度：
      - 输入模型是否相同（哈希）
      - 模型统计（节点/杆件/工况数）
      - 各工况结果数值差异（最大位移/反力/控制内力）
      - 环境差异
    """
    c1 = load_capsule(path1)
    c2 = load_capsule(path2)

    report: dict[str, Any] = {
        "capsule_a": c1.id,
        "capsule_b": c2.id,
        "input_identical": c1.input_hash == c2.input_hash,
        "model_stats_diff": {},
        "results_diff": {},
        "environment_diff": {},
    }

    # 模型统计差异
    for key in set(c1.model_stats) | set(c2.model_stats):
        v1, v2 = c1.model_stats.get(key), c2.model_stats.get(key)
        if v1 != v2:
            report["model_stats_diff"][key] = {"a": v1, "b": v2}

    # 环境差异
    for key in set(c1.environment) | set(c2.environment):
        v1, v2 = c1.environment.get(key), c2.environment.get(key)
        if v1 != v2:
            report["environment_diff"][key] = {"a": v1, "b": v2}

    # 结果差异
    all_cases = set(c1.results) | set(c2.results)
    for case in sorted(all_cases):
        r1, r2 = c1.results.get(case), c2.results.get(case)
        if r1 is None or r2 is None:
            report["results_diff"][case] = {
                "status": "missing_in_a" if r1 is None else "missing_in_b",
            }
            continue

        case_diff: dict[str, Any] = {}

        # 最大位移
        d1 = r1["max_displacement"]["value_mm"]
        d2 = r2["max_displacement"]["value_mm"]
        if abs(d1 - d2) > 1e-10:
            rel = abs(d1 - d2) / max(abs(d1), 1e-30) * 100
            case_diff["max_displacement_mm"] = {
                "a": d1, "b": d2, "abs_diff": abs(d1 - d2), "rel_diff_pct": rel,
            }

        # 最大反力
        f1 = r1["max_reaction_force"]["value_kN"]
        f2 = r2["max_reaction_force"]["value_kN"]
        if abs(f1 - f2) > 1e-10:
            rel = abs(f1 - f2) / max(abs(f1), 1e-30) * 100
            case_diff["max_reaction_kN"] = {
                "a": f1, "b": f2, "abs_diff": abs(f1 - f2), "rel_diff_pct": rel,
            }

        # 控制内力
        m1 = r1["controlling_member"]["moment_z_kNm"]
        m2 = r2["controlling_member"]["moment_z_kNm"]
        if abs(m1 - m2) > 1e-10:
            case_diff["controlling_moment_kNm"] = {
                "a": m1, "b": m2, "abs_diff": abs(m1 - m2),
            }

        # 平衡校核
        eq1 = r1["equilibrium"]["ok"]
        eq2 = r2["equilibrium"]["ok"]
        if eq1 != eq2:
            case_diff["equilibrium_ok"] = {"a": eq1, "b": eq2}

        if case_diff:
            report["results_diff"][case] = case_diff

    report["summary"] = {
        "cases_with_numeric_diff": sum(
            1 for v in report["results_diff"].values()
            if isinstance(v, dict) and any(k not in ("status",) for k in v)
        ),
        "model_stats_changed": bool(report["model_stats_diff"]),
        "environment_changed": bool(report["environment_diff"]),
    }
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_save(args: list[str]) -> int:
    if not args:
        print("用法: python -m capsule save <model.json> [--label TEXT] [--note TEXT]")
        return 1

    model_path = Path(args[0])
    label = None
    note = None
    i = 1
    while i < len(args):
        if args[i] == "--label" and i + 1 < len(args):
            label = args[i + 1]; i += 2
        elif args[i] == "--note" and i + 1 < len(args):
            note = args[i + 1]; i += 2
        else:
            i += 1

    from model_io import from_dict, validate_payload
    from frame3d import solve

    payload = json.loads(model_path.read_text(encoding="utf-8"))
    errors = validate_payload(payload)
    if errors:
        print("校验未通过：")
        for e in errors:
            print("  ", e)
        return 1

    model = from_dict(payload)
    sol = solve(model)
    path = save_capsule(payload, model, sol, label=label, note=note, source="cli")
    print(f"胶囊已保存: {path}")
    print(f"  ID: {path.stem}")
    print(f"  节点 {model_stats_str(model)}")
    return 0


def model_stats_str(model: Frame) -> str:
    return (f"{len(model.nodes)} / 杆件 {len(model.members)} / "
            f"工况 {len(model.load_cases)} / 组合 {len(model.combos)}")


def _cli_list(_args: list[str]) -> int:
    capsules = list_capsules()
    if not capsules:
        print("（暂无胶囊，运行 python -m capsule save <model.json> 创建第一个）")
        return 0
    print(f"{'ID':<28} {'时间':<20} {'节点':>4} {'杆件':>4} {'工况':>4} {'标签'}")
    print("-" * 90)
    for c in capsules:
        label = c["metadata"].get("label", "-")
        ms = c["model_stats"]
        print(f"{c['id']:<28} {c['timestamp']:<20} "
              f"{ms.get('nodes', 0):>4} {ms.get('members', 0):>4} "
              f"{c['case_count']:>4} {label}")
    return 0


def _cli_show(args: list[str]) -> int:
    if not args:
        print("用法: python -m capsule show <capsule_id>")
        return 1
    path = find_capsule(args[0])
    if not path:
        print(f"未找到胶囊: {args[0]}")
        return 1
    cap = load_capsule(path)
    print(f"胶囊 ID: {cap.id}")
    print(f"时间: {cap.timestamp}")
    print(f"输入哈希: {cap.input_hash[:16]}...")
    print(f"模型: {cap.model_stats}")
    print(f"环境: {cap.environment.get('os')} / Python {cap.environment.get('python')}")
    print(f"元数据: {cap.metadata}")
    print(f"\n{'工况/组合':<24} {'最大位移(mm)':>14} {'最大反力(kN)':>14} {'平衡':>8}")
    print("-" * 64)
    for name, r in cap.results.items():
        eq = "通过" if r["equilibrium"]["ok"] else "失败"
        print(f"{name:<24} {r['max_displacement']['value_mm']:>14.4f} "
              f"{r['max_reaction_force']['value_kN']:>14.4f} {eq:>8}")
    return 0


def _cli_diff(args: list[str]) -> int:
    if len(args) < 2:
        print("用法: python -m capsule diff <capsule_id1> <capsule_id2>")
        return 1
    p1 = find_capsule(args[0])
    p2 = find_capsule(args[1])
    if not p1 or not p2:
        print(f"未找到胶囊: {args[0] if not p1 else ''} {args[1] if not p2 else ''}")
        return 1
    report = diff_capsules(p1, p2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print("用法: python -m capsule <save|list|show|diff> [args...]")
        return 1
    cmd = sys.argv[1]
    args = sys.argv[2:]
    handlers = {
        "save": _cli_save,
        "list": _cli_list,
        "show": _cli_show,
        "diff": _cli_diff,
    }
    if cmd not in handlers:
        print(f"未知命令: {cmd}，可用: {', '.join(handlers)}")
        return 1
    return handlers[cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
