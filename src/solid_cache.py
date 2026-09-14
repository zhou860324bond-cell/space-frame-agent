"""局部实体双后端结果的保守磁盘恢复。

缓存只用于跳过已经正式收敛、且输入与网格计划完全相同的阶段。旧版摘要没有
输入指纹，或主结果文件已经被移动/截断时，一律视为不可复用；重新计算比把别的
模型结果混进当前对标更便宜。
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Callable
from uuid import uuid4

from change_preview import model_digest


CACHE_SCHEMA = "solid-joint-cache/v1"
ORPHAN_MIN_AGE_SECONDS = 24 * 60 * 60
_RUN_ID_RE = re.compile(r"\d{8}T\d{6}_\d{6}Z_[0-9a-f]{8}")
_TARGET_RE = re.compile(r"node_(\d+)_(.*)")
_BACKENDS = {
    "native": ("results/native_solid_joint", "finest_vtu"),
    "abaqus": ("results/solid_joint", "finest_odb"),
}


class SolidHistoryCleanupError(ValueError):
    """历史清理请求不满足可恢复、安全且可核验的目录约束。"""


def start_solid_run(output_dir: Path | str, node_id: int,
                    case: str) -> dict[str, Any]:
    """为一次实体求解创建独立目录；不改写当前 latest 摘要。"""
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%S_%fZ_") + uuid4().hex[:8]
    case_dir = Path(output_dir).resolve() / f"node_{int(node_id)}_{case}"
    run_dir = case_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return {
        "run_id": run_id,
        "generated_at": now.isoformat(timespec="seconds"),
        "case_dir": case_dir,
        "run_dir": run_dir,
    }


def write_solid_summary(payload: dict[str, Any], case_dir: Path,
                        run_dir: Path) -> None:
    """先固化本次摘要，再原子更新 latest；中断不会留下半份 JSON。"""
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    (run_dir / "summary.json").write_text(text, encoding="utf-8")
    pending = case_dir / f".summary.{payload.get('run_id', 'building')}.tmp"
    pending.write_text(text, encoding="utf-8")
    pending.replace(case_dir / "summary.json")


def _spec_dict(spec: Any) -> dict[str, Any]:
    if hasattr(spec, "to_dict"):
        value = spec.to_dict()
    elif is_dataclass(spec):
        value = asdict(spec)
    elif hasattr(spec, "__dict__"):
        value = {key: item for key, item in vars(spec).items()
                 if not key.startswith("_")}
    elif isinstance(spec, dict):
        value = spec
    else:
        raise TypeError("节点实体输入必须可转换为字典")
    if not isinstance(value, dict):
        raise TypeError("节点实体输入摘要必须是字典")
    return value


def make_solid_cache_identity(model: dict[str, Any], spec: Any,
                              mesh_plan_mm: list[float]) -> dict[str, Any]:
    """构造可写入 summary.json 的稳定缓存身份。"""
    return {
        "schema": CACHE_SCHEMA,
        "model_digest": model_digest(model),
        "joint_input_digest": model_digest(_spec_dict(spec)),
        "mesh_plan_mm": [float(value) for value in mesh_plan_mm],
    }


def _same_plan(actual: Any, expected: list[float]) -> bool:
    if not isinstance(actual, list) or len(actual) != len(expected):
        return False
    try:
        values = [float(value) for value in actual]
    except (TypeError, ValueError):
        return False
    return all(math.isfinite(left) and math.isfinite(right)
               and math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-10)
               for left, right in zip(values, expected))


def solid_cache_matches(payload: dict[str, Any], backend: str,
                        model: dict[str, Any], spec: Any,
                        mesh_plan_mm: list[float]) -> bool:
    """判断内存中的摘要是否属于这一次任务，不检查磁盘附件。"""
    if backend not in _BACKENDS or not isinstance(payload, dict):
        return False
    identity = payload.get("cache_identity")
    expected = make_solid_cache_identity(model, spec, mesh_plan_mm)
    actual_backend = str(payload.get("backend") or "").lower()
    backend_matches = ((backend == "native" and "native" in actual_backend)
                       or (backend == "abaqus" and "abaqus" in actual_backend))
    spec_data = _spec_dict(spec)
    try:
        same_target = (int(payload.get("node_id")) == int(spec_data["node_id"])
                       and str(payload.get("case") or "") ==
                       str(spec_data.get("case") or ""))
    except (KeyError, TypeError, ValueError):
        same_target = False
    meshes = payload.get("meshes")
    mesh_key = "global_mesh_size_mm" if backend == "native" else "mesh_size_mm"
    actual_mesh_plan = ([row.get(mesh_key) for row in meshes]
                        if isinstance(meshes, list)
                        and all(isinstance(row, dict) for row in meshes) else None)
    return bool(
        backend_matches
        and str(payload.get("schema") or "").startswith("solid-joint-analysis/")
        and same_target
        and payload.get("hot_spot_all_converged") is True
        and _same_plan(actual_mesh_plan, expected["mesh_plan_mm"])
        and isinstance(identity, dict)
        and identity.get("schema") == CACHE_SCHEMA
        and identity.get("model_digest") == expected["model_digest"]
        and identity.get("joint_input_digest") == expected["joint_input_digest"]
        and _same_plan(identity.get("mesh_plan_mm"), expected["mesh_plan_mm"])
    )


def _artifact_exists(payload: dict[str, Any], summary_path: Path,
                     artifact_key: str) -> bool:
    files = payload.get("files")
    raw = files.get(artifact_key) if isinstance(files, dict) else None
    if not isinstance(raw, str) or not raw.strip():
        return False
    path = Path(raw)
    candidates = [path] if path.is_absolute() else [Path.cwd() / path,
                                                    summary_path.parent / path]
    for candidate in candidates:
        try:
            if candidate.is_file() and candidate.stat().st_size > 0:
                return True
        except OSError:
            continue
    return False


def load_solid_cache(model: dict[str, Any], spec: Any,
                     plans: dict[str, list[float]],
                     root: Path | str = ".") -> dict[str, Any]:
    """从固定结果目录恢复两个后端；无效项只报告原因，不中断新计算。"""
    base = Path(root)
    usable: dict[str, dict[str, Any]] = {}
    ignored: dict[str, str] = {}
    node_id = int(getattr(spec, "node_id", _spec_dict(spec).get("node_id")))
    case = str(getattr(spec, "case", _spec_dict(spec).get("case") or ""))

    for backend, plan in plans.items():
        if backend not in _BACKENDS:
            ignored[backend] = "未知实体后端"
            continue
        directory, artifact_key = _BACKENDS[backend]
        summary_path = base / directory / f"node_{node_id}_{case}" / "summary.json"
        if not summary_path.is_file():
            continue
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            ignored[backend] = "summary.json 损坏或不可读"
            continue
        if not solid_cache_matches(payload, backend, model, spec, plan):
            ignored[backend] = "输入指纹、后端、收敛状态或网格计划不匹配"
            continue
        if not _artifact_exists(payload, summary_path, artifact_key):
            ignored[backend] = f"主结果文件 {artifact_key} 缺失或为空"
            continue
        usable[backend] = payload
    return {"usable": usable, "ignored": ignored}


def load_solid_history(node_id: int, case: str,
                       root: Path | str = ".") -> list[dict[str, Any]]:
    """读取同一节点/工况的历次摘要；损坏项跳过，最新项不重复列出。"""
    base = Path(root)
    history: list[dict[str, Any]] = []
    seen: set[str] = set()
    for backend, (directory, _artifact_key) in _BACKENDS.items():
        case_dir = base / directory / f"node_{int(node_id)}_{case}"
        candidates = sorted((case_dir / "runs").glob("*/summary.json"))
        latest = case_dir / "summary.json"
        if latest.is_file():
            candidates.append(latest)
        for summary_path in candidates:
            try:
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            try:
                same_target = (int(payload.get("node_id")) == int(node_id)
                               and str(payload.get("case") or "") == str(case))
            except (TypeError, ValueError):
                same_target = False
            actual_backend = str(payload.get("backend") or "").lower()
            if not same_target or backend not in actual_backend:
                continue
            run_id = str(payload.get("run_id") or summary_path.resolve())
            identity = f"{backend}:{run_id}"
            if identity in seen:
                continue
            seen.add(identity)
            saved = dict(payload)
            saved["_history_id"] = identity
            saved["_history_summary"] = str(summary_path.resolve())
            saved["_result_origin"] = "history"
            history.append(saved)
    history.sort(key=lambda item: str(item.get("generated_at") or ""), reverse=True)
    return history


def load_all_solid_records(root: Path | str = ".", *,
                           now: float | None = None) -> dict[str, Any]:
    """扫描两个受管理结果根目录，不依赖当前模型或某次成功摘要。

    只把形如 ``node_<id>_<case>`` 的直接子目录当作目标；符号链接和其它目录
    都跳过。这样首次运行就在求解阶段失败、尚未生成 latest 摘要时，仍能从
    全局结果中心找到残留目录。
    """
    base = Path(root)
    targets: set[tuple[int, str]] = set()
    for directory, _artifact_key in _BACKENDS.values():
        backend_root = base / directory
        if not backend_root.is_dir() or backend_root.is_symlink():
            continue
        try:
            case_dirs = backend_root.iterdir()
            for case_dir in case_dirs:
                if not case_dir.is_dir() or case_dir.is_symlink():
                    continue
                match = _TARGET_RE.fullmatch(case_dir.name)
                if match:
                    targets.add((int(match.group(1)), match.group(2)))
        except OSError:
            continue

    history: list[dict[str, Any]] = []
    orphans: list[dict[str, Any]] = []
    seen_history: set[str] = set()
    seen_orphans: set[str] = set()
    for node_id, case in sorted(targets, key=lambda item: (item[0], item[1])):
        for payload in load_solid_history(node_id, case, root=base):
            identifier = str(payload.get("_history_id") or "")
            if identifier and identifier not in seen_history:
                seen_history.add(identifier)
                history.append(payload)
        for payload in load_solid_orphans(node_id, case, root=base, now=now):
            identifier = str(payload.get("_history_id") or "")
            if identifier and identifier not in seen_orphans:
                seen_orphans.add(identifier)
                orphans.append(payload)
    history.sort(key=lambda item: str(item.get("generated_at") or ""), reverse=True)
    orphans.sort(key=lambda item: str(item.get("generated_at") or ""), reverse=True)
    return {
        "history": history, "orphans": orphans,
        "targets": [{"node_id": node_id, "case": case}
                    for node_id, case in sorted(targets)],
    }


def _directory_usage(directory: Path) -> tuple[int, int, float]:
    total = 0
    file_count = 0
    modified = directory.stat().st_mtime
    for directory_name, _subdirs, filenames in os.walk(
            directory, followlinks=False):
        base = Path(directory_name)
        for filename in filenames:
            path = base / filename
            if path.is_symlink():
                continue
            stat = path.stat()
            total += stat.st_size
            file_count += 1
            modified = max(modified, stat.st_mtime)
    return total, file_count, modified


def _incomplete_stage(run_dir: Path, summary_exists: bool) -> str:
    names = {path.name.lower() for path in run_dir.iterdir() if path.is_file()}
    suffixes = {Path(name).suffix for name in names}
    if summary_exists:
        return "摘要损坏或身份不匹配"
    if "run_console.txt" in names:
        return "求解器返回失败"
    if ".odb" in suffixes:
        return "后处理或摘要未完成"
    if ".vtu" in suffixes:
        return "结果汇总未完成"
    if suffixes & {".inp", ".sta", ".msg", ".dat", ".log"}:
        return "建模或求解未完成"
    return "初始化后中断"


def load_solid_orphans(node_id: int, case: str, root: Path | str = ".",
                       *, now: float | None = None) -> list[dict[str, Any]]:
    """列出没有形成有效历史摘要的独立运行目录，不把近期目录当垃圾。"""
    base = Path(root)
    current_time = time.time() if now is None else float(now)
    complete_dirs = {
        str(Path(item["_history_summary"]).resolve().parent)
        for item in load_solid_history(node_id, case, root=base)
        if item.get("_history_summary")
    }
    orphans: list[dict[str, Any]] = []
    for backend, (directory, _artifact_key) in _BACKENDS.items():
        case_dir = base / directory / f"node_{int(node_id)}_{case}"
        runs_dir = case_dir / "runs"
        if not runs_dir.is_dir():
            continue
        latest_run = ""
        try:
            latest = json.loads((case_dir / "summary.json").read_text(
                encoding="utf-8"))
            if isinstance(latest, dict):
                latest_run = str(latest.get("run_id") or "")
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass
        try:
            candidates = sorted(runs_dir.iterdir())
        except OSError:
            continue
        for run_dir in candidates:
            if (not run_dir.is_dir() or run_dir.is_symlink()
                    or not _RUN_ID_RE.fullmatch(run_dir.name)
                    or str(run_dir.resolve()) in complete_dirs):
                continue
            try:
                size, files, modified = _directory_usage(run_dir)
                stage = _incomplete_stage(
                    run_dir, (run_dir / "summary.json").is_file())
            except OSError:
                continue
            age = max(0.0, current_time - modified)
            is_latest = run_dir.name == latest_run
            generated = datetime.fromtimestamp(
                modified, timezone.utc).isoformat(timespec="seconds")
            orphans.append({
                "_history_id": f"orphan:{backend}:{run_dir.name}",
                "_orphan": True,
                "_run_directory": str(run_dir.resolve()),
                "backend": backend,
                "node_id": int(node_id), "case": str(case),
                "run_id": run_dir.name, "generated_at": generated,
                "incomplete_stage": stage, "bytes": size, "files": files,
                "age_seconds": age, "latest": is_latest,
                "cleanup_eligible": (
                    not is_latest and age >= ORPHAN_MIN_AGE_SECONDS),
            })
    orphans.sort(key=lambda item: str(item["generated_at"]), reverse=True)
    return orphans


def _solid_history_run_context(payload: dict[str, Any]) -> tuple[Path, Path]:
    """返回 ``(run_dir, case_dir)``，并拒绝旧版平铺结果和越界路径。"""
    raw_summary = payload.get("_history_summary")
    run_id = str(payload.get("run_id") or "")
    history_id = str(payload.get("_history_id") or "")
    if not isinstance(raw_summary, str) or not raw_summary or not run_id:
        raise SolidHistoryCleanupError("该记录缺少独立运行目录信息，不能清理")
    summary = Path(raw_summary).resolve()
    run_dir = summary.parent
    case_dir = run_dir.parent.parent
    if (summary.name != "summary.json" or run_dir.name != run_id
            or run_dir.parent.name != "runs"):
        raise SolidHistoryCleanupError("仅允许清理规范 runs/<run_id> 目录")
    backend = history_id.partition(":")[0]
    expected_directory = _BACKENDS.get(backend, (None, None))[0]
    expected_name = Path(expected_directory).name if expected_directory else None
    try:
        expected_case = f"node_{int(payload.get('node_id'))}_{payload.get('case') or ''}"
    except (TypeError, ValueError) as exc:
        raise SolidHistoryCleanupError("历史记录的节点编号无效") from exc
    if (not expected_name or case_dir.name != expected_case
            or case_dir.parent.name != expected_name
            or case_dir.parent.parent.name != "results"):
        raise SolidHistoryCleanupError("历史目录不属于受管理的实体结果路径")
    if not run_dir.is_dir() or run_dir.is_symlink():
        raise SolidHistoryCleanupError("运行目录不存在或不是普通目录")
    return run_dir, case_dir


def solid_history_cleanup_state(payload: dict[str, Any]) -> dict[str, Any]:
    """轻量核对清理资格；只读两个摘要，不遍历大型 ODB/VTU 目录。"""
    try:
        run_dir, case_dir = _solid_history_run_context(payload)
    except SolidHistoryCleanupError as exc:
        return {
            "directory": None, "latest": None, "valid": False,
            "reason": str(exc),
        }
    latest_path = case_dir / "summary.json"
    try:
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        latest_run = str(latest.get("run_id") or "") if isinstance(latest, dict) else ""
        if not latest_run:
            raise ValueError("latest 摘要缺少 run_id")
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return {
            "directory": str(run_dir), "latest": None, "valid": True,
            "reason": f"latest 指针不可核验：{exc}",
        }
    is_latest = latest_run == str(payload.get("run_id") or "")
    return {
        "directory": str(run_dir), "latest": is_latest, "valid": True,
        "reason": "该后端最新成功结果" if is_latest else "可清理的历史运行",
    }


def solid_history_storage(payload: dict[str, Any]) -> dict[str, Any]:
    """统计一条历史记录的独立目录，并核对它是否为该后端 latest。"""
    state = solid_history_cleanup_state(payload)
    directory = state.get("directory")
    if directory is None:
        return {**state, "bytes": 0, "files": 0}
    try:
        total, file_count, _modified = _directory_usage(Path(directory))
    except OSError as exc:
        return {
            **state, "bytes": 0, "files": 0, "valid": False,
            "reason": f"目录不可完整读取：{exc}",
        }
    return {**state, "bytes": total, "files": file_count}


def move_solid_history_to_trash(
        payload: dict[str, Any], protected_ids: set[str],
        move_to_trash: Callable[[str], bool]) -> dict[str, Any]:
    """经双层门禁后把一次旧运行移入系统回收站，不直接永久删除。"""
    identifier = str(payload.get("_history_id") or "")
    if not identifier or identifier in protected_ids:
        raise SolidHistoryCleanupError("当前版本 A/B 受保护，不能清理")
    storage = solid_history_storage(payload)
    if not storage["valid"]:
        raise SolidHistoryCleanupError(str(storage["reason"]))
    if storage["latest"] is None:
        raise SolidHistoryCleanupError("latest 指针不可核验，为避免误删已停止清理")
    if storage["latest"]:
        raise SolidHistoryCleanupError("该记录是后端最新成功结果，不能清理")
    directory = str(storage["directory"])
    try:
        moved = bool(move_to_trash(directory))
    except OSError as exc:
        raise SolidHistoryCleanupError(f"移入回收站失败：{exc}") from exc
    if not moved:
        raise SolidHistoryCleanupError("系统未能把运行目录移入回收站")
    return {**storage, "history_id": identifier}


def _solid_orphan_run_context(payload: dict[str, Any]) -> tuple[Path, Path]:
    identifier = str(payload.get("_history_id") or "")
    parts = identifier.split(":", 2)
    backend = parts[1] if len(parts) == 3 and parts[0] == "orphan" else ""
    run_id = str(payload.get("run_id") or "")
    raw_directory = payload.get("_run_directory")
    if (not isinstance(raw_directory, str) or not raw_directory or not run_id
            or len(parts) != 3 or parts[2] != run_id
            or not _RUN_ID_RE.fullmatch(run_id)):
        raise SolidHistoryCleanupError("不完整运行缺少可核验的目录身份")
    run_dir = Path(raw_directory).resolve()
    case_dir = run_dir.parent.parent
    expected_directory = _BACKENDS.get(backend, (None, None))[0]
    expected_name = Path(expected_directory).name if expected_directory else None
    try:
        expected_case = f"node_{int(payload.get('node_id'))}_{payload.get('case') or ''}"
    except (TypeError, ValueError) as exc:
        raise SolidHistoryCleanupError("不完整运行的节点编号无效") from exc
    if (run_dir.name != run_id or run_dir.parent.name != "runs"
            or case_dir.name != expected_case or not expected_name
            or case_dir.parent.name != expected_name
            or case_dir.parent.parent.name != "results"):
        raise SolidHistoryCleanupError("不完整运行不属于受管理的实体结果路径")
    if not run_dir.is_dir() or run_dir.is_symlink():
        raise SolidHistoryCleanupError("不完整运行目录不存在或不是普通目录")
    return run_dir, case_dir


def move_solid_orphan_to_trash(
        payload: dict[str, Any], move_to_trash: Callable[[str], bool],
        *, now: float | None = None) -> dict[str, Any]:
    """重新核验后清理超过保护期的不完整运行目录。"""
    run_dir, case_dir = _solid_orphan_run_context(payload)
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            actual_backend = (str(summary.get("backend") or "").lower()
                              if isinstance(summary, dict) else "")
            backend = str(payload.get("backend") or "").lower()
            if (isinstance(summary, dict)
                    and str(summary.get("run_id") or "") == run_dir.name
                    and int(summary.get("node_id")) == int(payload.get("node_id"))
                    and str(summary.get("case") or "") == str(payload.get("case") or "")
                    and backend in actual_backend):
                raise SolidHistoryCleanupError("该运行已经形成有效摘要，请刷新历史后再操作")
        except SolidHistoryCleanupError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            pass
    latest_path = case_dir / "summary.json"
    latest_run = ""
    if latest_path.exists():
        try:
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
            latest_run = (str(latest.get("run_id") or "")
                          if isinstance(latest, dict) else "")
            if not latest_run:
                raise ValueError("latest 摘要缺少 run_id")
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise SolidHistoryCleanupError(
                "latest 指针不可核验，为避免误删已停止清理") from exc
    if latest_run == run_dir.name:
        raise SolidHistoryCleanupError("该目录被后端 latest 引用，不能清理")
    try:
        size, files, modified = _directory_usage(run_dir)
    except OSError as exc:
        raise SolidHistoryCleanupError(f"不完整运行目录不可完整读取：{exc}") from exc
    current_time = time.time() if now is None else float(now)
    age = max(0.0, current_time - modified)
    if age < ORPHAN_MIN_AGE_SECONDS:
        remaining = math.ceil((ORPHAN_MIN_AGE_SECONDS - age) / 3600.0)
        raise SolidHistoryCleanupError(
            f"该目录最近仍有写入，至少再等待 {remaining} 小时才能清理")
    try:
        moved = bool(move_to_trash(str(run_dir)))
    except OSError as exc:
        raise SolidHistoryCleanupError(f"移入回收站失败：{exc}") from exc
    if not moved:
        raise SolidHistoryCleanupError("系统未能把不完整运行移入回收站")
    return {
        "history_id": str(payload.get("_history_id")),
        "directory": str(run_dir), "bytes": size, "files": files,
        "age_seconds": age,
    }
