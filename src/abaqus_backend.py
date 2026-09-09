r"""用 Abaqus 作为求解后端。

**它是可选后端，不是替换。** 自研求解器一次几十毫秒、零依赖、可离线评测；
Abaqus 起一次作业至少几秒到几十秒，还要授权。所以默认仍走自研，
需要它的时候（对标验证、或者梁单元做不了的分析）再切过来。

对模型来说这只是多了一个求解工具：契约不变，返回格式与 `solve_model` 对齐。

流程：写 .inp → `abaqus job=... interactive` → 读 `.sta`/`.msg` 判断成败
→ `abaqus python extract_odb.py` 导出 CSV → 整理成与自研求解器相同的结构。

Abaqus 6.14 的 Python 是 2.7，所以读 ODB 那一步必须交给 `abaqus python`，
不能用本环境的 Python 3。
"""

from __future__ import annotations

import csv
import math
import re
import shutil
import subprocess
import tempfile

import numpy as np
from pathlib import Path
from typing import Any

from inp_writer import write_inp
from units import of as unit_system_of

HERE = Path(__file__).resolve().parent
EXTRACTOR_CANDIDATES = ("abaqus_bench/extract_odb.py", "extract_odb.py")

# .msg / .dat 里这些字样意味着这次作业没算成，光看返回码不够可靠
_FATAL_PATTERNS = (
    re.compile(r"\*\*\*\s*ERROR", re.IGNORECASE),
    re.compile(r"THE ANALYSIS HAS BEEN TERMINATED", re.IGNORECASE),
    re.compile(r"NUMERICAL SINGULARITY", re.IGNORECASE),
    re.compile(r"ZERO PIVOT", re.IGNORECASE),
)
_WARN_PATTERN = re.compile(r"\*\*\*\s*WARNING[^\n]*", re.IGNORECASE)


class AbaqusError(RuntimeError):
    """Abaqus 侧的失败。消息面向用户与模型，尽量具体。"""


# Abaqus 只把 abaqus.bat 放进「Abaqus Command」那个终端的 PATH。从普通 cmd 或
# 从图形界面启动时 which 找不到它 —— 机器上装了却报「没装」。所以再翻一遍默认装法
# 会用的几个目录。这些是 Windows 上的固定位置，Linux 装法一律靠 PATH。
_WINDOWS_HINTS = (
    r"C:\SIMULIA\Commands", r"C:\SIMULIA\Abaqus\Commands",
    r"C:\Program Files\SIMULIA\Commands",
    r"C:\Program Files\Dassault Systemes\SimulationServices\V6R2019x\win_b64\code\command",
)


def find_abaqus() -> str | None:
    """返回可执行的 abaqus 命令；找不到返回 None。"""
    found = shutil.which("abaqus")
    if found:
        return found
    for hint in _WINDOWS_HINTS:
        for name in ("abaqus.bat", "abq2019.bat", "abaqus"):
            path = Path(hint) / name
            if path.is_file():
                return str(path)
    return None


def abaqus_available() -> bool:
    return find_abaqus() is not None


def find_extractor(root: Path | None = None) -> Path | None:
    root = root or HERE.parent
    for rel in EXTRACTOR_CANDIDATES:
        path = root / rel
        if path.is_file():
            return path
    for path in root.rglob("extract_odb.py"):
        if ".venv" not in path.parts:
            return path
    return None


def scan_log(text: str) -> dict[str, Any]:
    """从 .msg / .dat 里挑出致命错误与警告。

    Abaqus 的返回码不总能反映分析是否真的完成，日志才是准的。
    """
    fatal = [line.strip() for line in text.splitlines()
             if any(p.search(line) for p in _FATAL_PATTERNS)]
    warnings = [m.group(0).strip() for m in _WARN_PATTERN.finditer(text)]
    return {"fatal": fatal[:10], "warnings": warnings[:10],
            "ok": not fatal}


def read_result_csv(path: Path) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            node = int(float(row["node"]))
            out[node] = {k: float(v) for k, v in row.items() if k != "node"}
    return out


def summarise(model: dict[str, Any], results: dict[int, dict[str, float]],
              case: str) -> dict[str, Any]:
    """整理成与 solve_model 对齐的结构，外加整体平衡校核。"""
    units = unit_system_of(model)
    nodes = {int(n["id"]): n for n in model.get("nodes", [])}
    supports = {int(s["node"]) for s in model.get("supports", [])}

    peak_node, peak = None, -1.0
    for nid, row in results.items():
        mag = math.sqrt(row.get("u1", 0.0) ** 2 + row.get("u2", 0.0) ** 2
                        + row.get("u3", 0.0) ** 2)
        if mag > peak:
            peak, peak_node = mag, nid

    reactions = {}
    total = 0.0
    for nid in sorted(supports):
        row = results.get(nid)
        if row is None:
            continue
        r = [row.get("rf1", 0.0), row.get("rf2", 0.0), row.get("rf3", 0.0)]
        n = nodes.get(nid)
        reactions[str(nid)] = {"xyz": [n["x"], n["y"], n["z"]] if n else None,
                               "R": [round(v / 1e3, 6) for v in r]}
        total += r[2]

    node = nodes.get(peak_node) or {}
    return {
        "backend": "abaqus",
        "case": case,
        "max_displacement_mm": round(peak * units.disp_scale, 6),
        "at_node": peak_node,
        "coordinates_xyz": [node.get("x"), node.get("y"), node.get("z")],
        "reactions": reactions,
        "vertical_total_kN": round(total / 1e3, 6),
        "nodes_returned": len(results),
    }


def normalized_error(ours: list[float], theirs: list[float]) -> float | None:
    """全场归一化相对误差。参考解整体为零时返回 None——那里归一化没有意义。

        e = sqrt(Σ(a−b)²) / sqrt(Σb²)

    这一份是唯一实现，对标脚本与界面里的并排对比都用它。
    """
    a = np.asarray(ours, dtype=float)
    b = np.asarray(theirs, dtype=float)
    denom = float(np.sqrt(np.sum(b ** 2)))
    if denom <= 1e-30:
        return None
    return float(np.sqrt(np.sum((a - b) ** 2)) / denom)


_COMPARE_FIELDS = (("u1", "位移 U1"), ("u2", "位移 U2"), ("u3", "位移 U3"),
                   ("ur1", "转角 UR1"), ("ur2", "转角 UR2"), ("ur3", "转角 UR3"),
                   ("rf1", "反力 RF1"), ("rf2", "反力 RF2"), ("rf3", "反力 RF3"))


def compare_rows(native: dict[int, dict[str, float]],
                 abaqus: dict[int, dict[str, float]],
                 displacement_scale: float = 1000.0) -> dict[str, Any]:
    """逐分量比较两个后端的节点结果。

    两侧行数据都保持模型原生单位，误差本身与单位无关；只有峰值展示需要
    用 ``displacement_scale`` 换成 mm。默认值保持 N-m-Pa 调用兼容。
    """
    shared = sorted(set(native) & set(abaqus))
    if not shared:
        return {"error": "两侧没有共同的节点编号"}
    out: dict[str, Any] = {"nodes_compared": len(shared), "errors": {}}
    for key, label in _COMPARE_FIELDS:
        out["errors"][key] = {
            "label": label,
            "e": normalized_error([native[n].get(key, 0.0) for n in shared],
                                  [abaqus[n].get(key, 0.0) for n in shared]),
        }
    mag_a = [math.sqrt(sum(native[n].get(k, 0.0) ** 2 for k in ("u1", "u2", "u3")))
             for n in shared]
    mag_b = [math.sqrt(sum(abaqus[n].get(k, 0.0) ** 2 for k in ("u1", "u2", "u3")))
             for n in shared]
    out["errors"]["|U|"] = {"label": "合位移 |U|", "e": normalized_error(mag_a, mag_b)}
    peak = int(np.argmax(mag_b)) if mag_b else 0
    out["peak"] = {"node": shared[peak],
                   "native_mm": round(mag_a[peak] * displacement_scale, 6),
                   "abaqus_mm": round(mag_b[peak] * displacement_scale, 6)}
    return out


def _run(cmd: list[str], cwd: Path, timeout: float) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                              timeout=timeout)
    except FileNotFoundError as exc:
        raise AbaqusError(
            "找不到 abaqus 命令。从开始菜单打开「Abaqus Command」窗口再运行，"
            "或把 Abaqus 的 commands 目录加进 PATH。") from exc
    except subprocess.TimeoutExpired as exc:
        raise AbaqusError(f"Abaqus 作业超过 {timeout:.0f} 秒仍未结束，已放弃。") from exc


def solve(model: dict[str, Any], workdir: Path, case: str | None = None,
          element: str = "B33", job: str = "agentjob",
          timeout: float = 900.0) -> dict[str, Any]:
    """跑一次 Abaqus，返回与 solve_model 对齐的结果摘要。"""
    exe = find_abaqus()
    if exe is None:
        raise AbaqusError(
            "找不到 abaqus 命令。若本机装了 Abaqus，请从开始菜单的「Abaqus Command」"
            "里启动本程序（那个终端才把 abaqus 放进 PATH）。"
            "默认的自研求解器不需要它。")
    extractor = find_extractor()
    if extractor is None:
        raise AbaqusError("找不到 extract_odb.py，无法读取 ODB。")

    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    # Abaqus 2019 及更早版本的启动脚本/Python 2 对 Unicode 路径支持不完整。
    # 项目放在“桌面\结构计算”一类目录时，它可能只留下 .com 就退出。自动在
    # 系统临时目录的纯 ASCII 路径中计算，结束后把可追溯文件复制回用户目录。
    staging = None
    run_dir = workdir
    if not str(workdir).isascii():
        staging = tempfile.TemporaryDirectory(prefix="spaceframe_abaqus_")
        run_dir = Path(staging.name)

    scan: dict[str, Any] = {"warnings": []}
    try:
        inp = write_inp(model, element, run_dir / f"{job}.inp", case=case)
        proc = _run([exe, f"job={job}", f"input={inp.name}", "interactive",
                     "ask_delete=OFF"], run_dir, timeout)

        logs = ""
        for suffix in (".msg", ".dat", ".sta", ".log"):
            path = run_dir / f"{job}{suffix}"
            if path.is_file():
                logs += path.read_text(encoding="utf-8", errors="replace")
        scan = scan_log(
            logs + "\n" + (proc.stdout or "") + "\n" + (proc.stderr or ""))

        odb = run_dir / f"{job}.odb"
        if not scan["ok"] or not odb.is_file():
            detail = "；".join(scan["fatal"][:3]) if scan["fatal"] else (
                (proc.stderr or proc.stdout or "").strip()[:300]
                or f"没有生成 {odb.name}，返回码 {proc.returncode}。")
            raise AbaqusError("Abaqus 作业未成功。" + detail)

        # extractor 本身也复制到 ASCII 工作目录，避免 Abaqus 自带 Python 2
        # 无法打开中文项目路径中的脚本。
        local_extractor = run_dir / "extract_odb.py"
        if extractor.resolve() != local_extractor.resolve():
            shutil.copy2(extractor, local_extractor)
        extract = _run([exe, "python", local_extractor.name, odb.name],
                       run_dir, timeout)
        csv_path = run_dir / f"{job}_abaqus.csv"
        if not csv_path.is_file():
            raise AbaqusError(
                "ODB 导出失败：" + (extract.stderr or extract.stdout or "无输出")[:300])

        summary = summarise(model, read_result_csv(csv_path), case or "default")
        summary["element"] = element
        summary["warnings"] = scan["warnings"][:5]
    finally:
        if staging is not None:
            # 失败也复制日志，用户不必去临时目录里找诊断依据。
            for path in run_dir.glob(f"{job}*"):
                if path.is_file():
                    shutil.copy2(path, workdir / path.name)
            staging.cleanup()

    inp = workdir / f"{job}.inp" if staging is not None else inp
    odb = workdir / f"{job}.odb" if staging is not None else odb
    csv_path = workdir / f"{job}_abaqus.csv" if staging is not None else csv_path
    summary["files"] = {"inp": str(inp), "odb": str(odb), "csv": str(csv_path)}
    return summary
