"""standard.exe 崩溃的二分探针。

背景：局部实体作业里 pre.exe 跑完、许可证正常签出，然后 standard.exe 以
Windows 系统错误码中止，.dat 里一条 ***ERROR 都没有。这种崩法有两类原因，
必须先分开，否则改哪儿都是猜：

  A. 这台机器上 C3D10 实体单元 / 稀疏求解器本身就跑不了
     （已知它跑得动 70 方程的 B33 梁算例，但那是完全不同的代码路径）
  B. C3D10 没问题，是我们这个模型的几何或网格有问题
     （布尔合并出来的相贯区有 725 个畸变单元）

两个探针：

  P1  一个手写的单 C3D10 单元输入文件，**不经过 CAE**，直接交给求解器。
      它崩 -> 属于 A，跟我们的模型无关。
  P2  把上一次失败留下的 solid_joint_0.inp 直接用 abaqus job= 跑，
      同样绕开 CAE。CAE 提交时不落 .log，直接跑会落，报错更完整。

P1 过、P2 崩 -> 问题在我们的几何/网格。
两个都崩 -> 问题在这台机器的实体求解路径，我们的代码没得改。

用法：run_abaqus_probe.bat
"""

from __future__ import annotations

import datetime as _dt
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

WORK = ROOT / "results" / "abaqus_probe"
LOG_PATH = WORK / "probe_log.txt"
_lines: list[str] = []


def say(text: str = "") -> None:
    print(text)
    _lines.append(text)


# 单个 C3D10 四面体：角点 1-4，中间点 5=(1,2) 6=(2,3) 7=(3,1) 8=(1,4)
# 9=(2,4) 10=(3,4)。底面三点固定，顶点加一个力。
# 手写输入文件是有意的——这样连 CAE 都不需要，只考验求解器本身。
MINIMAL_C3D10 = """\
*HEADING
minimal C3D10 probe
*NODE
1,  0.0, 0.0, 0.0
2, 10.0, 0.0, 0.0
3,  0.0,10.0, 0.0
4,  0.0, 0.0,10.0
5,  5.0, 0.0, 0.0
6,  5.0, 5.0, 0.0
7,  0.0, 5.0, 0.0
8,  0.0, 0.0, 5.0
9,  5.0, 0.0, 5.0
10, 0.0, 5.0, 5.0
*ELEMENT, TYPE=C3D10, ELSET=EALL
1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
*SOLID SECTION, ELSET=EALL, MATERIAL=STEEL
*MATERIAL, NAME=STEEL
*ELASTIC
206000., 0.3
*NSET, NSET=FIXED
1, 2, 3, 5, 6, 7
*NSET, NSET=TIP
4
*STEP
*STATIC
*BOUNDARY
FIXED, 1, 3
*CLOAD
TIP, 3, -1000.0
*NODE PRINT
U,
*END STEP
"""


def run_job(executable: str, directory: Path, job: str,
            timeout: float = 900.0) -> bool:
    """跑一个输入文件，把控制台和 .log/.sta 原样记下来。"""
    say(f"命令：{executable} job={job} interactive")
    say(f"目录：{directory}")
    try:
        proc = subprocess.run(
            [executable, f"job={job}", "interactive"], cwd=str(directory),
            capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        say(f"!! 超过 {timeout:.0f} 秒未结束")
        return False
    say(f"返回码 = {proc.returncode}")
    say("--- stdout ---")
    say((proc.stdout or "").strip() or "(空)")
    say("--- stderr ---")
    say((proc.stderr or "").strip() or "(空)")
    for suffix in (".log", ".sta"):
        path = directory / (job + suffix)
        say(f"--- {job}{suffix} ---")
        if path.is_file():
            say(path.read_text(encoding="utf-8", errors="replace").strip() or "(空)")
        else:
            say("(没有生成)")
    ok = proc.returncode == 0
    status = directory / (job + ".sta")
    if status.is_file() and "COMPLETED SUCCESSFULLY" in status.read_text(
            encoding="utf-8", errors="replace"):
        say(">>> 结论：跑通了")
        return True
    say(">>> 结论：没跑通")
    return ok and False


def main() -> int:
    import abaqus_backend

    say(f"Abaqus 探针　{_dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    executable = abaqus_backend.find_abaqus()
    say(f"find_abaqus() -> {executable!r}")
    if executable is None:
        say("找不到 Abaqus，无法继续。")
        return 1

    WORK.mkdir(parents=True, exist_ok=True)

    say()
    say("=" * 70)
    say("# P1　单个 C3D10 单元（手写输入文件，不经过 CAE）")
    say("=" * 70)
    say("它崩 => 这台机器的实体求解路径本身有问题，与我们的模型无关。")
    say()
    p1_dir = WORK / "p1_minimal"
    p1_dir.mkdir(exist_ok=True)
    (p1_dir / "probe_c3d10.inp").write_text(MINIMAL_C3D10, encoding="ascii")
    p1_ok = run_job(executable, p1_dir, "probe_c3d10")

    say()
    say("=" * 70)
    say("# P2　上次失败留下的 solid_joint_0.inp（同样绕开 CAE）")
    say("=" * 70)
    source = ROOT / "results" / "solid_joint" / "node_2_LC1" / "solid_joint_0.inp"
    if not source.is_file():
        say(f"找不到 {source}，跳过。先跑一次 run_joint.bat 让它留下输入文件。")
        p2_ok = None
    else:
        p2_dir = WORK / "p2_joint"
        p2_dir.mkdir(exist_ok=True)
        shutil.copy2(source, p2_dir / "solid_joint_0.inp")
        say(f"输入文件来自 {source}")
        say()
        p2_ok = run_job(executable, p2_dir, "solid_joint_0")

    say()
    say("=" * 70)
    say("# 结论")
    say("=" * 70)
    if not p1_ok:
        say("P1 就崩了 —— 这台机器跑不了 C3D10 实体，问题不在我们的模型里。")
        say("排查方向是 Abaqus 安装本身，不是这个项目的代码。")
    elif p2_ok is False:
        say("P1 通过、P2 崩 —— C3D10 没问题，是我们生成的这个模型的几何或网格。")
        say("最可能的是布尔合并相贯区的畸变单元（上次报了 725 个）。")
    elif p2_ok:
        say("两个都通过 —— 那么问题出在 CAE 提交作业这条路径上，")
        say("而不是求解器。可以改成先写输入文件、再单独调求解器。")
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    finally:
        WORK.mkdir(parents=True, exist_ok=True)
        LOG_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
        print(f"\n日志已写入 {LOG_PATH}")
    sys.exit(code)
