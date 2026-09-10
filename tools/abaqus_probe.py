"""standard.exe / pre.exe 崩溃的二分探针。

背景：局部实体作业在 Windows 临时目录（纯 ASCII 路径）里跑时，pre.exe 正常
跑完、许可证正常签出，然后 standard.exe 以系统错误码 1073741795 中止，.dat
里一条 ***ERROR 都没有。

第一版探针把作业放在项目文件夹里跑，结果 **pre.exe 就崩了**，错误码却是
另一个（529697949）。项目路径含中文（agent开发），所以那一版探针测的根本
不是它想测的东西——它自己的工作目录就引入了一个新变量。这一版把路径单独
拿出来做对照。

三个探针，每个都用 abaqus job= 直接跑，不经过 CAE：

  P1a  手写的单 C3D10 单元输入文件，放在 **ASCII 临时目录**
       它崩 -> 这台机器跑不了 C3D10，与我们的模型无关
  P1b  同一个输入文件，放在 **项目文件夹**（路径含中文）
       P1a 过而 P1b 崩 -> 是路径里的非 ASCII 字符，不是模型
  P2   上次失败留下的 solid_joint_0.inp，放在 ASCII 临时目录
       P1a 过而 P2 崩 -> 是我们生成的几何或网格

用法：run_abaqus_probe.bat
"""

from __future__ import annotations

import datetime as _dt
import shutil
import subprocess
import sys
import tempfile
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


def run_job(executable: str, directory: Path, job: str, keep: Path,
            timeout: float = 900.0) -> bool:
    """跑一个输入文件，把控制台和 .log/.sta/.dat 原样记下来。

    产物复制到 ``keep`` 保存——临时目录会被清掉，而证据不能跟着没。
    复制是 Python 干的，落在含中文的路径上没问题；Abaqus 自己不碰那儿。
    """
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

    keep.mkdir(parents=True, exist_ok=True)
    for path in directory.iterdir():
        if path.is_file():
            try:
                shutil.copy2(path, keep / path.name)
            except OSError:
                pass
    for suffix in (".log", ".sta"):
        path = directory / (job + suffix)
        say(f"--- {job}{suffix} ---")
        if path.is_file():
            say(path.read_text(encoding="utf-8", errors="replace").strip() or "(空)")
        else:
            say("(没有生成)")

    status = directory / (job + ".sta")
    if status.is_file() and "COMPLETED SUCCESSFULLY" in status.read_text(
            encoding="utf-8", errors="replace"):
        say(">>> 结论：跑通了")
        return True
    say(">>> 结论：没跑通")
    return False


def probe(executable: str, title: str, why: str, deck: str, job: str,
          directory: Path, keep_name: str) -> bool:
    say()
    say("=" * 70)
    say(f"# {title}")
    say("=" * 70)
    say(why)
    say()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / (job + ".inp")).write_text(deck, encoding="ascii")
    return run_job(executable, directory, job, WORK / keep_name)


def main() -> int:
    import abaqus_backend

    say(f"Abaqus 探针　{_dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    executable = abaqus_backend.find_abaqus()
    say(f"find_abaqus() -> {executable!r}")
    if executable is None:
        say("找不到 Abaqus，无法继续。")
        return 1
    WORK.mkdir(parents=True, exist_ok=True)

    ascii_root = Path(tempfile.mkdtemp(prefix="abqprobe_"))
    say(f"ASCII 临时目录：{ascii_root}")
    say(f"项目目录（含中文）：{ROOT}")

    p1a = probe(executable,
                "P1a　单个 C3D10 单元，ASCII 临时目录",
                "它崩 => 这台机器跑不了 C3D10 实体，与我们的模型无关。",
                MINIMAL_C3D10, "probe_c3d10", ascii_root / "p1a", "p1a_ascii")

    p1b = probe(executable,
                "P1b　同一个输入文件，放在项目文件夹（路径含中文）",
                "P1a 过而这个崩 => 是路径里的非 ASCII 字符，不是模型。",
                MINIMAL_C3D10, "probe_c3d10", WORK / "p1b_cjk_path", "p1b_cjk_path")

    source = ROOT / "results" / "solid_joint" / "node_2_LC1" / "solid_joint_0.inp"
    say()
    say("=" * 70)
    say("# P2　上次失败留下的 solid_joint_0.inp，ASCII 临时目录")
    say("=" * 70)
    if not source.is_file():
        say(f"找不到 {source}，跳过。先跑一次 run_joint.bat 让它留下输入文件。")
        p2 = None
    else:
        say("P1a 过而这个崩 => 是我们生成的几何或网格。")
        say(f"输入文件来自 {source}")
        say()
        p2_dir = ascii_root / "p2"
        p2_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, p2_dir / "solid_joint_0.inp")
        p2 = run_job(executable, p2_dir, "solid_joint_0", WORK / "p2_joint")

    say()
    say("=" * 70)
    say("# 结论")
    say("=" * 70)
    say(f"P1a（C3D10 / ASCII 路径）　　　{'通过' if p1a else '失败'}")
    say(f"P1b（同一文件 / 中文路径）　　{'通过' if p1b else '失败'}")
    say(f"P2 （我们的模型 / ASCII 路径）{'跳过' if p2 is None else ('通过' if p2 else '失败')}")
    say()
    if not p1a:
        say("P1a 就崩了 —— 这台机器跑不了 C3D10 实体单元。一个单元、十个节点的")
        say("输入文件都过不去，不可能是模型的问题。排查方向是 Abaqus 安装本身。")
    elif not p1b:
        say("路径里的中文是致命的：同一个输入文件，ASCII 路径能跑、项目路径不能。")
        say("局部实体分析本来就在临时目录里跑，所以这一条不是它失败的原因；")
        say("但 abaqus_bench 是直接在项目文件夹里跑作业的，那套基准现在必然是坏的。")
        if p2 is False:
            say("而 P2 在 ASCII 路径下仍然崩 —— 那才是局部实体的真正问题，")
            say("方向是我们生成的几何或网格（上次报了 725 个畸变单元）。")
        elif p2:
            say("P2 在 ASCII 路径下跑通了 —— 那么问题在 CAE 提交作业这条路径上，")
            say("而不是求解器本身。改成先写输入文件、再单独调求解器即可。")
    elif p2 is False:
        say("C3D10 与路径都没问题，是我们生成的这个模型的几何或网格。")
        say("最可能的是布尔合并相贯区的畸变单元（上次报了 725 个）。")
    else:
        say("全部通过 —— 那么问题在 CAE 提交作业这条路径上，而不是求解器。")
        say("改成先写输入文件、再单独调求解器即可。")
    say()
    say(f"各作业产物已保存到 {WORK}")
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
