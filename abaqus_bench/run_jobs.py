r"""逐个提交 Abaqus 作业，带日志、可续跑、单个失败不影响其余。

**为什么不用 .bat 的 for 循环**：cmd 里 `abaqus` 本身是个批处理，它内部的 `exit`
会把父脚本一并杀掉；`for` 对通配符的枚举在目录被大量写入时也不稳。
实际现象就是跑了几个作业之后整批静默中止，连日志都不留。
用 Python 的 subprocess 逐个调起，每个作业独立，失败也只影响它自己。

用法（在装了 Abaqus 的机器上，Abaqus Command 窗口或 PATH 里有 abaqus 即可）：

    python abaqus_bench\run_jobs.py                跑 inp/ 下所有还没结果的作业
    python abaqus_bench\run_jobs.py --force        已完成的也重跑
    python abaqus_bench\run_jobs.py --only space_frame_B33
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import sys
import time
from pathlib import Path


from abaqus_backend import solver_environment
from console import use_utf8   # 见 src/console.py：别让一个字符打死一次成功的运行

use_utf8()
HERE = Path(__file__).resolve().parent
DEFAULT_INP_DIR = HERE / "inp"
DONE_MARK = "THE ANALYSIS HAS BEEN COMPLETED"


def already_done(inp: Path) -> bool:
    dat = inp.with_suffix(".dat")
    return dat.is_file() and DONE_MARK in dat.read_text(encoding="utf-8",
                                                        errors="replace")


def run_one(inp: Path, timeout: float, log) -> dict:
    """跑一个作业。用 cmd /c 隔离，避免 abaqus.bat 里的 exit 波及本进程。

    **作业在纯 ASCII 的临时目录里跑，不在 inp 所在目录。** 实测：同一个输入
    文件，放在 ASCII 路径下 `THE ANALYSIS HAS COMPLETED SUCCESSFULLY`；放在
    含中文的项目路径下，pre.exe 以系统错误码 529697949 中止，一条 ***ERROR
    都没有。这个仓库的默认位置就带中文（agent开发），所以原来 cwd=inp.parent
    的写法会让整套对标基准**静默失效**——而它正是"与商软对标"那一节的全部证据。

    产物跑完复制回 inp 所在目录，外部行为不变（already_done 仍看那边的 .dat）。
    复制是 Python 干的，落在含中文的路径上没问题；Abaqus 自己不碰那儿。
    """
    job = inp.stem
    started = time.time()
    cmd = ["abaqus", f"job={job}", f"input={inp.name}", "interactive", "ask_delete=OFF"]
    if sys.platform.startswith("win"):
        cmd = ["cmd", "/c"] + cmd
    try:
        with tempfile.TemporaryDirectory(prefix="abq_bench_") as temp:
            run_dir = Path(temp)
            shutil.copy2(inp, run_dir / inp.name)
            proc = subprocess.run(cmd, cwd=str(run_dir), capture_output=True,
                                  text=True, timeout=timeout,
                                  env=solver_environment())
            code = proc.returncode
            output = (proc.stdout or "") + (proc.stderr or "")
            for produced in run_dir.iterdir():
                if produced.is_file() and produced.name != inp.name:
                    try:
                        shutil.copy2(produced, inp.parent / produced.name)
                    except OSError as exc:
                        output += f"\n[复制 {produced.name} 回来失败：{exc}]"
    except subprocess.TimeoutExpired:
        code, output = -1, f"超过 {timeout:.0f} 秒未结束"
    except FileNotFoundError:
        return {"job": job, "ok": False, "seconds": 0.0,
                "note": "找不到 abaqus 命令"}

    elapsed = time.time() - started
    ok = already_done(inp)
    log.write(f"\n{'=' * 70}\n{job}  返回码 {code}  用时 {elapsed:.1f}s  "
              f"{'完成' if ok else '未完成'}\n{'=' * 70}\n{output}\n")
    log.flush()

    note = ""
    if not ok:
        msg = inp.with_suffix(".msg")
        if msg.is_file():
            fatal = [l.strip() for l in msg.read_text(encoding="utf-8",
                                                      errors="replace").splitlines()
                     if "***ERROR" in l.upper()]
            note = fatal[0][:120] if fatal else "没有生成完整的 .dat"
        else:
            note = f"Abaqus 未产生输出（返回码 {code}）"
    return {"job": job, "ok": ok, "seconds": elapsed, "note": note}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp-dir", default=str(DEFAULT_INP_DIR))
    parser.add_argument("--force", action="store_true", help="已完成的也重跑")
    parser.add_argument("--only", nargs="*", help="只跑这些作业名")
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()

    if shutil.which("abaqus") is None:
        print("PATH 上没有 abaqus 命令。")
        print("请从开始菜单打开「Abaqus Command」窗口，在那里运行本脚本。")
        raise SystemExit(1)

    inp_dir = Path(args.inp_dir)
    jobs = sorted(inp_dir.glob("*.inp"))
    if args.only:
        wanted = set(args.only)
        jobs = [j for j in jobs if j.stem in wanted]
    if not jobs:
        print(f"{inp_dir} 下没有 .inp。先跑 export_inp.py 生成。")
        raise SystemExit(1)

    pending = jobs if args.force else [j for j in jobs if not already_done(j)]
    skipped = len(jobs) - len(pending)
    print(f"共 {len(jobs)} 个作业，已完成 {skipped} 个，本次跑 {len(pending)} 个\n")
    if not pending:
        print("全部已完成。加 --force 可重跑。")
        return

    log_path = inp_dir / "run_jobs.log"
    results = []
    with log_path.open("w", encoding="utf-8") as log:
        for index, inp in enumerate(pending, 1):
            print(f"[{index}/{len(pending)}] {inp.stem} … ", end="", flush=True)
            r = run_one(inp, args.timeout, log)
            results.append(r)
            print(f"{'完成' if r['ok'] else '失败'}  {r['seconds']:.1f}s"
                  + (f"  {r['note']}" if r["note"] else ""))

    good = [r for r in results if r["ok"]]
    print(f"\n本次 {len(good)}/{len(results)} 成功，"
          f"合计 {sum(r['seconds'] for r in results):.0f} 秒")
    if len(good) < len(results):
        print(f"失败的作业详见 {log_path}")
    print("\n接下来回项目目录跑对比：")
    print("    python abaqus_bench\\compare.py --element B33")
    print("    python abaqus_bench\\compare.py --element B31")


if __name__ == "__main__":
    main()
