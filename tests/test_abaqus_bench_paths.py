"""对标基准的作业必须跑在纯 ASCII 路径上。

实测结论：同一个输入文件，放在 ASCII 目录下 Abaqus 报
`THE ANALYSIS HAS COMPLETED SUCCESSFULLY`；放在含中文的目录下，pre.exe 以系统
错误码 529697949 中止，`.dat` 里一条 ***ERROR 都没有——**失败得毫无痕迹**。

这个仓库的默认位置就带中文（agent开发），而 run_jobs.py 原来是在 .inp 所在
目录里提交作业的。也就是说整套 Abaqus 对标基准会静默失效，而它正是报告里
"与商软对标"那一节的全部证据。

这里不需要装 Abaqus：换掉 subprocess.run，只验证**作业跑在哪个目录**、
产物有没有复制回来。
"""

from __future__ import annotations

from pathlib import Path


import run_jobs


class _Silent:
    def write(self, *args) -> None:
        pass

    def flush(self) -> None:
        pass


def _fake_abaqus(seen: dict, *, succeed: bool = True):
    def run(cmd, cwd, capture_output, text, timeout, env=None):
        seen["cwd"] = cwd
        seen["cmd"] = cmd
        seen["env"] = env
        directory = Path(cwd)
        assert (directory / "demo.inp").is_file(), "输入文件没有被带到运行目录"
        if succeed:
            (directory / "demo.dat").write_text(
                "THE ANALYSIS HAS BEEN COMPLETED\n", encoding="utf-8")
            (directory / "demo.sta").write_text(
                "THE ANALYSIS HAS COMPLETED SUCCESSFULLY\n", encoding="utf-8")

        class Result:
            returncode = 0 if succeed else 1
            stdout = "done"
            stderr = ""

        return Result()

    return run


def _job_in(tmp_path: Path) -> Path:
    cjk = tmp_path / "agent开发" / "abaqus_bench" / "inp"
    cjk.mkdir(parents=True)
    inp = cjk / "demo.inp"
    inp.write_text("*HEADING\ndemo\n", encoding="ascii")
    return inp


def test_the_job_never_runs_in_a_non_ascii_directory(tmp_path, monkeypatch):
    inp = _job_in(tmp_path)
    seen: dict = {}
    monkeypatch.setattr(run_jobs.subprocess, "run", _fake_abaqus(seen))

    run_jobs.run_one(inp, 60.0, _Silent())

    assert seen["cwd"].isascii(), f"作业跑在了非 ASCII 路径：{seen['cwd']}"
    assert str(inp.parent) not in seen["cwd"], "作业不该在 .inp 所在目录里跑"


def test_the_outputs_come_back_next_to_the_input(tmp_path, monkeypatch):
    """跑在临时目录是实现细节，外部行为不能变：产物仍要落在 .inp 旁边。"""
    inp = _job_in(tmp_path)
    monkeypatch.setattr(run_jobs.subprocess, "run", _fake_abaqus({}))

    result = run_jobs.run_one(inp, 60.0, _Silent())

    assert (inp.parent / "demo.dat").is_file(), "产物没有复制回来"
    assert (inp.parent / "demo.sta").is_file()
    assert result["ok"], "完成判定仍应基于复制回来的 .dat"


def test_a_failed_job_is_still_reported_as_failed(tmp_path, monkeypatch):
    """临时目录不能把失败吞掉——没有 .dat 就是没跑完。"""
    inp = _job_in(tmp_path)
    monkeypatch.setattr(run_jobs.subprocess, "run",
                        _fake_abaqus({}, succeed=False))

    result = run_jobs.run_one(inp, 60.0, _Silent())

    assert not result["ok"]
    assert result["note"]


def test_the_input_file_itself_is_not_copied_back_over_itself(tmp_path, monkeypatch):
    """只把产物搬回来。原样复制 .inp 会白改它的时间戳，
    让 --force 之外的续跑判断变得不可预料。"""
    inp = _job_in(tmp_path)
    before = inp.stat().st_mtime_ns
    monkeypatch.setattr(run_jobs.subprocess, "run", _fake_abaqus({}))

    run_jobs.run_one(inp, 60.0, _Silent())

    assert inp.stat().st_mtime_ns == before


def test_the_bench_also_gets_the_amd_workaround(tmp_path, monkeypatch):
    """对标作业和局部实体走同一个求解器，同一台机器上就该带同一套环境变量。
    漏掉这里，报告里"与商软对标"那一节会在 AMD 上整批静默失败。"""
    import abaqus_backend

    inp = _job_in(tmp_path)
    seen: dict = {}
    monkeypatch.setattr(run_jobs.subprocess, "run", _fake_abaqus(seen))
    monkeypatch.setattr(abaqus_backend, "is_amd_cpu", lambda env=None: True)

    run_jobs.run_one(inp, 60.0, _Silent())

    assert seen["env"] is not None, "没有传环境变量"
    assert seen["env"].get("MKL_DEBUG_CPU_TYPE") == "5"
