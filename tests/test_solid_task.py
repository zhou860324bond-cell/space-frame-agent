from dataclasses import dataclass

from desktop.solid_task import run_backend_pair


@dataclass
class Result:
    ok: bool
    payload: dict


def test_pair_reuses_completed_native_stage_and_only_runs_abaqus():
    calls = []

    def run(backend, sizes):
        calls.append((backend, sizes))
        return Result(True, {"backend": backend})

    got = run_backend_pair(
        run, {"native": [40.0, 30.0, 20.0], "abaqus": [8.0, 5.6, 4.0]},
        cached={"native": {"backend": "native-c3d10+gmsh"}})

    assert got["ok"] is True
    assert calls == [("abaqus", [8.0, 5.6, 4.0])]
    assert set(got["completed"]) == {"native", "abaqus"}


def test_pair_keeps_first_stage_when_second_stage_fails():
    def run(backend, _sizes):
        return (Result(True, {"backend": "native-c3d10+gmsh"})
                if backend == "native" else
                Result(False, {"error": "Abaqus license unavailable"}))

    got = run_backend_pair(
        run, {"native": [40.0, 30.0, 20.0], "abaqus": [8.0, 5.6, 4.0]})

    assert got["ok"] is False
    assert got["failed_backend"] == "abaqus"
    assert set(got["completed"]) == {"native"}


def test_pair_cancellation_is_checked_before_starting_the_next_backend():
    cancelled = False
    calls = []

    def run(backend, _sizes):
        nonlocal cancelled
        calls.append(backend)
        cancelled = True
        return Result(True, {"backend": "native-c3d10+gmsh"})

    got = run_backend_pair(
        run, {"native": [40.0, 30.0, 20.0], "abaqus": [8.0, 5.6, 4.0]},
        is_cancelled=lambda: cancelled)

    assert got["cancelled"] is True
    assert calls == ["native"]
    assert set(got["completed"]) == {"native"}
