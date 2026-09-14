import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from solid_cache import (load_all_solid_records, load_solid_cache, load_solid_history,
                         load_solid_orphans,
                         make_solid_cache_identity, start_solid_run,
                         move_solid_history_to_trash,
                         move_solid_orphan_to_trash,
                         ORPHAN_MIN_AGE_SECONDS,
                         SolidHistoryCleanupError,
                         solid_history_storage, write_solid_summary)


MODEL = {"nodes": [{"id": 2, "xyz": [1.0, 2.0, 3.0]}]}
SPEC = SimpleNamespace(
    node_id=2, case="P", anchor_member=1,
    force_n=(1000.0, 0.0, 0.0), moment_nmm=(0.0, 20.0, 0.0))
PLANS = {"native": [40.0, 30.0, 20.0], "abaqus": [8.0, 5.6, 4.0]}


def _write_cache(root: Path, backend: str, *, artifact=True,
                 identity=None, converged=True) -> Path:
    directory = ("native_solid_joint" if backend == "native" else
                 "solid_joint")
    artifact_key = "finest_vtu" if backend == "native" else "finest_odb"
    artifact_path = root / "artifacts" / (
        "native_joint_2.vtu" if backend == "native" else "solid_joint_2.odb")
    if artifact:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(b"valid-result")
    payload = {
        "schema": ("solid-joint-analysis/native-v1" if backend == "native"
                   else "solid-joint-analysis/v2"),
        "backend": ("native-c3d10+gmsh" if backend == "native"
                    else "abaqus-6.14"),
        "node_id": 2,
        "case": "P",
        "hot_spot_all_converged": converged,
        "meshes": [({"global_mesh_size_mm": value}
                    if backend == "native" else {"mesh_size_mm": value})
                   for value in PLANS[backend]],
        "cache_identity": (identity if identity is not None else
                           make_solid_cache_identity(MODEL, SPEC, PLANS[backend])),
        "files": {artifact_key: str(artifact_path)},
    }
    summary = (root / "results" / directory / "node_2_P" / "summary.json")
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(payload), encoding="utf-8")
    return summary


def test_loads_only_complete_matching_native_and_abaqus_results(tmp_path):
    _write_cache(tmp_path, "native")
    _write_cache(tmp_path, "abaqus")

    got = load_solid_cache(MODEL, SPEC, PLANS, root=tmp_path)

    assert set(got["usable"]) == {"native", "abaqus"}
    assert got["ignored"] == {}


def test_rejects_same_node_and_case_when_model_or_joint_input_changed(tmp_path):
    _write_cache(tmp_path, "native")

    changed_model = {"nodes": [{"id": 2, "xyz": [9.0, 2.0, 3.0]}]}
    changed_spec = SimpleNamespace(**{**vars(SPEC), "force_n": (2000.0, 0.0, 0.0)})

    assert load_solid_cache(changed_model, SPEC, PLANS, root=tmp_path)["usable"] == {}
    assert load_solid_cache(MODEL, changed_spec, PLANS, root=tmp_path)["usable"] == {}


def test_rejects_changed_mesh_plan_and_unconverged_result(tmp_path):
    _write_cache(tmp_path, "native")
    changed_plans = {**PLANS, "native": [40.0, 28.0, 20.0]}
    assert load_solid_cache(MODEL, SPEC, changed_plans, root=tmp_path)["usable"] == {}

    _write_cache(tmp_path, "native", converged=False)
    assert load_solid_cache(MODEL, SPEC, PLANS, root=tmp_path)["usable"] == {}


def test_ignores_legacy_corrupt_or_incomplete_cache_without_raising(tmp_path):
    summary = _write_cache(tmp_path, "native", artifact=False, identity={})
    got = load_solid_cache(MODEL, SPEC, PLANS, root=tmp_path)
    assert got["usable"] == {}
    assert "native" in got["ignored"]

    summary.write_text("{broken", encoding="utf-8")
    got = load_solid_cache(MODEL, SPEC, PLANS, root=tmp_path)
    assert got["usable"] == {}
    assert "损坏" in got["ignored"]["native"]

    _write_cache(tmp_path, "native", artifact=False)
    got = load_solid_cache(MODEL, SPEC, PLANS, root=tmp_path)
    assert got["usable"] == {}
    assert "缺失或为空" in got["ignored"]["native"]


def test_each_solid_run_has_an_independent_directory_and_atomic_latest(tmp_path):
    output = tmp_path / "results" / "native_solid_joint"
    first = start_solid_run(output, 2, "P")
    second = start_solid_run(output, 2, "P")
    assert first["run_id"] != second["run_id"]
    assert first["run_dir"].parent == second["run_dir"].parent

    base = {"backend": "native-c3d10+gmsh", "node_id": 2, "case": "P"}
    one = {**base, "run_id": first["run_id"],
           "generated_at": first["generated_at"], "marker": 1}
    two = {**base, "run_id": second["run_id"],
           "generated_at": second["generated_at"], "marker": 2}
    write_solid_summary(one, first["case_dir"], first["run_dir"])
    write_solid_summary(two, second["case_dir"], second["run_dir"])

    assert json.loads((first["run_dir"] / "summary.json").read_text())["marker"] == 1
    assert json.loads((second["run_dir"] / "summary.json").read_text())["marker"] == 2
    latest = json.loads((second["case_dir"] / "summary.json").read_text())
    assert latest["run_id"] == second["run_id"]
    assert not list(second["case_dir"].glob(".summary.*.tmp"))


def test_history_lists_run_summaries_without_duplicating_latest(tmp_path):
    output = tmp_path / "results" / "solid_joint"
    runs = [start_solid_run(output, 2, "P") for _ in range(2)]
    for index, run in enumerate(runs):
        payload = {
            "backend": "abaqus-6.14", "node_id": 2, "case": "P",
            "run_id": run["run_id"], "generated_at": run["generated_at"],
            "stress_concentration_factor": 1.2 + 0.1 * index,
        }
        write_solid_summary(payload, run["case_dir"], run["run_dir"])

    history = load_solid_history(2, "P", root=tmp_path)

    assert len(history) == 2
    assert {item["run_id"] for item in history} == {run["run_id"] for run in runs}
    assert all(item["_history_id"].startswith("abaqus:") for item in history)


def test_history_storage_counts_only_the_independent_run_directory(tmp_path):
    output = tmp_path / "results" / "solid_joint"
    run = start_solid_run(output, 2, "P")
    payload = {
        "backend": "abaqus-6.14", "node_id": 2, "case": "P",
        "run_id": run["run_id"], "generated_at": run["generated_at"],
    }
    write_solid_summary(payload, run["case_dir"], run["run_dir"])
    nested = run["run_dir"] / "artifacts"
    nested.mkdir()
    (nested / "result.odb").write_bytes(b"123456789")
    loaded = load_solid_history(2, "P", root=tmp_path)[0]

    storage = solid_history_storage(loaded)

    assert storage["valid"] is True
    assert storage["latest"] is True
    assert storage["files"] == 2
    assert storage["bytes"] >= 9


def test_history_cleanup_rejects_latest_protected_and_flat_results(tmp_path):
    output = tmp_path / "results" / "solid_joint"
    runs = [start_solid_run(output, 2, "P") for _ in range(2)]
    for run in runs:
        write_solid_summary({
            "backend": "abaqus-6.14", "node_id": 2, "case": "P",
            "run_id": run["run_id"], "generated_at": run["generated_at"],
        }, run["case_dir"], run["run_dir"])
    history = load_solid_history(2, "P", root=tmp_path)
    latest = next(item for item in history
                  if item["run_id"] == runs[-1]["run_id"])
    old = next(item for item in history if item["run_id"] == runs[0]["run_id"])
    mover = lambda _path: True

    with pytest.raises(SolidHistoryCleanupError, match="最新成功结果"):
        move_solid_history_to_trash(latest, set(), mover)
    with pytest.raises(SolidHistoryCleanupError, match="A/B"):
        move_solid_history_to_trash(old, {old["_history_id"]}, mover)
    flat = {**old, "_history_summary": str(runs[0]["case_dir"] / "summary.json")}
    with pytest.raises(SolidHistoryCleanupError, match="runs"):
        move_solid_history_to_trash(flat, set(), mover)


def test_history_cleanup_moves_only_an_unprotected_old_run(tmp_path):
    output = tmp_path / "results" / "native_solid_joint"
    runs = [start_solid_run(output, 2, "P") for _ in range(2)]
    for run in runs:
        write_solid_summary({
            "backend": "native-c3d10+gmsh", "node_id": 2, "case": "P",
            "run_id": run["run_id"], "generated_at": run["generated_at"],
        }, run["case_dir"], run["run_dir"])
    history = load_solid_history(2, "P", root=tmp_path)
    old = next(item for item in history if item["run_id"] == runs[0]["run_id"])

    with pytest.raises(SolidHistoryCleanupError, match="系统未能"):
        move_solid_history_to_trash(old, set(), lambda _path: False)
    assert runs[0]["run_dir"].is_dir()
    moved = []

    def move(directory: str) -> bool:
        moved.append(Path(directory))
        return True

    removed = move_solid_history_to_trash(old, set(), move)

    assert removed["history_id"] == old["_history_id"]
    assert moved == [runs[0]["run_dir"]]
    assert runs[1]["run_dir"].is_dir()


def test_incomplete_runs_are_classified_without_mixing_into_history(tmp_path):
    output = tmp_path / "results" / "solid_joint"
    complete = start_solid_run(output, 2, "P")
    write_solid_summary({
        "backend": "abaqus-6.14", "node_id": 2, "case": "P",
        "run_id": complete["run_id"], "generated_at": complete["generated_at"],
    }, complete["case_dir"], complete["run_dir"])
    failed = start_solid_run(output, 2, "P")
    (failed["run_dir"] / "failed.odb").write_bytes(b"partial-odb")
    recent = start_solid_run(output, 2, "P")
    (recent["run_dir"] / "building.inp").write_bytes(b"partial-inp")
    current = time.time()

    found = load_solid_orphans(
        2, "P", root=tmp_path,
        now=current + ORPHAN_MIN_AGE_SECONDS + 10)

    assert len(load_solid_history(2, "P", root=tmp_path)) == 1
    assert {item["run_id"] for item in found} == {
        failed["run_id"], recent["run_id"]}
    stages = {item["run_id"]: item["incomplete_stage"] for item in found}
    assert stages[failed["run_id"]] == "后处理或摘要未完成"
    assert stages[recent["run_id"]] == "建模或求解未完成"
    assert all(item["cleanup_eligible"] for item in found)


def test_incomplete_run_cleanup_waits_24_hours_and_rechecks_latest(tmp_path):
    output = tmp_path / "results" / "native_solid_joint"
    complete = start_solid_run(output, 2, "P")
    write_solid_summary({
        "backend": "native-c3d10+gmsh", "node_id": 2, "case": "P",
        "run_id": complete["run_id"], "generated_at": complete["generated_at"],
    }, complete["case_dir"], complete["run_dir"])
    failed = start_solid_run(output, 2, "P")
    (failed["run_dir"] / "partial.vtu").write_bytes(b"partial")
    current = time.time()
    recent = load_solid_orphans(2, "P", root=tmp_path, now=current)[0]

    with pytest.raises(SolidHistoryCleanupError, match="至少再等待"):
        move_solid_orphan_to_trash(recent, lambda _path: True, now=current)
    complete["case_dir"].joinpath("summary.json").write_text(
        json.dumps({"run_id": failed["run_id"]}), encoding="utf-8")
    stale_time = current + ORPHAN_MIN_AGE_SECONDS + 10
    stale = load_solid_orphans(2, "P", root=tmp_path, now=stale_time)[0]
    with pytest.raises(SolidHistoryCleanupError, match="latest"):
        move_solid_orphan_to_trash(stale, lambda _path: True, now=stale_time)

    write_solid_summary({
        "backend": "native-c3d10+gmsh", "node_id": 2, "case": "P",
        "run_id": complete["run_id"], "generated_at": complete["generated_at"],
    }, complete["case_dir"], complete["run_dir"])
    stale = load_solid_orphans(2, "P", root=tmp_path, now=stale_time)[0]
    moved = []

    def move(directory: str) -> bool:
        moved.append(Path(directory))
        return True

    removed = move_solid_orphan_to_trash(stale, move, now=stale_time)

    assert removed["bytes"] >= len(b"partial")
    assert moved == [failed["run_dir"]]
    assert complete["run_dir"].is_dir()


def test_global_records_find_and_clean_a_first_failed_run_without_latest(tmp_path):
    failed = start_solid_run(
        tmp_path / "results" / "native_solid_joint", 17, "FIRST")
    (failed["run_dir"] / "solver.msg").write_bytes(b"stopped")
    complete = start_solid_run(
        tmp_path / "results" / "solid_joint", 2, "P")
    write_solid_summary({
        "backend": "abaqus-6.14", "node_id": 2, "case": "P",
        "run_id": complete["run_id"], "generated_at": complete["generated_at"],
    }, complete["case_dir"], complete["run_dir"])
    stale_time = time.time() + ORPHAN_MIN_AGE_SECONDS + 10

    records = load_all_solid_records(tmp_path, now=stale_time)

    assert records["targets"] == [
        {"node_id": 2, "case": "P"},
        {"node_id": 17, "case": "FIRST"},
    ]
    assert len(records["history"]) == 1
    assert len(records["orphans"]) == 1
    orphan = records["orphans"][0]
    assert orphan["node_id"] == 17
    assert orphan["incomplete_stage"] == "建模或求解未完成"
    assert orphan["cleanup_eligible"] is True
    moved = []
    removed = move_solid_orphan_to_trash(
        orphan, lambda path: moved.append(Path(path)) or True, now=stale_time)
    assert removed["history_id"] == orphan["_history_id"]
    assert moved == [failed["run_dir"]]
