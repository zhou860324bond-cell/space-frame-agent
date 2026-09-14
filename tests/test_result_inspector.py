"""结果探针、极值定位、截面正应力与显示控制。"""

import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyvista")

from PySide6.QtCore import Qt                                     # noqa: E402
from PySide6.QtWidgets import QApplication, QDialogButtonBox       # noqa: E402

from agent import Session                                          # noqa: E402
from desktop import scene                                          # noqa: E402
from desktop.main_window import MainWindow                         # noqa: E402
from desktop.convergence_dialog import SolidConvergenceDialog      # noqa: E402
from desktop.result_inspector import (assess_result_trust, assess_solid_result,
                                      build_solid_history_management,
                                      compare_solid_history,
                                      global_extreme, probe_member, projected_station,
                                      solid_result_provenance,
                                      section_stress_grid)          # noqa: E402
from sections import rectangle                                     # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    yield QApplication.instance() or QApplication([])


def solved() -> Session:
    section = rectangle("B", 0.2, 0.4)
    session = Session()
    assert session.set_model(model={
        "schema_version": 1, "units": "N-m-Pa",
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                  {"id": 2, "x": 6, "y": 0, "z": 0}],
        "members": [{"id": 1, "i": 1, "j": 2,
                     "section": "B", "material": "Steel"}],
        "materials": [{"name": "Steel", "E": 2.06e11, "nu": 0.3}],
        "sections": [section],
        "supports": [{"node": 1, "fix": [1, 1, 1, 1, 1, 1]}],
        "load_cases": [{"name": "P", "nodal_loads": [
            {"name": "Tip", "node": 2,
             "load": [0, 0, -50e3, 0, 0, 0]}]}],
    }).ok
    assert session.solve_model().ok
    return session


def test_probe_projects_to_member_and_returns_six_signed_components_and_stress():
    session = solved()
    data = probe_member(session.frame, session.solution, 1,
                        np.array([2.25, 0.4, -0.2]), "P")
    assert data["x"] == pytest.approx(2.25)
    assert set(data["forces"]) == {"N", "Vy", "Vz", "T", "My", "Mz"}
    assert data["force_unit"] == "kN"
    assert data["moment_unit"] == "kN·m"
    assert data["displacement_unit"] == "mm"
    assert data["stress"] is not None
    assert data["stress"]["min"] < 0 < data["stress"]["max"]


def test_projected_station_clamps_outside_points_to_member_ends():
    session = solved()
    assert projected_station(session.frame, 1, [-3, 0, 0]) == 0.0
    assert projected_station(session.frame, 1, [20, 0, 0]) == 6.0


def test_global_extreme_returns_a_traceable_member_station_and_point():
    session = solved()
    extreme = global_extreme(session.frame, session.solution, "Mz", "P")
    assert extreme["member"] == 1
    assert extreme["x"] == pytest.approx(0.0)
    assert abs(extreme["value"]) == pytest.approx(300.0)
    assert extreme["point"] == pytest.approx([0, 0, 0])


def test_circular_stress_grid_masks_points_outside_the_real_section():
    class Circular:
        name = "Tube"
        A = 0.01
        Iy = Iz = 1e-4
        cy = cz = 0.2
        circular = True

    grid = section_stress_grid(Circular(), 0.0, 1e3, 2e3, samples=21)
    assert np.isnan(grid["sigma"][0, 0])
    assert np.isfinite(grid["sigma"][10, 10])


def test_sign_filter_changes_only_a_copy():
    source = np.array([-2.0, 0.0, 3.0])
    assert scene.sign_filtered(source, "positive") == pytest.approx([0, 0, 3])
    assert scene.sign_filtered(source, "negative") == pytest.approx([-2, 0, 0])
    assert source == pytest.approx([-2, 0, 3])


def test_result_panel_controls_are_wired_to_contour_options(qt_app):
    window = MainWindow(solved())
    assert window.results.range_mode.currentData() is None, "默认应展示真实全量程"
    window.results.range_mode.setCurrentIndex(1)
    window.results.sign_mode.setCurrentIndex(1)
    window.results.overlay_deformed.setChecked(True)
    window.results.levels.setCurrentText("8 级")
    options = window.result_display_options
    assert options == {"percentile": None, "sign": "positive", "levels": 8,
                       "palette": "rainbow", "shading": True,
                       "overlay_deformed": True, "show_extrema": True}


def test_result_panel_only_shows_tools_relevant_to_current_result(qt_app):
    window = MainWindow(solved())
    window.results.show_rows("强度校核", ["杆件", "校核结论"],
                             [[1, "通过"]], context="strength")
    assert window.results.cloud_controls.isHidden()
    assert not window.results.result_actions.isHidden()
    assert not window.results.btn_trust.isHidden()
    assert window.results.btn_extreme.isHidden()
    assert window.results.btn_stress.isHidden()

    window.results.set_context("contour")
    assert not window.results.cloud_controls.isHidden()
    assert not window.results.result_actions.isHidden()
    assert not window.results.btn_extreme.isHidden()


def test_result_panel_summarizes_type_rows_and_engineering_status(qt_app):
    window = MainWindow(solved())
    window.results.show_rows(
        "强度校核，共 2 根杆件", ["杆件", "校核结论"],
        [[1, "通过"], [2, "超限"]], marks=["pass", "fail"],
        context="strength")

    assert window.results.summary_type.text() == "强度校核"
    assert window.results.summary_count.text() == "2 行"
    assert window.results.summary_status.text() == "1 项超限"
    assert window.results.summary_status.property("severity") == "fail"


def test_linear_solution_trust_separates_display_sampling_from_analysis_mesh():
    assessment = assess_result_trust(solved(), display_stations=41)
    assert assessment["label"] == "基础检查通过"
    assert assessment["severity"] == "pass"
    assert assessment["physical_members"] == 1
    assert assessment["analysis_elements"] == 1
    display_row = next(row for row in assessment["rows"]
                       if row[0] == "显示细分")
    assert "不改变刚度矩阵" in display_row[2]


def test_solid_trust_accepts_converged_hotspot_but_not_the_singular_peak():
    assessment = assess_solid_result({
        "backend": "native-c3d10+gmsh", "node_id": 2, "case": "LC1",
        "meshes": [{}, {}, {}],
        "hot_spot_convergence": {
            "3": {"verdict": "converging", "reason": "最后两档变化 2.0%"}},
        "hot_spot_all_converged": True,
        "governing_arm": "3", "stress_concentration_factor": 1.42,
        "peak_convergence": {"verdict": "diverging", "reason": "峰值持续增长"},
    })
    assert assessment["label"] == "Kt 证据已通过"
    assert assessment["severity"] == "pass"
    gate_index = next(i for i, row in enumerate(assessment["rows"])
                      if row[0] == "Kt 发布门禁")
    assert assessment["marks"][gate_index] == "pass"
    assert assessment["marks"][-1] == "unclear"
    assert any(row[0] == "相贯线峰值" and row[1] == "仅作诊断"
               for row in assessment["rows"])


def test_solid_trust_flags_a_kt_without_all_arm_evidence():
    assessment = assess_solid_result({
        "backend": "abaqus-6.14", "node_id": 8, "case": "LC1",
        "meshes": [{}, {}, {}],
        "hot_spot_convergence": {
            "2": {"verdict": "converging", "reason": "已稳定"},
            "3": {"verdict": "inconclusive", "reason": "缺少第三档"}},
        "hot_spot_all_converged": False,
        "governing_arm": "2", "stress_concentration_factor": 1.7,
    })
    assert assessment["severity"] == "fail"
    gate = next(row for row in assessment["rows"] if row[0] == "Kt 发布门禁")
    assert gate[1] == "发布契约异常"


def test_solid_result_page_uses_entity_evidence_instead_of_beam_trust(qt_app):
    window = MainWindow(solved())
    window._last_convergence_payload = {
        "backend": "native-c3d10+gmsh", "node_id": 2, "case": "P",
        "meshes": [{}, {}, {}],
        "hot_spot_convergence": {
            "2": {"verdict": "converging", "reason": "三档变化已稳定"}},
        "hot_spot_all_converged": True,
        "governing_arm": "2", "stress_concentration_factor": 1.35,
        "peak_convergence": {"verdict": "diverging"},
    }
    window.results.set_context("solid_joint")
    window.refresh()

    assert window.results.summary_trust.text() == "Kt 证据已通过"
    window.results.btn_trust.click()
    assert window.results.context == "trust"
    assert any(window.results.table.item(row, 0).text() == "Kt 发布门禁"
               for row in range(window.results.table.rowCount()))


def test_solid_provenance_shows_origin_fingerprints_mesh_and_artifact(
        qt_app, tmp_path):
    artifact = tmp_path / "solid_joint_2.odb"
    artifact.write_bytes(b"odb-evidence")
    payload = {
        "backend": "abaqus-6.14", "node_id": 2, "case": "P",
        "generated_at": "2026-09-14T04:30:00+00:00",
        "_result_origin": "disk", "hot_spot_all_converged": True,
        "cache_identity": {
            "schema": "solid-joint-cache/v1",
            "model_digest": "a" * 64, "joint_input_digest": "b" * 64,
            "mesh_plan_mm": [8.0, 5.6, 4.0],
        },
        "files": {"finest_odb": str(artifact)},
    }

    provenance = solid_result_provenance(payload)

    assert provenance["source"] == "磁盘恢复 · abaqus-6.14"
    assert provenance["fingerprint"] == "模型 aaaaaaaaaa… / 节点 bbbbbbbbbb…"
    assert provenance["mesh"] == "8 / 5.6 / 4 mm"
    assert provenance["artifact"].startswith("ODB 完整")
    assert provenance["status"] == "证据链完整"

    window = MainWindow(solved())
    window.results.set_solid_provenance(provenance)
    window.results.set_context("solid_joint")
    assert not window.results.provenance.isHidden()
    assert window.results.provenance_source.text() == "磁盘恢复 · abaqus-6.14"
    assert window.results.provenance_status.text() == "证据链完整"
    assert window.results.provenance_fingerprint.toolTip().endswith("b" * 64)
    window.results.set_context("strength")
    assert window.results.provenance.isHidden()


def test_solid_history_selector_stays_inline_and_preserves_selection(qt_app):
    window = MainWindow(solved())
    provenance = solid_result_provenance({
        "backend": "abaqus-6.14", "hot_spot_all_converged": False,
    })
    entries = [("run-a", "09-14 10:00 · native"),
               ("run-b", "09-14 11:00 · Abaqus"),
               ("run-c", "09-14 12:00 · Abaqus")]
    selected = []
    window.results.history_compare_requested.connect(
        lambda left, right: selected.append((left, right)))
    window.results.set_solid_provenance(provenance)
    window.results.set_solid_history(entries)
    window.results.set_context("solid_joint")
    window.results.history_left.setCurrentIndex(2)
    window.results.history_right.setCurrentIndex(0)
    window.results.set_solid_history(entries)

    assert not window.results.history_controls.isHidden()
    assert window.results.history_left.currentData() == "run-c"
    assert window.results.history_right.currentData() == "run-a"
    window.results.btn_history_compare.click()
    assert selected == [("run-c", "run-a")]


def test_history_management_reuses_inline_controls_and_protects_a_b(qt_app):
    window = MainWindow(solved())
    entries = [("run-a", "运行 A"), ("run-b", "运行 B"),
               ("run-c", "运行 C"), ("run-d", "运行 D")]
    managed = []
    removed = []
    window.results.history_manage_requested.disconnect(
        window.toggle_solid_history_management)
    window.results.history_manage_requested.connect(lambda: managed.append(True))
    window.results.history_remove_requested.connect(removed.append)
    window.results.set_solid_provenance(solid_result_provenance({
        "backend": "abaqus-6.14", "hot_spot_all_converged": False,
    }))
    window.results.set_solid_history(entries, {"run-b", "run-c", "run-d"})
    window.results.set_context("solid_joint")

    window.results.btn_history_manage.click()
    assert managed == [True]
    window.results.set_context("solid_history_manage")
    assert not window.results.history_left.isHidden()
    assert window.results.history_caption.text() == "保护版本"
    assert not window.results.history_cleanup.isHidden()
    assert window.results.history_cleanup.currentData() == "run-c"

    window.results.set_context("solid_joint")
    window.results.history_right.setCurrentIndex(2)
    window.results.set_context("solid_history_manage")
    assert window.results.history_cleanup.currentData() == "run-b"
    window.results.btn_history_remove.click()
    assert removed == ["run-b"]
    window.results.set_solid_history([("run-a", "运行 A")], set())
    window.results.set_context("solid_joint")
    assert not window.results.history_controls.isHidden()
    assert window.results.history_left.isHidden()
    assert not window.results.btn_history_manage.isHidden()


def test_history_management_reports_sizes_and_keeps_backend_latest(tmp_path):
    import time
    from solid_cache import (ORPHAN_MIN_AGE_SECONDS, load_solid_history,
                             load_solid_orphans, start_solid_run,
                             write_solid_summary)

    output = tmp_path / "results" / "solid_joint"
    runs = [start_solid_run(output, 2, "P") for _ in range(2)]
    for index, run in enumerate(runs):
        artifact = run["run_dir"] / f"result-{index}.odb"
        artifact.write_bytes(b"x" * (1024 * (index + 1)))
        write_solid_summary({
            "backend": "abaqus-6.14", "node_id": 2, "case": "P",
            "run_id": run["run_id"], "generated_at": run["generated_at"],
            "stress_concentration_factor": 1.2 + index / 10,
            "files": {"finest_odb": str(artifact)},
        }, run["case_dir"], run["run_dir"])
    history = load_solid_history(2, "P", root=tmp_path)
    incomplete = start_solid_run(output, 2, "P")
    (incomplete["run_dir"] / "partial.odb").write_bytes(b"partial")
    orphans = load_solid_orphans(
        2, "P", root=tmp_path,
        now=time.time() + ORPHAN_MIN_AGE_SECONDS + 10)

    management = build_solid_history_management(history, set(), orphans)

    assert len(management["rows"]) == 3
    assert len(management["removable_ids"]) == 2
    assert "1 个不完整" in management["summary"]
    assert "Windows 回收站" in management["summary"]
    assert any("后端最新成功结果" in row[-1] for row in management["rows"])


def test_history_comparison_separates_input_change_from_response_difference(tmp_path):
    artifact = tmp_path / "run.odb"
    artifact.write_bytes(b"odb")

    def payload(run_id, joint_hash, kt, hotspot):
        return {
            "run_id": run_id, "generated_at": "2026-09-14T04:30:00+00:00",
            "backend": "abaqus-6.14", "node_id": 2, "case": "P",
            "governing_arm": "3", "stress_concentration_factor": kt,
            "hot_spot_all_converged": True,
            "hot_spot_extrapolation": {"3": {"hot_spot_mpa": hotspot}},
            "cache_identity": {
                "schema": "solid-joint-cache/v1", "model_digest": "a" * 64,
                "joint_input_digest": joint_hash,
                "mesh_plan_mm": [8.0, 5.6, 4.0],
            },
            "files": {"finest_odb": str(artifact)},
        }

    left = payload("a", "b" * 64, 1.20, 120.0)
    right = payload("b", "b" * 64, 1.26, 126.0)
    comparable = compare_solid_history(left, right)
    assert "可审查" in comparable["summary"]
    assert comparable["rows"][4][-1] == "4.76%"
    assert comparable["rows"][5][-1] == "4.76%"

    changed = compare_solid_history(left, payload("c", "c" * 64, 1.26, 126.0))
    assert "不能归因" in changed["summary"]
    input_row = next(row for row in changed["rows"]
                     if row[0] == "节点六分量输入")
    assert input_row[-1] == "不同"


def test_desktop_history_compare_reuses_the_result_table(qt_app, monkeypatch):
    identity = {"schema": "solid-joint-cache/v1",
                "model_digest": "a" * 64,
                "joint_input_digest": "b" * 64,
                "mesh_plan_mm": [8.0, 5.6, 4.0]}
    first = {"backend": "abaqus-6.14", "generated_at": "2026-09-14T01:00:00+00:00",
             "cache_identity": identity, "stress_concentration_factor": 1.2,
             "files": {}}
    second = {**first, "generated_at": "2026-09-14T02:00:00+00:00",
              "stress_concentration_factor": 1.3}
    window = MainWindow(solved())
    window._last_convergence_payload = second
    window._solid_history_payloads = {"first": first, "second": second}
    window.results.set_solid_provenance(solid_result_provenance(second))
    window.results.set_solid_history([("first", "第一次"), ("second", "第二次")])
    window.results.set_context("solid_joint")

    window.results.btn_history_compare.click()

    assert window.results.context == "solid_history"
    assert window.results.table.columnCount() == 4
    assert any(window.results.table.item(row, 0).text() == "正式 Kt"
               for row in range(window.results.table.rowCount()))
    assert not window.results.provenance.isHidden()
    window.results.btn_trust.click()
    assert any(window.results.table.item(row, 0).text() == "Kt 发布门禁"
               for row in range(window.results.table.rowCount()))

    orphan = {
        "_history_id": "orphan:abaqus:20260912T010203_123456Z_deadbeef",
        "_orphan": True, "backend": "abaqus", "node_id": 2, "case": "P",
        "run_id": "20260912T010203_123456Z_deadbeef",
        "generated_at": "2026-09-12T01:02:03+00:00",
        "incomplete_stage": "求解器返回失败", "bytes": 4096, "files": 3,
        "age_seconds": 172800, "latest": False, "cleanup_eligible": True,
    }
    monkeypatch.setattr("solid_cache.load_solid_orphans",
                        lambda *_args, **_kwargs: [orphan])
    window._last_convergence_payload = {"node_id": 2, "case": "P"}
    window.show_solid_history_management()

    assert window.results.context == "solid_history_manage"
    assert any("不完整" in window.results.table.item(row, 1).text()
               for row in range(window.results.table.rowCount()))
    assert window.results.history_cleanup.currentData() == orphan["_history_id"]


def test_global_solid_center_opens_without_a_successful_current_result(
        qt_app, monkeypatch):
    orphan = {
        "_history_id": "orphan:native:20260912T010203_123456Z_deadbeef",
        "_orphan": True, "backend": "native", "node_id": 17, "case": "FIRST",
        "run_id": "20260912T010203_123456Z_deadbeef",
        "generated_at": "2026-09-12T01:02:03+00:00",
        "incomplete_stage": "建模或求解未完成", "bytes": 4096, "files": 3,
        "age_seconds": 172800, "latest": False, "cleanup_eligible": True,
    }
    monkeypatch.setattr("solid_cache.load_all_solid_records", lambda: {
        "history": [], "orphans": [orphan],
        "targets": [{"node_id": 17, "case": "FIRST"}],
    })
    window = MainWindow(solved())
    window._last_convergence_payload = None

    window.actions_by_name["solid_storage"].trigger()

    assert window.results.context == "solid_storage"
    assert window.results.provenance_source.text() == "全局实体存储扫描"
    assert "节点 17 / FIRST" in window.results.table.item(0, 0).text()
    assert not window.results.history_controls.isHidden()
    assert window.results.history_left.isHidden()
    assert window.results.btn_history_manage.isHidden()
    assert window.results.history_cleanup.currentData() == orphan["_history_id"]
    assert window.results.result_actions.isHidden()
    assert window.results.summary_trust.text() == "不适用"


def test_solid_provenance_marks_legacy_or_missing_artifact_as_incomplete():
    got = solid_result_provenance({
        "backend": "native-c3d10+gmsh", "files": {},
        "hot_spot_all_converged": True,
    })
    assert got["fingerprint"] == "旧版结果，无输入指纹"
    assert got["artifact"] == "VTU 缺失或为空"
    assert got["status"] == "溯源信息不完整"
    assert got["severity"] == "unclear"

    corrupt = solid_result_provenance({
        "backend": "abaqus-6.14", "hot_spot_all_converged": True,
        "cache_identity": {"schema": "solid-joint-cache/v1",
                           "model_digest": "short",
                           "joint_input_digest": "short",
                           "mesh_plan_mm": ["broken"]},
        "files": {},
    })
    assert corrupt["mesh"] == "网格计划损坏"
    assert corrupt["status"] == "溯源信息不完整"


def test_result_panel_exposes_trust_details_after_solve(qt_app):
    window = MainWindow(solved())
    assert window.results.summary_trust.text() == "基础检查通过"
    assert window.results.summary_trust.property("severity") == "pass"
    window.results.btn_trust.click()
    assert window.results.context == "trust"
    assert window.results.table.rowCount() >= 5
    assert window.results.table.item(0, 0).text() == "模型合法性"


def test_result_panel_exposes_a_convergence_workspace_without_faking_mesh_data(qt_app):
    window = MainWindow(solved())
    assert not window.results.btn_convergence.isHidden()
    window.results.btn_convergence.click()
    assert window.results.context == "convergence"
    assert window.results.table.item(0, 5).text() == "尚未执行专项细化"
    assert "41 个显示点不参与刚度组装" in window.results.table.item(0, 6).text()


def test_convergence_workspace_compares_saved_native_and_abaqus_joint_runs(qt_app):
    window = MainWindow(solved())

    def payload(backend, stresses, sizes, kt):
        return {
            "backend": backend, "node_id": 2, "case": "P",
            "nominal_normal_mpa": 100.0,
            "stress_concentration_factor": kt,
            "hot_spot_all_converged": True,
            "hot_spot_convergence": {"2": {
                "verdict": "converging", "reason": "三档热点已稳定",
                "values": [{"hotspot_mesh_size_mm": size,
                            "hot_spot_mpa": stress}
                           for size, stress in zip(sizes, stresses)],
            }},
            "peak_convergence": {"verdict": "diverging"},
            "meshes": [{"mesh_size_mm": size,
                        "max_abs_principal_mpa": 500.0 + index}
                       for index, size in enumerate(sizes)],
        }

    native = payload("native-c3d10+gmsh", [100.0, 103.0, 104.0],
                     [16.0, 12.0, 8.0], 1.04)
    abaqus = payload("abaqus-6.14", [104.5, 105.0, 105.2],
                     [8.0, 5.6, 4.0], 1.052)
    window._last_convergence_payload = abaqus
    window._solid_backend_payloads = {
        (2, "P", "native"): native, (2, "P", "abaqus"): abaqus}

    window.show_convergence_workspace()

    first_column = [window.results.table.item(row, 0).text()
                    for row in range(window.results.table.rowCount())]
    assert any("实体同尺度对标" in text for text in first_column)
    assert any(text == "实体 Kt 对标" for text in first_column)
    assert "匹配到 1 个同尺度热点档位" in window.results.caption.text()


def test_solid_task_dialog_defaults_to_three_visible_mesh_levels(qt_app):
    dialog = SolidConvergenceDialog(
        node_id=8, case="LC1", reference_mm=8.0, has_abaqus=False)
    assert dialog.backend() == "native"
    assert dialog.mode() == "convergence"
    assert dialog.mesh_sizes() == [40.0, 30.0, 20.0]
    assert dialog.mesh_grid.rowCount() == 4
    assert "数分钟" in dialog.note.text()

    dialog.mode_box.setCurrentIndex(dialog.mode_box.findData("quick"))
    assert dialog.mesh_sizes() == [20.0]
    assert "不发布正式 Kt" in dialog.note.text()


def test_solid_task_dialog_exposes_two_explicit_plans_for_comparison(qt_app):
    dialog = SolidConvergenceDialog(
        node_id=8, case="LC1", reference_mm=8.0, has_abaqus=True)
    dialog.backend_box.setCurrentIndex(
        dialog.backend_box.findData("compare"))

    assert dialog.backend() == "compare"
    assert dialog.mesh_plans() == {
        "native": [40.0, 30.0, 20.0],
        "abaqus": [8.0, 5.6, 4.0],
    }
    assert dialog.mesh_grid.rowCount() == 7
    assert "6 个网格作业" in dialog.note.text()
    assert dialog.buttons.button(
        QDialogButtonBox.StandardButton.Ok).text() == "开始双后端对标"


def test_desktop_pair_task_runs_both_backends_and_opens_comparison(qt_app, monkeypatch):
    window = MainWindow(solved())
    calls = []

    def analyze(*, backend, mesh_sizes_mm, **_kwargs):
        from agent import ToolResult
        calls.append((backend, mesh_sizes_mm))
        stresses = ([100.0, 103.0, 104.0] if backend == "native" else
                    [104.5, 105.0, 105.2])
        sizes = ([16.0, 12.0, 8.0] if backend == "native" else
                 [8.0, 5.6, 4.0])
        payload = {
            "backend": ("native-c3d10+gmsh" if backend == "native" else
                        "abaqus-6.14"),
            "node_id": 2, "case": "P", "nominal_normal_mpa": 100.0,
            "hot_spot_all_converged": True,
            "stress_concentration_factor": 1.04 if backend == "native" else 1.052,
            "hot_spot_convergence": {"2": {
                "verdict": "converging", "reason": "三档已稳定",
                "values": [{"hotspot_mesh_size_mm": size,
                            "hot_spot_mpa": stress}
                           for size, stress in zip(sizes, stresses)],
            }},
            "peak_convergence": {"verdict": "diverging"},
            "meshes": [{"mesh_size_mm": size,
                        "max_abs_principal_mpa": 500.0 + index}
                       for index, size in enumerate(sizes)],
        }
        return ToolResult(True, payload)

    monkeypatch.setattr(window.session, "analyze_joint_solid", analyze)
    spec = SimpleNamespace(node_id=2, case="P", anchor_member=1)
    plans = {"native": [40.0, 30.0, 20.0], "abaqus": [8.0, 5.6, 4.0]}

    window._run_solid_backend_pair(spec, plans)
    assert window.runner.wait(5000)

    assert calls == [("native", plans["native"]), ("abaqus", plans["abaqus"])]
    assert set(key[2] for key in window._solid_backend_payloads) == {
        "native", "abaqus"}
    assert window.results.context == "convergence"
    assert any(window.results.table.item(row, 0).text() == "实体 Kt 对标"
               for row in range(window.results.table.rowCount()))


def test_desktop_pair_task_restores_valid_disk_stage_after_restart(qt_app, monkeypatch):
    from solid_cache import make_solid_cache_identity
    import solid_cache

    window = MainWindow(solved())
    spec = SimpleNamespace(node_id=2, case="P", anchor_member=1)
    plans = {"native": [40.0, 30.0, 20.0], "abaqus": [8.0, 5.6, 4.0]}
    restored = {
        "backend": "native-c3d10+gmsh", "node_id": 2, "case": "P",
        "schema": "solid-joint-analysis/native-v1",
        "hot_spot_all_converged": True,
        "meshes": [{"global_mesh_size_mm": size}
                   for size in plans["native"]],
        "cache_identity": make_solid_cache_identity(
            window.session.model, spec, plans["native"]),
    }
    monkeypatch.setattr(
        solid_cache, "load_solid_cache",
        lambda *_args, **_kwargs: {"usable": {"native": restored}, "ignored": {}})
    calls = []

    def analyze(*, backend, mesh_sizes_mm, **_kwargs):
        from agent import ToolResult
        calls.append((backend, mesh_sizes_mm))
        return ToolResult(True, {
            "backend": "abaqus-6.14", "node_id": 2, "case": "P",
            "hot_spot_all_converged": True, "hot_spot_convergence": {},
            "peak_convergence": {}, "meshes": [],
        })

    monkeypatch.setattr(window.session, "analyze_joint_solid", analyze)
    window._run_solid_backend_pair(spec, plans)
    assert window.runner.wait(5000)

    assert calls == [("abaqus", plans["abaqus"])]
    assert set(key[2] for key in window._solid_backend_payloads) == {
        "native", "abaqus"}
    assert not window.results.provenance.isHidden()
    assert window.results.provenance_source.text().startswith("本次计算 · abaqus")


def test_contour_controls_state_that_display_detail_is_not_analysis_mesh(qt_app):
    window = MainWindow(solved())
    window.results.set_context("contour")
    assert "只影响曲线和色带" in window.results.display_scope.text()
    window.results.btn_analysis_mesh.click()
    assert window.mode == "分析网格"


def test_result_table_formats_numbers_without_losing_numeric_sort(qt_app):
    window = MainWindow(solved())
    window.results.show_rows("数值显示", ["杆件", "值"],
                             [[10, 1200.0], [2, 0.0000123]])
    assert window.results.table.item(0, 1).text() == "1,200"
    assert window.results.table.item(1, 1).text() == "1.230e-05"
    window.results.table.sortItems(0, Qt.SortOrder.AscendingOrder)
    assert [window.results.table.item(r, 0).text() for r in range(2)] == ["2", "10"]


def test_member_probe_populates_result_table_and_extreme_can_be_located(qt_app):
    session = solved()
    window = MainWindow(session)
    from agent import ToolResult
    window.result = ToolResult(True, {"cases": ["P"]})
    window.case = "P"
    window.probe_member_result(1, np.array([3.0, 0.0, 0.0]))
    assert window.results.table.rowCount() >= 12
    assert window.results.table.columnCount() == 5
    assert window.results.table.item(0, 0).text() == "截面内力"
    assert "轴力 N" in window.results.table.item(0, 1).text()
    assert "U(global)" not in " ".join(
        window.results.table.item(r, 1).text()
        for r in range(window.results.table.rowCount()))
    assert "结果探针" in window.results.caption.text()
    window.locate_current_extreme()
    assert window.viewport.selection == ("member", 1)
    assert "绝对极值" in window.results.caption.text()


def test_bc_edit_invalidates_backend_and_desktop_result_together(qt_app):
    session = solved()
    window = MainWindow(session)
    from agent import ToolResult
    window.result = ToolResult(True, {"cases": ["P"]})
    window.case = "P"
    window.bc.set_selection("node", 2)
    window.bc.spn_fz.setValue(-40e3)
    window.bc._apply_nodal_load()
    assert session.solution is None and session.result_db is None
    assert window.result is None and window.case is None
    assert window.mode == "模型"
    assert "失效" in window.results.caption.text()


def test_load_can_be_copied_and_support_has_an_explicit_delete(qt_app, monkeypatch):
    from PySide6.QtWidgets import QInputDialog

    session = solved()
    window = MainWindow(session)
    window.bc.set_selection("node", 2)
    monkeypatch.setattr(QInputDialog, "getText",
                        lambda *args, **kwargs: ("Tip-Copy", True))
    window.bc._copy_nodal_load()
    case = session.model["load_cases"][0]
    assert {entry["name"] for entry in case["nodal_loads"]} == {
        "Tip", "Tip-Copy"}

    window.bc.set_selection("node", 1)
    window.bc._clear_support()
    assert all(int(item["node"]) != 1 for item in session.model["supports"])


def test_the_stress_field_is_the_worse_extreme_fibre_and_keeps_its_sign(qt_app):
    """σ 取上下缘里绝对值大的那一侧，**并保留符号**——受拉为正、受压为负，
    正好落在发散色标的两端。取 |σ| 会把受压画成受拉那一头。"""
    from internal_forces import member_diagram

    session = solved()
    frame, sol = session.frame, session.solution
    mid = sorted(frame.members)[0]
    diagram = member_diagram(frame, sol, mid, sol.primary, stations=21)
    values = scene.member_scalar(frame, frame.members[mid], diagram, scene.STRESS)

    from stress import extreme_normal_stress
    low, high = extreme_normal_stress(frame.sections[frame.members[mid].section],
                                      diagram.N, diagram.My, diagram.Mz)
    for k in range(len(values)):
        assert values[k] in (low[k], high[k])
        assert abs(values[k]) == pytest.approx(max(abs(low[k]), abs(high[k])))


def test_a_section_without_fibre_distances_refuses_the_stress_contour(qt_app):
    """缺 cy/cz 时**明确拒绝**，不估算一个来路不明的应力。"""
    from frame3d import Section
    from internal_forces import member_diagram
    from stress import StressUnavailable

    session = solved()
    frame, sol = session.frame, session.solution
    mid = sorted(frame.members)[0]
    name = frame.members[mid].section
    old = frame.sections[name]
    frame.sections[name] = Section(name, A=old.A, Iy=old.Iy, Iz=old.Iz, J=old.J)
    diagram = member_diagram(frame, sol, mid, sol.primary, stations=5)
    with pytest.raises(StressUnavailable):
        scene.member_scalar(frame, frame.members[mid], diagram, scene.STRESS)


def test_the_palette_and_shading_controls_reach_the_viewport(qt_app):
    """色系和"立体"是显示选项，必须一路走到视口，不能停在面板上。"""
    window = MainWindow(solved())
    window.results.palette.setCurrentIndex(
        window.results.palette.findData("diverging"))
    window.results.shading.setChecked(False)
    window.set_mode("云图")
    assert window.viewport.contour_palette == "diverging"
    assert window.viewport.contour_shading is False
