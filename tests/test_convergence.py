import pytest

from convergence import build_workspace, solid_backend_rows, solid_mesh_plan


def test_solid_mesh_plans_make_the_cost_and_evidence_level_explicit():
    assert solid_mesh_plan(8.0, backend="native", mode="quick") == [20.0]
    assert solid_mesh_plan(8.0, backend="native") == [40.0, 30.0, 20.0]
    assert solid_mesh_plan(8.0, backend="abaqus") == [8.0, 5.6, 4.0]
    with pytest.raises(ValueError, match="三档"):
        solid_mesh_plan(8.0, backend="abaqus", mode="quick")


def test_b31_needs_three_levels_and_then_uses_response_change():
    two = build_workspace(b31_records=[
        {"case": "portal", "factor": 1, "members": 3, "abaqus_mm": 10.0},
        {"case": "portal", "factor": 2, "members": 6, "abaqus_mm": 10.2},
    ])
    assert two["severity"] == "unclear"
    assert "少于三档" in two["summary"]

    three = build_workspace(b31_records=[
        {"case": "portal", "factor": 1, "members": 3, "abaqus_mm": 10.0},
        {"case": "portal", "factor": 2, "members": 6, "abaqus_mm": 10.2},
        {"case": "portal", "factor": 4, "members": 12, "abaqus_mm": 10.25},
    ])
    assert three["severity"] == "pass"
    assert three["rows"][-1][4] == pytest.approx(0.4878, rel=1e-3)
    assert "B31 离散响应" in three["rows"][-1][-1]


def test_solid_workspace_prefers_hotspot_path_over_singular_peak():
    payload = {
        "peak_convergence": {"verdict": "diverging", "reason": "峰值发散"},
        "hot_spot_convergence": {"7": {
            "verdict": "converging", "reason": "热点稳定",
            "values": [
                {"hotspot_mesh_size_mm": 16.0, "hot_spot_mpa": 100.0},
                {"hotspot_mesh_size_mm": 12.0, "hot_spot_mpa": 103.0},
                {"hotspot_mesh_size_mm": 8.0, "hot_spot_mpa": 104.0},
            ],
        }},
        "meshes": [{"mesh_size_mm": 16.0, "max_abs_principal_mpa": 900.0}],
    }
    got = build_workspace(solid_payload=payload)
    assert got["severity"] == "pass"
    assert all("实体热点" in row[0] for row in got["rows"])
    assert got["rows"][-1][3] == pytest.approx(104.0)
    assert "相贯线节点峰值" in got["rows"][-1][-1]


def test_solid_peak_divergence_is_not_reported_as_converged():
    got = build_workspace(solid_payload={
        "peak_convergence": {"verdict": "diverging", "reason": "奇异特征"},
        "meshes": [
            {"mesh_size_mm": 8.0, "max_abs_principal_mpa": 100.0},
            {"mesh_size_mm": 4.0, "max_abs_principal_mpa": 140.0},
            {"mesh_size_mm": 2.0, "max_abs_principal_mpa": 200.0},
        ],
    })
    assert got["severity"] == "fail"
    assert got["rows"][-1][5] == "峰值疑似奇异发散"
    assert "不得据此给出正式 Kt" in got["rows"][-1][-1]


def _solid_backend(backend, stresses, sizes=(16.0, 12.0, 8.0), kt=1.04):
    return {
        "backend": backend, "node_id": 2, "case": "LC1",
        "nominal_normal_mpa": 100.0,
        "stress_concentration_factor": kt,
        "hot_spot_convergence": {"7": {
            "verdict": "converging",
            "values": [{"hotspot_mesh_size_mm": size,
                        "hot_spot_mpa": stress}
                       for size, stress in zip(sizes, stresses)],
        }},
    }


def test_solid_backend_comparison_matches_physical_hotspot_sizes_not_level_names():
    native = _solid_backend("native-c3d10+gmsh", [100.0, 103.0, 104.0])
    abaqus = _solid_backend(
        "abaqus-6.14", [104.5, 105.0, 105.2], sizes=(8.0, 5.6, 4.0), kt=1.052)

    rows, marks, notes = solid_backend_rows([native, abaqus])

    matched = [row for row in rows if "同尺度" in row[0]]
    assert len(matched) == 1
    assert matched[0][1] == "h=8 / 8 mm"
    final = next(row for row in rows if "收敛解对标" in row[0])
    assert final[4] == pytest.approx(100.0 * (105.2 - 104.0) / 105.2)
    assert marks[rows.index(final)] == "pass"
    assert "匹配到 1 个同尺度" in notes[0]


def test_solid_backend_comparison_refuses_different_nominal_stress_bases():
    native = _solid_backend("native-c3d10+gmsh", [100.0, 103.0, 104.0])
    abaqus = _solid_backend("abaqus-6.14", [100.0, 103.0, 104.0])
    abaqus["nominal_normal_mpa"] = 110.0

    rows, marks, _notes = solid_backend_rows([native, abaqus])

    assert marks == ["fail"]
    assert rows[0][5] == "输入基准不一致"


def test_nonlinear_history_is_called_increment_convergence_not_mesh_convergence():
    got = build_workspace(analysis={
        "type": "pdelta", "tolerance": 1e-6,
        "convergence": {"P": [
            {"increment": 1, "load_factor": 0.5, "iterations": 3,
             "relative_change": 2e-8},
            {"increment": 2, "load_factor": 1.0, "iterations": 4,
             "relative_change": 5e-8},
        ]},
    }, case="P")
    assert got["severity"] == "pass"
    assert got["rows"][0][0] == "非线性增量 · P"
    assert "不代表空间网格无关" in got["rows"][0][-1]
    assert got["rows"][0][4] == "不同荷载级，不横向比较"


def test_linear_result_refuses_to_treat_display_points_as_mesh_levels():
    got = build_workspace()
    assert got["severity"] == "unclear"
    assert got["rows"][0][5] == "尚未执行专项细化"
    assert "41 个显示点不参与刚度组装" in got["rows"][0][-1]
