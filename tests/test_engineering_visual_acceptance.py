"""四个标准结构的工程语义验收：数值、方向与视图必须同时一致。"""

from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("pyvista", reason="未安装 pyvista，跳过视图语义验收")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "abaqus_bench"))

from models import cantilever_strong_axis, portal_frame, space_frame  # noqa: E402
from desktop import scene                                                # noqa: E402
from frame3d import check_equilibrium, local_axes, solve                 # noqa: E402
from internal_forces import member_diagram                               # noqa: E402
from model_io import from_dict                                           # noqa: E402


def frame_of(factory):
    model = factory()
    return from_dict({key: value for key, value in model.items()
                      if key not in {"name", "note"}})


def test_simple_beam_acceptance_has_signed_load_support_dofs_and_wl2_over_8():
    from agent import Session

    session = Session()
    assert session.set_model(model={
        "schema_version": 1, "units": "N-m-Pa",
        "nodes": [{"id": 1, "x": 0, "y": 0, "z": 0},
                  {"id": 2, "x": 8, "y": 0, "z": 0}],
        "members": [{"id": 1, "i": 1, "j": 2,
                     "section": "B", "material": "Steel"}],
        "materials": [{"name": "Steel", "E": 2.06e11, "nu": 0.3}],
        "sections": [{"name": "B", "A": 0.01, "Iy": 1e-4,
                      "Iz": 2e-4, "J": 1e-5}],
        "supports": [{"node": 1, "fix": [1, 1, 1, 1, 0, 0]},
                     {"node": 2, "fix": [0, 1, 1, 1, 0, 0]}],
        "load_cases": [{"name": "D", "member_loads": [
            {"member": 1, "w": [0, 0, -25e3]}]}],
    }).ok
    assert session.solve_model().ok
    diagram = member_diagram(session.frame, session.solution, 1, "D", stations=101)
    assert np.abs(diagram.Mz).max() == pytest.approx(25e3 * 8**2 / 8)
    assert scene.load_labels(session.frame, "D")[1] == ["wz=-25 kN/m [global]"]
    assert scene.support_labels(session.frame)[1] == [
        "N1 BC: U1 U2 U3 UR1", "N2 BC: U2 U3 UR1"]


def test_cantilever_acceptance_matches_pl_and_global_downward_load():
    frame = frame_of(cantilever_strong_axis)
    solution = solve(frame)
    root = member_diagram(frame, solution, 1, solution.primary, stations=21)
    assert abs(root.Mz[0]) == pytest.approx(50e3 * 6.0)
    assert scene.load_labels(frame, solution.primary)[1] == ["Fz=-50 kN [global]"]
    assert check_equilibrium(frame, solution)["ok"]


def test_portal_acceptance_is_perspective_balanced_and_keeps_load_directions():
    frame = frame_of(portal_frame)
    solution = solve(frame)
    labels = scene.load_labels(frame, solution.primary)[1]
    assert scene.preferred_view(frame) == "isometric"
    assert "Fx=+15 kN [global]" in labels
    assert labels.count("wz=-20 kN/m [global]") == 4
    assert all("U1 U2 U3 UR1 UR2 UR3" in label
               for label in scene.support_labels(frame)[1])
    assert check_equilibrium(frame, solution)["ok"]


def test_space_frame_acceptance_uses_iso_and_solver_local_right_hand_axes():
    frame = frame_of(space_frame)
    solution = solve(frame)
    member_id = sorted(frame.members)[0]
    member = frame.members[member_id]
    pi, pj = scene.member_endpoints(frame, member)
    _, rotation = local_axes(pi, pj, member.ref_vector)
    assert scene.preferred_view(frame) == "isometric"
    assert rotation @ rotation.T == pytest.approx(np.eye(3), abs=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    assert set(scene.member_local_axis_glyphs(frame, member_id)) == {"x", "y", "z"}
    assert "signed in member local axes" in scene.contour_caption("Mz", "kN·m")
    assert check_equilibrium(frame, solution)["ok"]
