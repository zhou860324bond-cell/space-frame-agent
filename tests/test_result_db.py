"""统一 Result DB 的字段语义、序列化和 Session 生命周期。"""

from __future__ import annotations

import json

import numpy as np

from agent import Session
from result_db import RESULT_DB_SCHEMA_VERSION


def _solved_cantilever() -> Session:
    model = {
        "units": "N-m-Pa",
        "materials": [{"name": "Steel", "E": 2.0e11, "nu": 0.3}],
        "sections": [{"name": "S", "A": 0.01, "Iy": 1.0e-5,
                      "Iz": 2.0e-5, "J": 5.0e-6}],
        "nodes": [{"id": 1, "x": 0.0, "y": 0.0, "z": 0.0},
                  {"id": 2, "x": 3.0, "y": 0.0, "z": 0.0}],
        "members": [{"id": 7, "i": 1, "j": 2,
                     "section": "S", "material": "Steel"}],
        "supports": [{"node": 1, "fix": [1, 1, 1, 1, 1, 1]}],
        "nodal_loads": [{"node": 2, "load": [0, 0, -1000, 0, 0, 0]}],
    }
    session = Session()
    assert session.set_model(model).ok
    assert session.solve_model().ok
    return session


def test_result_db_has_versioned_hierarchy_and_field_semantics():
    session = _solved_cantilever()
    db = session.result_db

    assert db.schema_version == RESULT_DB_SCHEMA_VERSION
    assert db.metadata["status"] == "COMPLETED"
    assert len(db.metadata["model_hash"]) == 64
    assert set(db.steps) == {"default"}
    assert db.field("default", "U").location == "NODE"
    assert db.field("default", "U").coordinate_system == "GLOBAL"
    assert db.field("default", "Mz").location == "ELEMENT_END"
    assert db.field("default", "Mz").coordinate_system == "LOCAL"
    assert db.field("default", "Mz").averaging == "none"


def test_result_db_is_json_serializable():
    document = _solved_cantilever().result_db.to_dict()

    encoded = json.dumps(document, ensure_ascii=False)

    assert '"schema_version": 1' in encoded
    assert document["steps"]["default"]["frames"][0]["fields"]["U"]["unit"] == "m"


def test_public_queries_and_plot_read_result_db_not_mutated_solver_arrays(
        tmp_path, monkeypatch):
    session = _solved_cantilever()
    expected_displacement = session.query_results("max_displacement").payload
    expected_reactions = session.query_results("reactions").payload
    expected_forces = session.query_results("member_forces").payload

    result = session.solution["default"]
    result.U[:] = np.nan
    result.R[:] = np.nan
    result.member_forces[7][:] = np.nan

    assert session.query_results("max_displacement").payload == expected_displacement
    assert session.query_results("reactions").payload == expected_reactions
    assert session.query_results("member_forces").payload == expected_forces
    monkeypatch.chdir(tmp_path)
    plotted = session.plot_results("deformed")
    assert plotted.ok, plotted.payload
    assert np.isfinite(plotted.payload["max_displacement_mm"])


def test_model_edit_invalidates_result_db():
    session = _solved_cantilever()

    assert session.edit_node(2, x=4.0).ok

    assert session.result_db is None
    assert not session.query_results("reactions").ok
