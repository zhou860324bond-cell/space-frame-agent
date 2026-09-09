"""最小 Domain IR 版本与旧模型迁移契约。"""

from __future__ import annotations

from copy import deepcopy

import pytest

from agent import Session
from model_io import (CURRENT_SCHEMA_VERSION, from_dict, migrate_payload,
                      validate_payload)


def _legacy_model() -> dict:
    return {
        "units": "N-m-Pa",
        "materials": [{"name": "Steel", "E": 2.0e11, "nu": 0.3}],
        "sections": [{"name": "S", "A": 0.01, "Iy": 1.0e-5,
                      "Iz": 2.0e-5, "J": 5.0e-6}],
        "nodes": [{"id": 10, "x": 0.0, "y": 0.0, "z": 0.0},
                  {"id": 20, "x": 3.0, "y": 0.0, "z": 0.0}],
        "members": [{"id": 30, "i": 10, "j": 20,
                     "section": "S", "material": "Steel"}],
        "supports": [{"node": 10, "fix": [1, 1, 1, 1, 1, 1]}],
    }


def test_legacy_model_remains_valid_and_migration_does_not_mutate_source():
    legacy = _legacy_model()
    before = deepcopy(legacy)

    assert validate_payload(legacy) == []
    migrated = migrate_payload(legacy)

    assert legacy == before
    assert migrated["schema_version"] == CURRENT_SCHEMA_VERSION
    assert validate_payload(migrated) == []


def test_migration_is_idempotent_and_preserves_stable_ids():
    once = migrate_payload(_legacy_model())
    twice = migrate_payload(once)

    assert twice == once
    assert twice is not once
    frame = from_dict(twice)
    assert set(frame.nodes) == {10, 20}
    assert set(frame.members) == {30}


def test_unknown_schema_version_is_rejected_instead_of_guessed():
    future = {**_legacy_model(), "schema_version": CURRENT_SCHEMA_VERSION + 1}

    assert any("schema_version" in error for error in validate_payload(future))
    with pytest.raises(ValueError, match="不支持的模型"):
        migrate_payload(future)


def test_session_upgrades_legacy_model_on_load():
    session = Session()

    result = session.set_model(_legacy_model())

    assert result.ok
    assert result.payload["migrated_from"] == 0
    assert session.model["schema_version"] == CURRENT_SCHEMA_VERSION
    assert session.model["members"][0]["id"] == 30
