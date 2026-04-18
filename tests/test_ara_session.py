"""Unit tests — AraSession end-to-end flow using MockPlugin.

Verifies the session behaves like a stateful chat-backed orchestrator:
open meta-model → create panel → recommend/adjust/fit → run → persist.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from simtool.ara import AraSession
from simtool.panels import MeasurementCapability

from tests._metamodel_fixtures import (
    make_nitrifying_biofilm_metamodel,
    make_small_panel_ir,
)
from tests.test_connector_plugin import MockPlugin


def _seeded_session(tmp_path: Path) -> AraSession:
    session = AraSession(root=tmp_path, plugin=MockPlugin())
    mm = make_nitrifying_biofilm_metamodel()
    session.store.save_metamodel(mm)
    return session


def test_open_metamodel_returns_summary(tmp_path):
    s = _seeded_session(tmp_path)
    reply = s.open_metamodel("nitrifying_biofilm")
    assert "nitrifying_biofilm" in reply
    assert "1.2.0" in reply


def test_open_metamodel_missing_raises(tmp_path):
    session = AraSession(root=tmp_path, plugin=MockPlugin())
    with pytest.raises(FileNotFoundError):
        session.open_metamodel("does_not_exist")


def test_create_panel_after_meta_model_open(tmp_path):
    s = _seeded_session(tmp_path)
    s.open_metamodel("nitrifying_biofilm")
    reply = s.create_panel(
        user_id="alice",
        title="my biofilm",
        derived_ir=make_small_panel_ir(),
        required_phenomena=["growth"],
        observable_ids=["thickness"],
    )
    assert "my biofilm" in reply
    assert s.active_panel is not None


def test_workflows_require_meta_model_and_panel(tmp_path):
    session = AraSession(root=tmp_path, plugin=MockPlugin())
    with pytest.raises(RuntimeError, match="no active meta-model"):
        session.recommend()


def test_scope_status_surfaces_missing_parameters(tmp_path):
    s = _seeded_session(tmp_path)
    s.open_metamodel("nitrifying_biofilm")
    s.create_panel(
        user_id="alice", title="t",
        derived_ir=make_small_panel_ir(),
        required_phenomena=["growth"],
        observable_ids=["thickness"],
    )
    reply = s.scope_status()
    assert "BLOCKED" in reply or "Verdict" in reply


def test_recommend_returns_submodel(tmp_path):
    s = _seeded_session(tmp_path)
    s.open_metamodel("nitrifying_biofilm")
    s.create_panel(
        user_id="alice", title="t",
        derived_ir=make_small_panel_ir(),
        required_phenomena=["growth"],
    )
    reply = s.recommend()
    # Simplest submodel for "growth" only is monod_chemostat_ode.
    assert "monod_chemostat_ode" in reply


def test_adjust_flags_speculative(tmp_path):
    s = _seeded_session(tmp_path)
    s.open_metamodel("nitrifying_biofilm")
    s.create_panel(
        user_id="alice", title="t",
        derived_ir=make_small_panel_ir(),
        required_phenomena=["growth"],
    )
    reply = s.adjust(kind="change_parameter", target_id="phlogiston")
    assert "SPECULATIVE" in reply


def test_run_end_to_end_with_mock_plugin(tmp_path):
    s = _seeded_session(tmp_path)
    s.open_metamodel("nitrifying_biofilm")
    s.create_panel(
        user_id="alice", title="t",
        derived_ir=make_small_panel_ir(),
        required_phenomena=["growth"],
    )
    reply, summary = s.run(auto_approve_assumptions=True)
    assert "succeeded" in reply
    assert summary is not None
    # Panel was persisted (run attached).
    restored = s.store.load_panel(s.active_panel.id)
    assert len(restored.run_history) == 1


def test_freeze_and_unfreeze_round_trip(tmp_path):
    s = _seeded_session(tmp_path)
    s.open_metamodel("nitrifying_biofilm")
    s.create_panel(
        user_id="alice", title="t",
        derived_ir=make_small_panel_ir(),
        required_phenomena=["growth"],
    )
    reply = s.freeze_active_panel()
    assert "FROZEN" in reply
    reply = s.unfreeze_active_panel()
    assert "DRAFT" in reply
