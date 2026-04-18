"""Unit tests — Panel construction, overrides, fork, freeze, propagation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from simtool.connector.ir import ParameterBinding
from simtool.panels import (
    MeasurementCapability,
    OverrideSource,
    Panel,
    ParameterOverride,
    PropagationOutcome,
    PublicationState,
    RunHistoryEntry,
    UserConstraints,
    propagate_to_version,
)

from tests._metamodel_fixtures import make_small_panel_ir


def _constraints(**overrides) -> UserConstraints:
    base = dict(
        predictive_priorities=["biofilm_thickness"],
        measurement_capabilities=[
            MeasurementCapability(observable_id="thickness", sampling_rate_hz=1 / 3600),
        ],
        time_horizon_s=7200.0,
        required_phenomena=["growth"],
    )
    base.update(overrides)
    return UserConstraints(**base)


def _panel(**overrides) -> Panel:
    base = dict(
        id="panel-1",
        title="test panel",
        user_id="alice",
        meta_model_id="nitrifying_biofilm",
        meta_model_version_pin="1.2.0",
        derived_ir=make_small_panel_ir(),
        constraints=_constraints(),
    )
    base.update(overrides)
    return Panel(**base)


def _ov(
    pid: str = "mu_max",
    *,
    source: OverrideSource = OverrideSource.USER_PROVIDED,
    justification: str = "user set from unpublished data",
    value: float = 1.2,
) -> ParameterOverride:
    return ParameterOverride(
        parameter_id=pid,
        context_keys={"species": "AOB"},
        override_binding=ParameterBinding(
            parameter_id=pid, canonical_unit="1/day",
            context_keys={"species": "AOB"},
            point_estimate=value,
        ),
        source=source,
        justification=justification,
    )


# --- Construction ---------------------------------------------------------


def test_minimal_panel_constructs() -> None:
    p = _panel()
    assert p.publication_state == PublicationState.DRAFT
    assert p.parameter_overrides == []


def test_time_horizon_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        _constraints(time_horizon_s=0.0)


# --- Overrides ------------------------------------------------------------


def test_set_override_adds() -> None:
    p = _panel()
    p.set_override(_ov())
    assert len(p.parameter_overrides) == 1
    assert p.find_override("mu_max", {"species": "AOB"}).override_binding.point_estimate == pytest.approx(1.2)


def test_set_override_replaces_same_key() -> None:
    p = _panel()
    p.set_override(_ov(value=1.2))
    p.set_override(_ov(value=1.5))
    assert len(p.parameter_overrides) == 1
    assert p.parameter_overrides[0].override_binding.point_estimate == pytest.approx(1.5)


def test_override_requires_justification() -> None:
    with pytest.raises(ValidationError, match="justification"):
        ParameterOverride(
            parameter_id="mu_max",
            override_binding=ParameterBinding(
                parameter_id="mu_max", canonical_unit="1/day", point_estimate=1.0,
            ),
            source=OverrideSource.USER_PROVIDED,
            justification="   ",
        )


def test_duplicate_overrides_on_same_key_rejected_at_construction() -> None:
    """Direct construction with two same-key overrides is a bug."""
    ov = _ov()
    with pytest.raises(ValidationError, match="duplicate parameter overrides"):
        _panel(parameter_overrides=[ov, ov])


def test_find_override_miss_returns_none() -> None:
    p = _panel()
    assert p.find_override("nothing_here") is None


# --- Run history ----------------------------------------------------------


def test_attach_run_adds_entry() -> None:
    p = _panel()
    entry = RunHistoryEntry(
        run_id="run-001", ir_id="x", framework="mock",
        framework_version="0.1", run_record_path="runs/run-001/metadata/run.json",
        status="succeeded",
    )
    p.attach_run(entry)
    assert len(p.run_history) == 1


def test_attach_run_duplicate_rejected() -> None:
    p = _panel()
    entry = RunHistoryEntry(
        run_id="run-001", ir_id="x", framework="m", framework_version="0.1",
        run_record_path="x", status="ok",
    )
    p.attach_run(entry)
    with pytest.raises(ValueError, match="duplicate run_id"):
        p.attach_run(entry)


# --- Fork -----------------------------------------------------------------


def test_fork_produces_independent_copy() -> None:
    p = _panel()
    p.set_override(_ov())
    p.attach_run(RunHistoryEntry(
        run_id="r1", ir_id="x", framework="m",
        framework_version="0.1", run_record_path="x", status="ok",
    ))
    fork = p.fork("panel-2", "bob")
    assert fork.id == "panel-2"
    assert fork.user_id == "bob"
    assert fork.forked_from_panel_id == "panel-1"
    # Overrides copied.
    assert len(fork.parameter_overrides) == 1
    # Run history is NOT carried over — fork starts fresh.
    assert fork.run_history == []
    # Collaborators reset.
    assert fork.collaborators == []
    # Publication state reset.
    assert fork.publication_state == PublicationState.DRAFT


def test_fork_is_deep_copy() -> None:
    p = _panel()
    p.set_override(_ov())
    fork = p.fork("panel-2", "bob")
    fork.parameter_overrides[0].override_binding = ParameterBinding(
        parameter_id="mu_max", canonical_unit="1/day",
        context_keys={"species": "AOB"},
        point_estimate=99.0,
    )
    # Original unchanged.
    assert p.parameter_overrides[0].override_binding.point_estimate == pytest.approx(1.2)


# --- Freeze / unfreeze ----------------------------------------------------


def test_freeze_locks_state_and_version() -> None:
    p = _panel()
    p.freeze()
    assert p.publication_state == PublicationState.FROZEN
    assert p.frozen_at_meta_model_version == "1.2.0"
    assert p.frozen_at is not None


def test_frozen_panel_blocks_mutations() -> None:
    p = _panel()
    p.freeze()
    with pytest.raises(RuntimeError, match="FROZEN"):
        p.set_override(_ov())


def test_frozen_panel_still_accepts_run_history() -> None:
    """Freezing locks CONFIGURATION, not observations — a reproducible
    panel can still record runs of itself."""
    p = _panel()
    p.freeze()
    p.attach_run(RunHistoryEntry(
        run_id="r", ir_id="x", framework="m", framework_version="0.1",
        run_record_path="x", status="ok",
    ))
    assert len(p.run_history) == 1


def test_unfreeze_clears_frozen_metadata() -> None:
    p = _panel()
    p.freeze()
    p.unfreeze()
    assert p.publication_state == PublicationState.DRAFT
    assert p.frozen_at is None
    assert p.frozen_at_meta_model_version is None


def test_unfreeze_noop_on_non_frozen() -> None:
    p = _panel()
    p.unfreeze()  # should not raise or change state
    assert p.publication_state == PublicationState.DRAFT


# --- Sharing --------------------------------------------------------------


def test_share_adds_collaborator_and_promotes_state() -> None:
    p = _panel()
    p.share_with("bob")
    assert "bob" in p.collaborators
    assert p.publication_state == PublicationState.SHARED


def test_share_with_self_is_noop() -> None:
    p = _panel()
    p.share_with("alice")  # user_id is alice
    assert p.collaborators == []


def test_share_same_user_twice_is_noop() -> None:
    p = _panel()
    p.share_with("bob")
    p.share_with("bob")
    assert p.collaborators == ["bob"]


# --- Propagation ----------------------------------------------------------


def test_auto_propagation_on_minor_bump() -> None:
    p = _panel()
    outcome = propagate_to_version(
        p, "1.3.0", policy_requires_confirmation=False,
    )
    assert outcome.applied
    assert outcome.kind == "auto_propagated"
    assert p.meta_model_version_pin == "1.3.0"


def test_major_bump_requires_confirmation() -> None:
    p = _panel()
    outcome = propagate_to_version(
        p, "2.0.0", policy_requires_confirmation=True,
    )
    assert not outcome.applied
    assert outcome.kind == "awaits_confirmation"
    assert p.meta_model_version_pin == "1.2.0"  # unchanged


def test_major_bump_applies_after_confirmation() -> None:
    p = _panel()
    outcome = propagate_to_version(
        p, "2.0.0",
        policy_requires_confirmation=True,
        user_confirmed=True,
    )
    assert outcome.applied
    assert p.meta_model_version_pin == "2.0.0"


def test_frozen_panel_propagation_blocked() -> None:
    p = _panel()
    p.freeze()
    outcome = propagate_to_version(
        p, "1.3.0", policy_requires_confirmation=False,
    )
    assert not outcome.applied
    assert outcome.kind == "blocked_frozen"
    assert p.meta_model_version_pin == "1.2.0"


# --- Round-trip -----------------------------------------------------------


def test_panel_round_trip() -> None:
    p = _panel()
    p.set_override(_ov())
    p.share_with("bob")
    restored = Panel.model_validate_json(p.model_dump_json())
    assert restored.id == p.id
    assert restored.publication_state == PublicationState.SHARED
    assert "bob" in restored.collaborators
    assert len(restored.parameter_overrides) == 1
