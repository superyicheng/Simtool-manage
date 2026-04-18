"""Canonical workflow 2: the ADJUSTMENT user story.

User narrative:

    "My panel models AOB+NOB biofilm. I want to add a maintenance
    process for AOB (substrate consumption for non-growth). Is the
    meta-model going to back me up on this?"

What the system must deliver:
  1. Trace the adjustment request against the meta-model.
  2. Classify the result as SUPPORTED, PARTIALLY_SUPPORTED, or
     SPECULATIVE — with the names of the meta-model artifacts it
     depends on and which are missing.
  3. For SPECULATIVE adjustments, the system must NOT present values
     as authoritative; anything outside the meta-model is clearly
     flagged.
  4. A supported adjustment carries no speculative-default flag; a
     partially-supported one leaves gaps for the user to fill.
"""

from __future__ import annotations

from simtool.panels import (
    AdjustmentRequest,
    MeasurementCapability,
    OverrideSource,
    Panel,
    ParameterOverride,
    SupportLevel,
    UserConstraints,
    adjust_model,
)
from simtool.connector.ir import ParameterBinding

from tests._metamodel_fixtures import (
    make_nitrifying_biofilm_metamodel,
    make_small_panel_ir,
)


def _panel() -> Panel:
    return Panel(
        id="adjust-panel", title="adjust test", user_id="alice",
        meta_model_id="nitrifying_biofilm", meta_model_version_pin="1.2.0",
        derived_ir=make_small_panel_ir(),
        constraints=UserConstraints(
            predictive_priorities=["thickness"],
            measurement_capabilities=[MeasurementCapability(observable_id="thickness")],
            time_horizon_s=7200.0,
            required_phenomena=["growth"],
        ),
    )


def test_user_changing_known_parameter_is_supported() -> None:
    """User adjusts AOB mu_max — a fully-reconciled parameter. The system
    backs it."""
    m = make_nitrifying_biofilm_metamodel()
    p = _panel()
    req = AdjustmentRequest(
        kind="change_parameter",
        target_id="mu_max",
        spec={"new_value": 1.2},
        user_note="my chemostat run came in lower",
    )
    proposal = adjust_model(p, m, req)
    assert proposal.support_level == SupportLevel.SUPPORTED
    assert "mu_max" in proposal.meta_model_has
    assert proposal.meta_model_missing == []


def test_user_adding_unknown_process_is_speculative() -> None:
    """User asks to add a process with a parameter the meta-model never
    heard of. The system flags the adjustment as speculative; it MUST NOT
    fabricate a value."""
    m = make_nitrifying_biofilm_metamodel()
    p = _panel()
    req = AdjustmentRequest(
        kind="add_process",
        spec={
            "process_kind": "photoinhibition",
            "required_parameter_ids": ["photon_quenching_constant"],
        },
        user_note="light-inhibition term from a recent paper",
    )
    proposal = adjust_model(p, m, req)
    assert proposal.support_level == SupportLevel.SPECULATIVE
    assert "photon_quenching_constant" in proposal.meta_model_missing
    assert "photon_quenching_constant" not in proposal.meta_model_has


def test_user_mixing_known_and_unknown_is_partially_supported() -> None:
    """Half the parameters are in the meta-model, half aren't."""
    m = make_nitrifying_biofilm_metamodel()
    p = _panel()
    req = AdjustmentRequest(
        kind="add_process",
        spec={
            "required_parameter_ids": [
                "mu_max",                        # in meta-model
                "novel_yield_factor",            # not in meta-model
            ],
        },
    )
    proposal = adjust_model(p, m, req)
    assert proposal.support_level == SupportLevel.PARTIALLY_SUPPORTED
    assert "mu_max" in proposal.meta_model_has
    assert "novel_yield_factor" in proposal.meta_model_missing
    # Reasoning names the adjustment type so the UI can render it.
    assert "add_process" in proposal.reasoning


def test_user_gets_submodel_candidates_for_scope_change() -> None:
    """Widening the panel's required phenomena returns a list of submodels
    that can support the new scope — the user picks."""
    m = make_nitrifying_biofilm_metamodel()
    p = _panel()
    req = AdjustmentRequest(
        kind="change_scope",
        spec={"required_phenomena": ["growth", "spatial_gradients"]},
    )
    proposal = adjust_model(p, m, req)
    assert proposal.support_level == SupportLevel.SUPPORTED
    assert set(proposal.depends_on_submodel_ids) >= {
        "continuum_pde_2d", "agent_based_3d",
    }


def test_speculative_adjustment_carries_reasoning_for_transparency() -> None:
    m = make_nitrifying_biofilm_metamodel()
    p = _panel()
    req = AdjustmentRequest(
        kind="change_parameter",
        target_id="phlogiston_affinity",
    )
    proposal = adjust_model(p, m, req)
    assert proposal.support_level == SupportLevel.SPECULATIVE
    assert proposal.reasoning, (
        "speculative adjustments must carry reasoning the user can audit"
    )


def test_user_override_flow_for_speculative_parameter() -> None:
    """A speculative adjustment leaves the meta-model gap explicit. The
    user can still proceed by creating an override — but the override's
    source must record the speculation so the assumption ledger can
    surface it at run time."""
    m = make_nitrifying_biofilm_metamodel()
    p = _panel()

    req = AdjustmentRequest(
        kind="change_parameter", target_id="phlogiston_affinity",
    )
    proposal = adjust_model(p, m, req)
    assert proposal.support_level == SupportLevel.SPECULATIVE

    # User chooses to proceed — creates a speculative override.
    ov = ParameterOverride(
        parameter_id="phlogiston_affinity",
        override_binding=ParameterBinding(
            parameter_id="phlogiston_affinity",
            canonical_unit="dimensionless",
            point_estimate=0.5,
        ),
        source=OverrideSource.AI_DEFAULT_SPECULATIVE,
        justification=(
            "speculative default — meta-model has no consensus; "
            "user should revisit"
        ),
    )
    p.set_override(ov)
    # The override records the speculation source — the UI and the
    # run-time ledger can surface it as speculative rather than
    # authoritative.
    saved = p.find_override("phlogiston_affinity")
    assert saved is not None
    assert saved.source == OverrideSource.AI_DEFAULT_SPECULATIVE
    assert "speculat" in saved.justification.lower()
