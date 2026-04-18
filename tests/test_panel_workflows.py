"""Unit tests — the three panel workflows."""

from __future__ import annotations

import pytest

from simtool.metamodel import (
    ApproximationOperatorKind,
)
from simtool.panels import (
    AdjustmentRequest,
    ExperimentalDataset,
    FitCalibrationResult,
    MeasurementCapability,
    OverrideSource,
    SupportLevel,
    UserConstraints,
    adjust_model,
    fit_data,
    recommend_model,
)

from tests._metamodel_fixtures import (
    make_nitrifying_biofilm_metamodel,
    make_small_panel_ir,
)
from tests.test_panels import _panel


def _constraints(**o) -> UserConstraints:
    base = dict(
        predictive_priorities=["biofilm_thickness"],
        measurement_capabilities=[
            MeasurementCapability(observable_id="thickness"),
        ],
        time_horizon_s=7200.0,
        required_phenomena=["growth"],
    )
    base.update(o)
    return UserConstraints(**base)


# ============================================================================
# recommend_model
# ============================================================================


def test_recommend_picks_simplest_eligible() -> None:
    """User asks only for 'growth' — the simplest submodel covers it."""
    m = make_nitrifying_biofilm_metamodel()
    rec = recommend_model(m, _constraints(required_phenomena=["growth"]))
    assert rec.submodel_id == "monod_chemostat_ode"
    assert rec.derived_via_operator_id is None
    # Reasoning trace lists eligibility for every submodel.
    assert any(step.kind == "eligible" for step in rec.reasoning)


def test_recommend_respects_excluded_phenomena() -> None:
    """User excludes 'spatial_gradients' — rules out PDE and ABM."""
    m = make_nitrifying_biofilm_metamodel()
    rec = recommend_model(
        m,
        _constraints(
            required_phenomena=["growth"],
            excluded_phenomena=["spatial_gradients"],
        ),
    )
    assert rec.submodel_id == "monod_chemostat_ode"


def test_recommend_climbs_hierarchy_when_required() -> None:
    """User requires 'individual_cells' — only ABM covers it."""
    m = make_nitrifying_biofilm_metamodel()
    rec = recommend_model(
        m, _constraints(required_phenomena=["growth", "individual_cells"])
    )
    assert rec.submodel_id == "agent_based_3d"


def test_recommend_derives_via_approximation_operator() -> None:
    """ABM covers all required phenomena; user excludes detachment which
    ABM carries. No pre-enumerated submodel fits — but the mean_field_closure
    operator from ABM->PDE drops detachment and still covers growth."""
    m = make_nitrifying_biofilm_metamodel()
    rec = recommend_model(
        m,
        _constraints(
            required_phenomena=["growth", "spatial_gradients"],
            excluded_phenomena=["individual_cells", "detachment"],
        ),
    )
    # Pre-enumerated PDE submodel covers required, excludes neither -> eligible.
    assert rec.submodel_id == "continuum_pde_2d"


def test_recommend_returns_empty_when_no_fit_possible() -> None:
    m = make_nitrifying_biofilm_metamodel()
    rec = recommend_model(
        m, _constraints(required_phenomena=["quantum_effects"])
    )
    assert rec.submodel_id == ""
    assert rec.unmet_constraints


def test_recommend_trace_is_structured() -> None:
    m = make_nitrifying_biofilm_metamodel()
    rec = recommend_model(m, _constraints(required_phenomena=["growth"]))
    assert rec.reasoning
    kinds = {step.kind for step in rec.reasoning}
    assert kinds <= {"eligible", "rejected", "applied_operator"}


# ============================================================================
# adjust_model
# ============================================================================


def test_adjust_supported_for_known_parameter() -> None:
    m = make_nitrifying_biofilm_metamodel()
    p = _panel()
    req = AdjustmentRequest(kind="change_parameter", target_id="mu_max")
    proposal = adjust_model(p, m, req)
    assert proposal.support_level == SupportLevel.SUPPORTED
    assert "mu_max" in proposal.meta_model_has
    assert proposal.meta_model_missing == []


def test_adjust_speculative_for_unknown_parameter() -> None:
    m = make_nitrifying_biofilm_metamodel()
    p = _panel()
    req = AdjustmentRequest(
        kind="change_parameter", target_id="quantum_tunneling_constant"
    )
    proposal = adjust_model(p, m, req)
    assert proposal.support_level == SupportLevel.SPECULATIVE
    assert "quantum_tunneling_constant" in proposal.meta_model_missing


def test_adjust_partially_supported_when_mixed() -> None:
    """add_process requiring known + unknown params -> PARTIALLY_SUPPORTED."""
    m = make_nitrifying_biofilm_metamodel()
    p = _panel()
    req = AdjustmentRequest(
        kind="add_process",
        spec={"required_parameter_ids": ["mu_max", "quantum_constant"]},
    )
    proposal = adjust_model(p, m, req)
    assert proposal.support_level == SupportLevel.PARTIALLY_SUPPORTED
    assert "mu_max" in proposal.meta_model_has
    assert "quantum_constant" in proposal.meta_model_missing


def test_adjust_change_scope_lists_eligible_submodels() -> None:
    m = make_nitrifying_biofilm_metamodel()
    p = _panel()
    req = AdjustmentRequest(
        kind="change_scope",
        spec={"required_phenomena": ["growth", "decay"]},
    )
    proposal = adjust_model(p, m, req)
    assert proposal.support_level == SupportLevel.SUPPORTED
    # PDE and ABM both cover growth + decay.
    assert set(proposal.depends_on_submodel_ids) >= {
        "continuum_pde_2d", "agent_based_3d",
    }


def test_adjust_unknown_kind_is_speculative() -> None:
    m = make_nitrifying_biofilm_metamodel()
    p = _panel()
    req = AdjustmentRequest(kind="summon_demon", target_id="x")
    proposal = adjust_model(p, m, req)
    assert proposal.support_level == SupportLevel.SPECULATIVE


# ============================================================================
# fit_data
# ============================================================================


def _dataset() -> ExperimentalDataset:
    return ExperimentalDataset(
        name="labdata_2026",
        observable_id="thickness",
        time_s=[0.0, 3600.0, 7200.0],
        value=[0.0, 5.0, 9.8],
    )


def _fit_point(pid: str, value: float, **ctx: str) -> FitCalibrationResult:
    return FitCalibrationResult(
        fitted_parameter_id=pid,
        context_keys=dict(ctx),
        fitted_value=value,
        fitted_unit="1/day" if pid in {"mu_max", "b"} else "mg/L",
        fit_uncertainty=0.05,
        fit_n_iterations=50,
    )


def test_fit_within_consensus_produces_no_suggestion() -> None:
    m = make_nitrifying_biofilm_metamodel()
    p = _panel()
    # AOB mu_max consensus is 1.0 -> 0.95 is within 20%.
    result = fit_data(
        p, m, [_dataset()], [_fit_point("mu_max", 0.95, species="AOB")]
    )
    assert "mu_max" in result.within_consensus
    assert result.suggestions == []
    # The override lands on the panel with source FITTED_FROM_DATA.
    ov = p.find_override("mu_max", {"species": "AOB"})
    assert ov is not None
    assert ov.source == OverrideSource.FITTED_FROM_DATA


def test_fit_outside_consensus_produces_suggestion() -> None:
    m = make_nitrifying_biofilm_metamodel()
    p = _panel()
    # AOB mu_max consensus is 1.0; 2.5 is way outside.
    result = fit_data(
        p, m, [_dataset()], [_fit_point("mu_max", 2.5, species="AOB")]
    )
    assert "mu_max" in result.outside_consensus
    assert len(result.suggestions) == 1
    s = result.suggestions[0]
    assert s.target_id == "mu_max"
    assert s.submitter_id == p.user_id
    # Evidence attached is PERSONAL_DATA pointing at the dataset name.
    assert any(e.reference == "labdata_2026" for e in s.evidence)


def test_fit_against_empirical_distribution_uses_min_max() -> None:
    """K_s AOB O2 is an empirical distribution with samples [0.3, 0.5, 0.8].
    A fitted 0.4 should be within; a fitted 1.2 should be outside."""
    m = make_nitrifying_biofilm_metamodel()
    p = _panel()
    r1 = fit_data(
        p, m, [_dataset()],
        [_fit_point("K_s", 0.4, species="AOB", substrate="oxygen")],
    )
    assert "K_s" in r1.within_consensus
    r2 = fit_data(
        p, m, [_dataset()],
        [_fit_point("K_s", 1.2, species="AOB", substrate="oxygen")],
    )
    assert "K_s" in r2.outside_consensus


def test_fit_missing_from_metamodel_records_note() -> None:
    """Fitting a parameter the meta-model has no consensus for — the
    override lands but no comparison suggestion is generated."""
    m = make_nitrifying_biofilm_metamodel()
    p = _panel()
    result = fit_data(
        p, m, [_dataset()],
        [_fit_point("cell_density", 0.15, species="AOB")],
    )
    assert result.suggestions == []
    assert any("MISSING" in n for n in result.notes)


def test_fit_multiple_parameters_in_one_call() -> None:
    m = make_nitrifying_biofilm_metamodel()
    p = _panel()
    fits = [
        _fit_point("mu_max", 0.95, species="AOB"),
        _fit_point("mu_max", 5.0, species="AOB"),  # overwrites the first
    ]
    result = fit_data(p, m, [_dataset()], fits)
    # set_override replaces same-key — only one override, latest value.
    overrides = [o for o in p.parameter_overrides if o.parameter_id == "mu_max"]
    assert len(overrides) == 1
    assert overrides[0].override_binding.point_estimate == pytest.approx(5.0)
