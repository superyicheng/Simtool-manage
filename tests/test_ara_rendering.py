"""Ara message-rendering tests.

Verify that every major Simtool object renders to readable text. Image
rendering is tested only if matplotlib is present.
"""

from __future__ import annotations

import pytest

from simtool.ara import (
    render_adjustment_proposal,
    render_assumption_ledger,
    render_fit_result,
    render_ir_compact,
    render_metamodel_parameters,
    render_metamodel_summary,
    render_output_bundle,
    render_panel_overrides,
    render_panel_summary,
    render_progress_line,
    render_progress_stream,
    render_recommendation,
    render_scope_status,
    render_submodel_hierarchy,
    render_suggestion,
    render_suggestion_ledger_summary,
)
from simtool.connector.assumptions import (
    Assumption,
    AssumptionCategory,
    AssumptionLedger,
    AssumptionSeverity,
)
from simtool.connector.ir import ParameterBinding
from simtool.connector.runs import OutputBundle, ProgressReport
from simtool.metamodel import (
    Evidence,
    EvidenceKind,
    SuggestionLedger,
    SuggestionTargetKind,
    evaluate_ir_against_scope,
    is_stale,
    staleness_warning,
)
from simtool.panels import (
    AdjustmentRequest,
    ExperimentalDataset,
    FitCalibrationResult,
    MeasurementCapability,
    OverrideSource,
    Panel,
    ParameterOverride,
    UserConstraints,
    adjust_model,
    fit_data,
    recommend_model,
)

from tests._metamodel_fixtures import (
    make_nitrifying_biofilm_metamodel,
    make_small_panel_ir,
)


# ---------------------------------------------------------------------------
# Meta-model renderers
# ---------------------------------------------------------------------------


def test_metamodel_summary_leads_with_version_and_counts() -> None:
    mm = make_nitrifying_biofilm_metamodel()
    text = render_metamodel_summary(mm)
    assert "# Meta-model" in text
    assert str(mm.version) in text
    assert "reconciled parameters" in text
    assert str(len(mm.reconciled_parameters)) in text


def test_metamodel_summary_flags_staleness() -> None:
    stale = make_nitrifying_biofilm_metamodel(last_ingested_days_ago=30)
    assert is_stale(stale)
    text = render_metamodel_summary(stale)
    assert "Staleness warning" in text


def test_metamodel_summary_no_warning_when_fresh() -> None:
    fresh = make_nitrifying_biofilm_metamodel(last_ingested_days_ago=2)
    text = render_metamodel_summary(fresh)
    assert "Staleness warning" not in text
    assert "last ingested" in text


def test_render_parameters_filters_by_id() -> None:
    mm = make_nitrifying_biofilm_metamodel()
    text = render_metamodel_parameters(mm, parameter_id="mu_max")
    assert "mu_max" in text
    assert "K_s" not in text


def test_render_parameters_empty_filter_reports_miss() -> None:
    mm = make_nitrifying_biofilm_metamodel()
    text = render_metamodel_parameters(mm, parameter_id="ghost")
    assert "no reconciled parameters" in text


def test_submodel_hierarchy_orders_by_complexity() -> None:
    mm = make_nitrifying_biofilm_metamodel()
    text = render_submodel_hierarchy(mm)
    pos_simple = text.find("monod_chemostat_ode")
    pos_pde = text.find("continuum_pde_2d")
    pos_abm = text.find("agent_based_3d")
    assert 0 < pos_simple < pos_pde < pos_abm


def test_submodel_hierarchy_lists_operators() -> None:
    mm = make_nitrifying_biofilm_metamodel()
    text = render_submodel_hierarchy(mm)
    assert "Approximation operators" in text
    assert "abm3d_to_pde2d_mean_field" in text


def test_scope_status_verdict_visible() -> None:
    mm = make_nitrifying_biofilm_metamodel()
    ir = make_small_panel_ir()
    report = evaluate_ir_against_scope(ir, mm)
    text = render_scope_status(report)
    assert "Verdict" in text
    # IR references S_bulk_initial which is not in meta-model -> BLOCKED.
    assert "BLOCKED" in text
    assert "S_bulk_initial" in text


# ---------------------------------------------------------------------------
# Panel renderers
# ---------------------------------------------------------------------------


def _panel() -> Panel:
    return Panel(
        id="ara-panel", title="Ara test panel", user_id="alice",
        meta_model_id="nitrifying_biofilm",
        meta_model_version_pin="1.2.0",
        derived_ir=make_small_panel_ir(),
        constraints=UserConstraints(
            predictive_priorities=["thickness"],
            measurement_capabilities=[MeasurementCapability(observable_id="thickness")],
            time_horizon_s=7200.0,
            required_phenomena=["growth"],
        ),
        tags=["testing", "nitrifier"],
    )


def test_panel_summary_has_state_and_pin() -> None:
    p = _panel()
    text = render_panel_summary(p)
    assert "Ara test panel" in text
    assert "DRAFT" in text
    assert "1.2.0" in text


def test_panel_summary_lists_tags() -> None:
    p = _panel()
    text = render_panel_summary(p)
    assert "testing" in text and "nitrifier" in text


def test_panel_summary_flags_frozen() -> None:
    p = _panel()
    p.freeze()
    text = render_panel_summary(p)
    assert "FROZEN" in text
    assert "frozen at" in text


def test_panel_overrides_shows_each() -> None:
    p = _panel()
    p.set_override(ParameterOverride(
        parameter_id="mu_max",
        context_keys={"species": "AOB"},
        override_binding=ParameterBinding(
            parameter_id="mu_max", canonical_unit="1/day",
            context_keys={"species": "AOB"}, point_estimate=1.5,
        ),
        source=OverrideSource.USER_PROVIDED,
        justification="my chemostat had higher growth",
    ))
    text = render_panel_overrides(p)
    assert "mu_max" in text
    assert "1.5" in text
    assert "user_provided" in text
    assert "chemostat" in text


def test_panel_overrides_empty() -> None:
    p = _panel()
    text = render_panel_overrides(p)
    assert "no overrides" in text


# ---------------------------------------------------------------------------
# Workflow renderers
# ---------------------------------------------------------------------------


def test_recommendation_renders_reasoning() -> None:
    mm = make_nitrifying_biofilm_metamodel()
    rec = recommend_model(
        mm,
        UserConstraints(
            predictive_priorities=["thickness"],
            measurement_capabilities=[MeasurementCapability(observable_id="thickness")],
            time_horizon_s=3600.0,
            required_phenomena=["growth"],
        ),
    )
    text = render_recommendation(rec)
    assert "monod_chemostat_ode" in text
    assert "Reasoning" in text
    assert "✓" in text or "✗" in text


def test_recommendation_no_fit_structured() -> None:
    mm = make_nitrifying_biofilm_metamodel()
    rec = recommend_model(
        mm,
        UserConstraints(
            predictive_priorities=["magic"],
            measurement_capabilities=[MeasurementCapability(observable_id="magic")],
            time_horizon_s=3600.0,
            required_phenomena=["telepathy"],
        ),
    )
    text = render_recommendation(rec)
    assert "NO FIT" in text
    assert "telepathy" in text


def test_adjustment_proposal_badge_supported() -> None:
    mm = make_nitrifying_biofilm_metamodel()
    p = _panel()
    prop = adjust_model(
        p, mm,
        AdjustmentRequest(kind="change_parameter", target_id="mu_max"),
    )
    text = render_adjustment_proposal(prop)
    assert "SUPPORTED" in text
    assert "mu_max" in text


def test_adjustment_proposal_badge_speculative() -> None:
    mm = make_nitrifying_biofilm_metamodel()
    p = _panel()
    prop = adjust_model(
        p, mm,
        AdjustmentRequest(kind="change_parameter", target_id="phlogiston"),
    )
    text = render_adjustment_proposal(prop)
    assert "SPECULATIVE" in text
    assert "phlogiston" in text


def test_fit_result_separates_within_and_outside() -> None:
    mm = make_nitrifying_biofilm_metamodel()
    p = _panel()
    result = fit_data(
        p, mm,
        [ExperimentalDataset(
            name="lab_data", observable_id="thickness",
            time_s=[0.0, 3600.0], value=[0.0, 5.0],
        )],
        [
            FitCalibrationResult(
                fitted_parameter_id="mu_max",
                context_keys={"species": "AOB"},
                fitted_value=0.95, fitted_unit="1/day",
                fit_uncertainty=0.05, fit_n_iterations=30,
            ),
            FitCalibrationResult(
                fitted_parameter_id="b",
                context_keys={"species": "AOB"},
                fitted_value=0.5, fitted_unit="1/day",  # consensus is 0.1 — way off
                fit_uncertainty=0.1, fit_n_iterations=30,
            ),
        ],
    )
    text = render_fit_result(result)
    assert "within meta-model consensus" in text
    assert "outside meta-model consensus" in text
    assert "mu_max" in text
    assert "Suggestions ready for community review" in text


# ---------------------------------------------------------------------------
# Assumption ledger rendering
# ---------------------------------------------------------------------------


def test_assumption_ledger_shows_blocking_when_pending() -> None:
    ledger = AssumptionLedger(
        ir_id="ir", framework="idynomics_2", framework_version="2.0.0",
    )
    ledger.add(Assumption(
        id="timestep", category=AssumptionCategory.NUMERICS,
        severity=AssumptionSeverity.MATERIAL,
        description="Fixed 60s timestep.",
        justification="default", surfaced_by="test",
        alternatives=["30s", "adaptive"],
    ))
    text = render_assumption_ledger(ledger)
    assert "Ready to run**: NO" in text
    assert "Blocking" in text
    assert "timestep" in text
    assert "30s" in text  # alternatives surfaced


def test_assumption_ledger_shows_ready_after_approval() -> None:
    ledger = AssumptionLedger(
        ir_id="ir", framework="idynomics_2", framework_version="2.0.0",
    )
    ledger.add(Assumption(
        id="ts", category=AssumptionCategory.NUMERICS,
        severity=AssumptionSeverity.MATERIAL,
        description="t", justification="j", surfaced_by="t",
    ))
    ledger.approve("ts", user_note="ok")
    text = render_assumption_ledger(ledger)
    assert "Ready to run**: yes" in text
    assert "✓" in text
    assert "ok" in text


# ---------------------------------------------------------------------------
# Progress + outputs
# ---------------------------------------------------------------------------


def test_progress_line_has_run_id_and_observables() -> None:
    pr = ProgressReport(
        run_id="run-42",
        sim_time_s=1800.0, sim_time_horizon_s=7200.0,
        timestep_index=30, timestep_total=120,
        wall_time_elapsed_s=5.0, wall_time_estimated_remaining_s=15.0,
        observables={"thickness": 2.5},
        message="reached seeding phase",
    )
    line = render_progress_line(pr)
    assert "run-42" in line
    assert "thickness=2.5" in line
    assert "25.0%" in line
    assert "seeding phase" in line


def test_progress_stream_joins_multiple() -> None:
    reports = [
        ProgressReport(run_id="r", sim_time_s=t)
        for t in (0.0, 3600.0, 7200.0)
    ]
    text = render_progress_stream(reports)
    assert text.count("\n") == 2


def test_output_bundle_summary() -> None:
    bundle = OutputBundle(
        run_id="r",
        scalar_time_series={"thickness": [(0.0, 0.0), (3600.0, 5.0), (7200.0, 9.8)]},
        flux_time_series={"NH4_flux": [(3600.0, -0.02)]},
        distributions={"mass": [0.1, 0.12, 0.15]},
    )
    text = render_output_bundle(bundle)
    assert "thickness" in text
    assert "3 points" in text
    assert "final = 9.8" in text
    assert "NH4_flux" in text
    assert "mass" in text


def test_output_bundle_empty_channels_ok() -> None:
    text = render_output_bundle(OutputBundle(run_id="r"))
    assert "run r" in text


# ---------------------------------------------------------------------------
# Suggestions
# ---------------------------------------------------------------------------


def _mk_suggestion():
    from simtool.metamodel import Suggestion
    return Suggestion(
        id="s1",
        meta_model_id="nitrifying_biofilm",
        meta_model_version_seen="1.2.0",
        target_kind=SuggestionTargetKind.PARAMETER,
        target_id="mu_max",
        target_context={"species": "AOB"},
        submitter_id="alice",
        summary="AOB mu_max should be lower per new data",
        proposed_change="lower point estimate to 0.8",
        evidence=[Evidence(kind=EvidenceKind.DOI, reference="10.1000/new")],
        submitter_confidence=0.75,
    )


def test_render_suggestion_shows_evidence_and_status() -> None:
    s = _mk_suggestion()
    text = render_suggestion(s)
    assert "s1" in text
    assert "mu_max" in text
    assert "alice" in text
    assert "10.1000/new" in text
    assert "pending" in text


def test_suggestion_ledger_summary_counts_by_status() -> None:
    ledger = SuggestionLedger(meta_model_id="nitrifying_biofilm")
    ledger.submit(_mk_suggestion())
    text = render_suggestion_ledger_summary(ledger)
    assert "pending: 1" in text
    assert "accepted: 0" in text
    assert "s1" in text


# ---------------------------------------------------------------------------
# IR compact
# ---------------------------------------------------------------------------


def test_ir_compact_shows_counts() -> None:
    ir = make_small_panel_ir()
    text = render_ir_compact(ir)
    assert ir.id in text
    assert ir.formalism in text
    assert "entities" in text
    assert "processes" in text


# ---------------------------------------------------------------------------
# Image rendering — only if matplotlib is installed
# ---------------------------------------------------------------------------


def _has_matplotlib() -> bool:
    try:
        import matplotlib  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_matplotlib(), reason="matplotlib not installed")
def test_render_scalar_timeseries_produces_png_bytes() -> None:
    from simtool.ara import render_scalar_timeseries
    png = render_scalar_timeseries(
        [(0.0, 0.0), (3600.0, 5.0), (7200.0, 9.8)],
        title="biofilm thickness",
        y_label="thickness (um)",
    )
    assert png.startswith(b"\x89PNG"), "output must be a valid PNG"


@pytest.mark.skipif(not _has_matplotlib(), reason="matplotlib not installed")
def test_render_distribution_handles_empirical() -> None:
    from simtool.ara import render_distribution
    from simtool.connector.ir import Distribution
    png = render_distribution(
        Distribution(shape="empirical", samples=[0.2, 0.5, 0.8, 0.3, 0.6]),
        title="K_s(O2) AOB",
    )
    assert png.startswith(b"\x89PNG")


def test_image_rendering_raises_clear_error_when_matplotlib_missing() -> None:
    """If matplotlib isn't installed, image renderers must raise a typed
    error with install guidance — not a generic ImportError."""
    if _has_matplotlib():
        pytest.skip("matplotlib is installed; cannot test the missing-case path")
    from simtool.ara.render_image import (
        ImageRenderingNotAvailable,
        render_scalar_timeseries,
    )
    with pytest.raises(ImageRenderingNotAvailable, match="matplotlib"):
        render_scalar_timeseries([(0.0, 0.0)], title="x")
