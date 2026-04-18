"""Integration tests — meta-model x panel x connector.

Verify that components CONNECT correctly — the extractor/reconciler's
meta-model populates a scope report the panel uses; the panel's derived
IR flows through the connector plugin; overrides travel from the user's
panel into the lowered artifact; version propagation respects freeze.

These are the tests that catch 'passes unit tests, dies on first real
user' failures.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from simtool.connector.ir import ParameterBinding, ScientificModel
from simtool.connector.plugin import DocSources
from simtool.connector.runs import RunLayout, RunRecord, RunStatus
from simtool.metamodel import (
    ChangelogEntry,
    ParameterStatus,
    PropagationPolicy,
    SemVer,
    SuggestionLedger,
    classify_change,
    evaluate_ir_against_scope,
    is_stale,
)
from simtool.panels import (
    AdjustmentRequest,
    ExperimentalDataset,
    FitCalibrationResult,
    MeasurementCapability,
    OverrideSource,
    Panel,
    ParameterOverride,
    PublicationState,
    RunHistoryEntry,
    UserConstraints,
    adjust_model,
    evaluate_panel_readiness,
    fit_data,
    propagate_to_version,
    recommend_model,
)

from tests._metamodel_fixtures import (
    make_nitrifying_biofilm_metamodel,
    make_nitrifying_biofilm_scope,
    make_small_panel_ir,
)
from tests.test_connector_plugin import MockPlugin


def _constraints(**o) -> UserConstraints:
    base = dict(
        predictive_priorities=["biofilm_thickness"],
        measurement_capabilities=[MeasurementCapability(observable_id="thickness")],
        time_horizon_s=7200.0,
        required_phenomena=["growth"],
    )
    base.update(o)
    return UserConstraints(**base)


def _panel(**o) -> Panel:
    base = dict(
        id="integration-panel",
        title="integration",
        user_id="alice",
        meta_model_id="nitrifying_biofilm",
        meta_model_version_pin="1.2.0",
        derived_ir=make_small_panel_ir(),
        constraints=_constraints(),
    )
    base.update(o)
    return Panel(**base)


# ---------------------------------------------------------------------------
# Meta-model + scope + panel seam
# ---------------------------------------------------------------------------


def test_panel_readiness_reflects_metamodel_scope() -> None:
    """The scope status of a panel's derived IR must match what
    evaluate_ir_against_scope computes directly against the meta-model."""
    m = make_nitrifying_biofilm_metamodel()
    p = _panel()
    direct = evaluate_ir_against_scope(p.derived_ir, m)
    via_helper = evaluate_panel_readiness(p, m)
    assert direct.per_parameter == via_helper.per_parameter


def test_scope_contract_threshold_affects_panel_readiness() -> None:
    from simtool.metamodel.scope import evaluate_ir_against_scope as eval_scope
    from simtool.metamodel import ScopeContract

    m = make_nitrifying_biofilm_metamodel()
    ir = make_small_panel_ir()
    strict = ScopeContract(min_records_for_reconciliation=5)
    report = eval_scope(ir, m, scope=strict)
    # Under strict threshold all parameters that had < 5 records drop to SINGLE.
    assert all(
        p.status != ParameterStatus.RECONCILED
        for p in report.per_parameter
        if p.parameter_id != "ghost"
    )


def test_staleness_warning_pairs_with_panel_version_pin() -> None:
    m = make_nitrifying_biofilm_metamodel(last_ingested_days_ago=30)
    p = _panel(meta_model_version_pin=str(m.version))
    assert is_stale(m)
    # Panel's pin matches the meta-model's current version but the data is
    # stale; the UI is expected to surface both facts independently.
    assert p.meta_model_version_pin == str(m.version)


# ---------------------------------------------------------------------------
# Panel override flows into the connector plugin's lowered artifact
# ---------------------------------------------------------------------------


def test_panel_override_flows_through_lowering(tmp_path: Path) -> None:
    """When the user overrides a parameter in the panel, the overridden
    value must land in the connector's lowered artifact — not the
    meta-model's original."""
    m = make_nitrifying_biofilm_metamodel()
    p = _panel()
    # Override AOB mu_max in the panel and rebuild the IR's binding from it.
    override = ParameterOverride(
        parameter_id="mu_max",
        context_keys={"species": "AOB"},
        original_binding=m.find_parameter("mu_max", {"species": "AOB"}).binding,
        override_binding=ParameterBinding(
            parameter_id="mu_max", canonical_unit="1/day",
            context_keys={"species": "AOB"},
            point_estimate=99.0,
        ),
        source=OverrideSource.USER_PROVIDED,
        justification="testing override propagation",
    )
    p.set_override(override)
    # The panel's derived IR needs to reflect the override before lowering.
    p.derived_ir = _apply_overrides_to_ir(p.derived_ir, p.parameter_overrides)

    plugin = MockPlugin()
    skill = plugin.parse_docs(DocSources(sources=[]))
    report = plugin.validate_ir(p.derived_ir, skill)
    assert report.ok
    artifact = plugin.lower(p.derived_ir, skill)
    ir_json = next(c for n, c in artifact.files if n == "ir.json")
    assert b"99.0" in ir_json, (
        "overridden parameter value did not reach the lowered artifact"
    )


def test_run_history_links_back_to_panel(tmp_path: Path) -> None:
    """Running the panel's IR produces a RunRecord; attaching it to the
    panel keeps the whole lifecycle traceable from panel -> run."""
    m = make_nitrifying_biofilm_metamodel()
    p = _panel()
    plugin = MockPlugin()
    skill = plugin.parse_docs(DocSources(sources=[]))
    assert plugin.validate_ir(p.derived_ir, skill).ok
    artifact = plugin.lower(p.derived_ir, skill)
    for a in artifact.assumptions.assumptions:
        artifact.assumptions.approve(a.id)
    layout = RunLayout.under(tmp_path / "integration_run")
    handle = plugin.execute(artifact, layout)
    list(plugin.monitor(handle))
    bundle = plugin.parse_outputs(layout)
    assert "thickness" in bundle.scalar_time_series

    run_record_path = layout.metadata / "run.json"
    layout.metadata.mkdir(parents=True, exist_ok=True)
    ledger_hash = "sha256:" + hashlib.sha256(
        artifact.assumptions.model_dump_json().encode()
    ).hexdigest()
    run = RunRecord(
        run_id=handle.run_id,
        ir_id=p.derived_ir.id,
        framework=plugin.name, framework_version="0.1",
        skill_file_version=skill.skill_schema_version,
        assumption_ledger_hash=ledger_hash,
        layout_root=layout.root,
        status=RunStatus.SUCCEEDED,
    )
    run_record_path.write_text(run.model_dump_json())

    p.attach_run(RunHistoryEntry(
        run_id=handle.run_id, ir_id=p.derived_ir.id,
        framework=plugin.name, framework_version="0.1",
        run_record_path=str(run_record_path.relative_to(layout.root)),
        status="succeeded",
    ))
    assert len(p.run_history) == 1
    assert p.run_history[0].run_id == handle.run_id


# ---------------------------------------------------------------------------
# Suggestion submission + accept + changelog credit + propagation
# ---------------------------------------------------------------------------


def test_fit_data_suggestion_closes_the_loop(tmp_path: Path) -> None:
    """End-to-end: fit produces an out-of-consensus suggestion; user
    submits it; maintainer accepts producing a new meta-model version;
    that version's changelog credits the submitter; panel propagation
    advances the pin."""
    m = make_nitrifying_biofilm_metamodel()
    p = _panel()

    # Fit AOB mu_max way off consensus.
    result = fit_data(
        p, m,
        [ExperimentalDataset(name="alice_2026", observable_id="thickness",
                              time_s=[0.0, 3600.0], value=[0.0, 8.0])],
        [FitCalibrationResult(
            fitted_parameter_id="mu_max",
            context_keys={"species": "AOB"},
            fitted_value=3.0, fitted_unit="1/day",
            fit_uncertainty=0.3, fit_n_iterations=80,
        )],
    )
    assert result.suggestions
    suggestion = result.suggestions[0]

    # User submits it to the community ledger.
    ledger = SuggestionLedger(meta_model_id=m.id)
    ledger.submit(suggestion)
    assert ledger.pending()
    # Maintainer reviews and accepts, producing version 1.3.0.
    new_version = "1.3.0"
    ledger.accept(
        suggestion.id, reviewer_id="maint@example",
        resulting_version=new_version,
        explanation="new corpus entry widens the reconciled range",
    )
    # Changelog records the acceptance + credits the submitter.
    credits = ledger.crediting_map()
    entry = ChangelogEntry(
        version=SemVer.parse(new_version), kind="update",
        summary="AOB mu_max reconciled range widened",
        affected_parameter_ids=["mu_max"],
        credited_to=credits[new_version],
        suggestion_ids=[suggestion.id],
    )
    assert p.user_id in entry.credited_to

    # Meta-model moves to 1.3.0.
    m.version = SemVer.parse(new_version)
    m.changelog.append(entry)

    # Panel propagates (MINOR -> auto).
    policy = PropagationPolicy()
    needs_confirm = policy.requires_user_confirmation(
        SemVer.parse(p.meta_model_version_pin),
        SemVer.parse(new_version),
    )
    outcome = propagate_to_version(
        p, new_version,
        policy_requires_confirmation=needs_confirm,
    )
    assert outcome.applied
    assert p.meta_model_version_pin == new_version


def test_major_propagation_to_frozen_panel_blocked() -> None:
    m = make_nitrifying_biofilm_metamodel()
    p = _panel()
    p.freeze()
    outcome = propagate_to_version(
        p, "2.0.0", policy_requires_confirmation=True,
    )
    assert outcome.kind == "blocked_frozen"
    # Freezing a panel is the user saying 'do not disturb for reproducibility'.
    assert p.meta_model_version_pin == "1.2.0"


# ---------------------------------------------------------------------------
# Recommendation + adjust + fit compose around one panel
# ---------------------------------------------------------------------------


def test_recommend_then_adjust_compose() -> None:
    m = make_nitrifying_biofilm_metamodel()
    rec = recommend_model(
        m, _constraints(required_phenomena=["growth", "spatial_gradients"])
    )
    assert rec.submodel_id == "continuum_pde_2d"

    p = _panel()
    req = AdjustmentRequest(
        kind="change_scope",
        spec={"required_phenomena": ["growth", "spatial_gradients", "decay"]},
    )
    proposal = adjust_model(p, m, req)
    assert proposal.depends_on_submodel_ids
    # Both PDE and ABM cover the expanded scope.
    assert "continuum_pde_2d" in proposal.depends_on_submodel_ids
    assert "agent_based_3d" in proposal.depends_on_submodel_ids


# ---------------------------------------------------------------------------
# Helper — minimal override applier for the IR (tests-only; production panels
# will rebuild the IR from the meta-model plus overrides more precisely).
# ---------------------------------------------------------------------------


def _apply_overrides_to_ir(
    ir: ScientificModel, overrides: list[ParameterOverride]
) -> ScientificModel:
    """Walk the IR and replace any ParameterBinding whose
    (parameter_id, context_keys) matches an override."""
    if not overrides:
        return ir
    ov_by_key = {
        (o.parameter_id, tuple(sorted(o.context_keys.items()))): o.override_binding
        for o in overrides
    }

    def _replace(b: ParameterBinding) -> ParameterBinding:
        return ov_by_key.get(
            (b.parameter_id, tuple(sorted(b.context_keys.items()))), b
        )

    new_ir = ir.model_copy(deep=True)
    for p in new_ir.processes:
        params = getattr(p, "parameters", None) or {}
        for k, v in list(params.items()):
            if isinstance(v, ParameterBinding):
                params[k] = _replace(v)
    for e in new_ir.entities:
        params = getattr(e, "parameters", None) or {}
        for k, v in list(params.items()):
            if isinstance(v, ParameterBinding):
                params[k] = _replace(v)
    for bc in new_ir.boundary_conditions:
        if bc.value is not None:
            bc.value = _replace(bc.value)
    return new_ir
