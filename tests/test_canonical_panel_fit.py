"""Canonical workflow 3: the DATA FITTING user story.

User narrative:

    "I have two weeks of biofilm thickness measurements from my
    bioreactor. Calibrate the AOB mu_max and decay rate against my
    data. Tell me how my fitted values compare to the literature — if
    they're outside the reconciled range, help me submit them back so
    the meta-model reflects them."

What the system must deliver:
  1. Accept the user's experimental data and fitted parameter results.
  2. Record each fitted value as a Panel override with source
     FITTED_FROM_DATA.
  3. Compare each fit to the meta-model's reconciled range: within ->
     confirm; outside -> generate a Suggestion stub.
  4. The user submits the suggestion to the community; maintainer
     accepts; a new meta-model version credits the user.
  5. The closed loop means individual experimental data can drive
     shared knowledge improvement.
"""

from __future__ import annotations

import pytest

from simtool.metamodel import (
    ChangelogEntry,
    PropagationPolicy,
    SemVer,
    SuggestionLedger,
    SuggestionStatus,
)
from simtool.panels import (
    ExperimentalDataset,
    FitCalibrationResult,
    MeasurementCapability,
    OverrideSource,
    Panel,
    PublicationState,
    UserConstraints,
    fit_data,
    propagate_to_version,
)

from tests._metamodel_fixtures import (
    make_nitrifying_biofilm_metamodel,
    make_small_panel_ir,
)


def _panel() -> Panel:
    return Panel(
        id="fit-panel", title="fit test", user_id="alice@lab.edu",
        meta_model_id="nitrifying_biofilm", meta_model_version_pin="1.2.0",
        derived_ir=make_small_panel_ir(),
        constraints=UserConstraints(
            predictive_priorities=["thickness"],
            measurement_capabilities=[MeasurementCapability(observable_id="thickness")],
            time_horizon_s=86400.0 * 14,
            required_phenomena=["growth"],
        ),
    )


def _labdata() -> list[ExperimentalDataset]:
    return [
        ExperimentalDataset(
            name="alice_2026_thickness",
            observable_id="thickness",
            time_s=[0.0, 86400.0, 2 * 86400.0, 4 * 86400.0, 7 * 86400.0],
            value=[0.0, 1.2, 3.1, 6.8, 11.9],
        ),
    ]


def test_user_fits_within_consensus_gets_confirmation() -> None:
    """Fitted value lands inside literature range — user is told their
    experiment agrees with prior work."""
    m = make_nitrifying_biofilm_metamodel()
    p = _panel()
    result = fit_data(
        p, m, _labdata(),
        [FitCalibrationResult(
            fitted_parameter_id="mu_max",
            context_keys={"species": "AOB"},
            fitted_value=0.92, fitted_unit="1/day",
            fit_uncertainty=0.04, fit_n_iterations=40,
        )],
    )
    assert "mu_max" in result.within_consensus
    assert result.suggestions == []
    # Override recorded with fitting provenance.
    ov = p.find_override("mu_max", {"species": "AOB"})
    assert ov is not None
    assert ov.source == OverrideSource.FITTED_FROM_DATA
    assert "alice_2026_thickness" in ov.justification


def test_user_fits_outside_consensus_gets_suggestion_stub() -> None:
    """Fitted value is outside literature range — user is given a ready
    Suggestion they can review and submit."""
    m = make_nitrifying_biofilm_metamodel()
    p = _panel()
    result = fit_data(
        p, m, _labdata(),
        [FitCalibrationResult(
            fitted_parameter_id="mu_max",
            context_keys={"species": "AOB"},
            fitted_value=2.4, fitted_unit="1/day",
            fit_uncertainty=0.2, fit_n_iterations=60,
        )],
    )
    assert "mu_max" in result.outside_consensus
    assert len(result.suggestions) == 1
    s = result.suggestions[0]
    assert s.submitter_id == p.user_id
    assert s.target_id == "mu_max"
    assert "outside current meta-model consensus" in s.summary
    # Evidence includes the dataset that produced the fit.
    assert any("alice_2026_thickness" in e.reference for e in s.evidence)


def test_closed_loop_user_submits_suggestion_community_accepts(
) -> None:
    """The full closed loop: fit -> suggestion -> submit -> accept ->
    new version -> credit -> panel propagates."""
    m = make_nitrifying_biofilm_metamodel()
    p = _panel()
    result = fit_data(
        p, m, _labdata(),
        [FitCalibrationResult(
            fitted_parameter_id="mu_max",
            context_keys={"species": "AOB"},
            fitted_value=3.1, fitted_unit="1/day",
            fit_uncertainty=0.3, fit_n_iterations=90,
        )],
    )
    suggestion = result.suggestions[0]

    ledger = SuggestionLedger(meta_model_id=m.id)
    ledger.submit(suggestion)
    assert ledger.pending()
    ledger.accept(
        suggestion.id, reviewer_id="maint@example.edu",
        resulting_version="1.3.0",
        explanation="alice's data widens the reconciled AOB mu_max range",
    )
    assert ledger.suggestions[0].status == SuggestionStatus.ACCEPTED

    credits = ledger.crediting_map()
    assert p.user_id in credits["1.3.0"]

    # Advance meta-model with a changelog entry crediting the user.
    m.version = SemVer.parse("1.3.0")
    m.changelog.append(ChangelogEntry(
        version=m.version, kind="update",
        summary="AOB mu_max reconciled range widened",
        affected_parameter_ids=["mu_max"],
        credited_to=credits["1.3.0"],
        suggestion_ids=[suggestion.id],
    ))

    # Panel propagates (MINOR -> auto).
    policy = PropagationPolicy()
    needs_confirm = policy.requires_user_confirmation(
        SemVer.parse(p.meta_model_version_pin),
        m.version,
    )
    outcome = propagate_to_version(
        p, str(m.version), policy_requires_confirmation=needs_confirm,
    )
    assert outcome.applied
    assert p.meta_model_version_pin == "1.3.0"


def test_user_fits_parameter_missing_from_metamodel_gets_note_not_suggestion(
) -> None:
    """When the meta-model has no consensus for a parameter, fitting it
    is still fine — but there's nothing to compare against, so no
    suggestion. The system records a note so the user knows why no
    comparison appeared."""
    m = make_nitrifying_biofilm_metamodel()
    p = _panel()
    result = fit_data(
        p, m, _labdata(),
        [FitCalibrationResult(
            fitted_parameter_id="cell_density",
            context_keys={"species": "AOB"},
            fitted_value=0.18, fitted_unit="pg/fL",
            fit_uncertainty=0.01, fit_n_iterations=30,
        )],
    )
    assert result.suggestions == []
    assert any("MISSING" in n for n in result.notes)
    assert p.find_override("cell_density", {"species": "AOB"}) is not None


def test_fit_does_not_pollute_other_panels() -> None:
    """Fits apply to the specific panel passed in. A second panel on the
    same meta-model is untouched — a fit is a personal override, not a
    community update (until the suggestion path is taken)."""
    m = make_nitrifying_biofilm_metamodel()
    p1 = _panel()
    p2 = _panel()
    p2.id = "p2"
    p2.user_id = "bob"
    fit_data(
        p1, m, _labdata(),
        [FitCalibrationResult(
            fitted_parameter_id="mu_max",
            context_keys={"species": "AOB"},
            fitted_value=2.0, fitted_unit="1/day",
            fit_uncertainty=0.1, fit_n_iterations=40,
        )],
    )
    assert p1.parameter_overrides
    assert p2.parameter_overrides == []


def test_frozen_panel_rejects_fits() -> None:
    """A frozen panel is locked for reproducibility — fits cannot mutate
    it. User must fork first."""
    m = make_nitrifying_biofilm_metamodel()
    p = _panel()
    p.freeze()
    assert p.publication_state == PublicationState.FROZEN
    with pytest.raises(RuntimeError, match="FROZEN"):
        fit_data(
            p, m, _labdata(),
            [FitCalibrationResult(
                fitted_parameter_id="mu_max",
                context_keys={"species": "AOB"},
                fitted_value=1.0, fitted_unit="1/day",
                fit_uncertainty=0.05, fit_n_iterations=40,
            )],
        )
