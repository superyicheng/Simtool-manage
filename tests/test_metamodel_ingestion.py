"""Unit tests — ingestion tiers, candidates, jobs, staleness."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from simtool.metamodel import (
    IngestionCadence,
    IngestionJob,
    IngestionTier,
    IntegrationDecision,
    IntegrationDecisionKind,
    PaperCandidate,
    PaperSource,
    is_stale,
    staleness_warning,
)

from tests._metamodel_fixtures import make_nitrifying_biofilm_metamodel


# --- PaperCandidate tier transitions ---------------------------------------


def test_paper_candidate_defaults_to_detected() -> None:
    c = PaperCandidate(doi="10.1000/a", source=PaperSource.OPENALEX)
    assert c.tier == IngestionTier.DETECTED


def test_paper_candidate_tier_transitions() -> None:
    c = PaperCandidate(doi="10.1000/a", source=PaperSource.PUBMED)
    c.tier = IngestionTier.PROCESSED
    c.extractor_confidence = 0.8
    assert c.tier == IngestionTier.PROCESSED
    c.tier = IngestionTier.INTEGRATED
    assert c.tier == IngestionTier.INTEGRATED


def test_extractor_confidence_bounded() -> None:
    with pytest.raises(ValidationError):
        PaperCandidate(
            doi="10.1000/a", source=PaperSource.OPENALEX,
            extractor_confidence=1.5,
        )


# --- IngestionJob summary --------------------------------------------------


def test_ingestion_job_summary_counts_by_tier() -> None:
    candidates = [
        PaperCandidate(doi=f"10.1000/{i}", source=PaperSource.OPENALEX,
                       tier=IngestionTier.INTEGRATED)
        for i in range(3)
    ] + [
        PaperCandidate(doi=f"10.1000/p{i}", source=PaperSource.OPENALEX,
                       tier=IngestionTier.PROCESSED)
        for i in range(2)
    ] + [
        PaperCandidate(
            doi="10.1000/flagged", source=PaperSource.BIORXIV,
            tier=IngestionTier.FLAGGED_FOR_REVIEW,
        ),
    ]
    job = IngestionJob(
        meta_model_id="nitrifying_biofilm",
        candidates=candidates,
        triggered_manually=True,
    )
    summary = job.summary()
    assert summary["integrated"] == 3
    assert summary["processed"] == 2
    assert summary["flagged_for_review"] == 1
    assert summary["detected"] == 0
    assert len(job.flagged()) == 1


# --- IntegrationDecision ---------------------------------------------------


def test_flagged_decision_requires_reason() -> None:
    with pytest.raises(ValidationError, match="flagged decisions require a reason"):
        IntegrationDecision(
            candidate_doi="10.1000/x",
            kind=IntegrationDecisionKind.FLAGGED_FOR_REVIEW,
            reason="",
        )


def test_auto_integrated_decision_ok_with_short_reason() -> None:
    d = IntegrationDecision(
        candidate_doi="10.1000/x",
        kind=IntegrationDecisionKind.AUTO_INTEGRATED,
        reason="",
        decided_by="auto",
    )
    assert d.kind == IntegrationDecisionKind.AUTO_INTEGRATED


# --- Staleness -------------------------------------------------------------


def test_fresh_metamodel_not_stale() -> None:
    m = make_nitrifying_biofilm_metamodel(last_ingested_days_ago=2)
    assert not is_stale(m)


def test_old_metamodel_stale_under_weekly_cadence() -> None:
    m = make_nitrifying_biofilm_metamodel(last_ingested_days_ago=30)
    assert is_stale(m)


def test_never_ingested_is_stale() -> None:
    m = make_nitrifying_biofilm_metamodel()
    m.ingestion.last_ingestion_at = None
    assert is_stale(m)


def test_override_threshold() -> None:
    m = make_nitrifying_biofilm_metamodel(last_ingested_days_ago=10)
    # Cadence weekly -> default threshold 14 -> fresh.
    assert not is_stale(m)
    # Tight threshold -> stale.
    assert is_stale(m, override_threshold_days=7)


def test_cadence_affects_threshold() -> None:
    m = make_nitrifying_biofilm_metamodel(last_ingested_days_ago=30)
    m.ingestion.cadence = IngestionCadence.MONTHLY
    # Monthly cadence -> threshold 45 days -> still fresh.
    assert not is_stale(m)


def test_staleness_warning_string() -> None:
    m = make_nitrifying_biofilm_metamodel(last_ingested_days_ago=30)
    w = staleness_warning(m)
    assert w is not None
    assert m.id in w
    assert "last ingested" in w


def test_staleness_warning_none_when_fresh() -> None:
    m = make_nitrifying_biofilm_metamodel(last_ingested_days_ago=2)
    assert staleness_warning(m) is None


def test_manual_cadence_loose_threshold() -> None:
    m = make_nitrifying_biofilm_metamodel(last_ingested_days_ago=60)
    m.ingestion.cadence = IngestionCadence.MANUAL
    # Manual cadence -> threshold 90 days -> still fresh.
    assert not is_stale(m)


# --- Job round-trip --------------------------------------------------------


def test_ingestion_job_round_trip() -> None:
    job = IngestionJob(
        meta_model_id="m",
        candidates=[PaperCandidate(doi="10.1/a", source=PaperSource.ARXIV)],
    )
    restored = IngestionJob.model_validate_json(job.model_dump_json())
    assert restored.meta_model_id == "m"
    assert restored.candidates[0].source == PaperSource.ARXIV
