"""Ingestion tiers + staleness.

Meta-models stay current via a recurring ingestion job (weekly by
default, manually triggerable) that queries OpenAlex, PubMed, bioRxiv,
and arXiv for new papers matching each meta-model's scope.

Paper lifecycle tiers:
  DETECTED   queued, not yet read.
  PROCESSED  extracted with confidence scores, not yet integrated.
  INTEGRATED reconciled into the meta-model -> a new version exists.

Integration decision logic (policy, enforced downstream):
  - Low-stakes additions agreeing with existing consensus: AUTO_INTEGRATED.
  - High-stakes additions (contradiction, new model structure, missing
    context, failed QC): FLAGGED_FOR_REVIEW.
Maintainers see the flagged queue; accepted review decisions produce a
new version.

This module defines the datatypes. The actual HTTP clients that query
OpenAlex/PubMed/etc. live in the corpus module (different session).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from simtool.metamodel.library import MetaModel


class PaperSource(str, Enum):
    OPENALEX = "openalex"
    PUBMED = "pubmed"
    BIORXIV = "biorxiv"
    ARXIV = "arxiv"
    MANUAL = "manual"


class IngestionTier(str, Enum):
    DETECTED = "detected"
    PROCESSED = "processed"
    INTEGRATED = "integrated"
    SKIPPED = "skipped"          # deduplicated or out-of-scope
    FLAGGED_FOR_REVIEW = "flagged_for_review"


class IntegrationDecisionKind(str, Enum):
    AUTO_INTEGRATED = "auto_integrated"
    FLAGGED_FOR_REVIEW = "flagged_for_review"
    REJECTED = "rejected"


class PaperCandidate(BaseModel):
    """One paper discovered by the ingestion pipeline.

    Progresses through tiers; stays attached to its meta-model throughout.
    """

    doi: str
    source: PaperSource
    detected_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    title: str = ""
    abstract_excerpt: str = ""
    tier: IngestionTier = IngestionTier.DETECTED
    extractor_confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Populated at PROCESSED tier.",
    )
    integration_decision: Optional[IntegrationDecisionKind] = None
    review_notes: str = ""


class IngestionJob(BaseModel):
    """A single run of the ingestion pipeline against one meta-model.

    A job can be scheduled (cadence-driven) or manual. The ``started_at``
    / ``ended_at`` bracket the run; ``candidates`` is the full set of
    papers it touched (grouped by final tier).
    """

    meta_model_id: str
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    ended_at: Optional[datetime] = None
    triggered_manually: bool = False
    candidates: list[PaperCandidate] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    def summary(self) -> dict[str, int]:
        out = {tier.value: 0 for tier in IngestionTier}
        for c in self.candidates:
            out[c.tier.value] += 1
        return out

    def flagged(self) -> list[PaperCandidate]:
        return [c for c in self.candidates if c.tier == IngestionTier.FLAGGED_FOR_REVIEW]


class IntegrationDecision(BaseModel):
    """Per-candidate decision the ingestion pipeline makes at integration time."""

    candidate_doi: str
    kind: IntegrationDecisionKind
    reason: str
    affects_parameter_ids: list[str] = Field(default_factory=list)
    affects_submodel_ids: list[str] = Field(default_factory=list)
    decided_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    decided_by: str = Field(
        default="",
        description="User id of the maintainer (for FLAGGED_FOR_REVIEW "
        "decisions) or 'auto' for AUTO_INTEGRATED.",
    )

    @model_validator(mode="after")
    def _check(self) -> "IntegrationDecision":
        if self.kind == IntegrationDecisionKind.FLAGGED_FOR_REVIEW and not self.reason:
            raise ValueError("flagged decisions require a reason")
        return self


# ---------------------------------------------------------------------------
# Staleness
# ---------------------------------------------------------------------------


DEFAULT_STALENESS_DAYS = {
    "weekly": 14,
    "biweekly": 21,
    "monthly": 45,
    "manual": 90,
}


def is_stale(
    meta_model: MetaModel,
    now: Optional[datetime] = None,
    override_threshold_days: Optional[int] = None,
) -> bool:
    """Is the meta-model considered out of date at ``now``?

    Threshold comes from the meta-model's declared cadence unless
    overridden. A meta-model that has never been ingested is always
    stale.
    """
    if override_threshold_days is not None:
        threshold_days = override_threshold_days
    else:
        threshold_days = DEFAULT_STALENESS_DAYS.get(
            meta_model.ingestion.cadence.value, 14
        )
    last = meta_model.ingestion.last_ingestion_at
    if last is None:
        return True
    ref = now or datetime.now(timezone.utc)
    return (ref - last) > timedelta(days=threshold_days)


def staleness_warning(meta_model: MetaModel, now: Optional[datetime] = None) -> Optional[str]:
    """If stale, return a user-facing warning string; else None.

    The warning names the meta-model and its last ingestion date so users
    can judge whether to proceed against stale data.
    """
    if not is_stale(meta_model, now):
        return None
    last = meta_model.ingestion.last_ingestion_at
    if last is None:
        return (
            f"meta-model '{meta_model.id}' (v{meta_model.version}) has "
            "never been ingested; literature may have moved since release"
        )
    return (
        f"meta-model '{meta_model.id}' (v{meta_model.version}) last "
        f"ingested {last.isoformat()}; "
        f"cadence={meta_model.ingestion.cadence.value}"
    )
