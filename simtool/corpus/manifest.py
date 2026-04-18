"""Corpus manifest + PRISMA log schemas.

The manifest tracks every paper considered — included or excluded —
so that the PRISMA flow diagram can be regenerated from the log and
every decision carries a logged reason.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class AccessStatus(str, Enum):
    OPEN_PMC = "open_pmc"
    OPEN_BIORXIV = "open_biorxiv"
    OPEN_OTHER = "open_other"  # e.g. eLife, PLOS when not in PMC
    INSTITUTIONAL = "institutional"
    PAYWALLED_LIKELY = "paywalled_likely"  # access not yet confirmed
    PAYWALLED_NO_ACCESS = "paywalled_no_access"
    UNKNOWN = "unknown"


class InclusionDecision(str, Enum):
    SCREENED_IN = "screened_in"  # passed title/abstract screen, full-text review pending
    INCLUDED = "included"  # full-text reviewed, included in final extraction set
    EXCLUDED_SCREENING = "excluded_screening"  # title/abstract screen
    EXCLUDED_ELIGIBILITY = "excluded_eligibility"  # full-text eligibility
    DEDUPLICATED = "deduplicated"


class CorpusEntry(BaseModel):
    doi: str
    title: str
    year: int
    journal: Optional[str] = None
    citation: Optional[str] = Field(
        default=None,
        description="Full 'authors (year) title. journal.' string. Optional at screening stage; "
        "should be filled on full-text ingestion.",
    )
    access: AccessStatus
    pmc_id: Optional[str] = None
    pdf_path: Optional[Path] = None
    verify_doi: bool = Field(
        default=False,
        description="True when the DOI was heuristically constructed and should be verified before fetching.",
    )

    decision: InclusionDecision
    decision_reason: str = Field(
        description="Required. Why this paper was included or excluded (e.g. 'no quantitative Monod params reported', 'duplicate of DOI X').",
    )
    decision_date: date
    decision_by: str = Field(
        description="Who/what made the decision — e.g. 'screener_v1' (automated agent) or a human name.",
    )

    # Optional metadata used by later pipeline stages
    target_organisms: list[str] = Field(default_factory=list)
    substrates: list[str] = Field(default_factory=list)
    methods_reported: list[str] = Field(default_factory=list)
    notes: Optional[str] = None


class SearchStep(BaseModel):
    """One step of the PRISMA search pipeline (database query, deduplication,
    screening, eligibility assessment)."""

    step_name: str
    description: str
    n_in: int
    n_out: int
    timestamp: date
    query: Optional[str] = None
    database: Optional[str] = None


class PrismaLog(BaseModel):
    """PRISMA flow log — reconstructs the flow diagram from counts + reasons."""

    target_system: str = Field(
        description="e.g. 'iDynoMiCS 2 nitrifying biofilm reference model v0.1'",
    )
    steps: list[SearchStep] = Field(default_factory=list)
    entries: list[CorpusEntry] = Field(default_factory=list)

    def n_by_decision(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self.entries:
            counts[e.decision.value] = counts.get(e.decision.value, 0) + 1
        return counts


def load_log(path: Path) -> PrismaLog:
    with open(path, "r", encoding="utf-8") as f:
        return PrismaLog.model_validate(yaml.safe_load(f))


def dump_log(log: PrismaLog, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(log.model_dump(mode="json"), f, sort_keys=False, allow_unicode=True)
