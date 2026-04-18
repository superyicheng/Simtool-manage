"""Community suggestions.

Every displayed parameter, reconciliation, and model structure in the UI
has an inline suggestion affordance. Users submit structured suggestions
(target, evidence, confidence); other users see them pending on the
artifact; a maintainer reviews each. Accepted suggestions produce a new
meta-model version and credit the suggester in the changelog.

Public by default. Rejections carry explanations; "request more
evidence" keeps the thread alive rather than closing it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class SuggestionTargetKind(str, Enum):
    PARAMETER = "parameter"
    RECONCILIATION = "reconciliation"
    MODEL_STRUCTURE = "model_structure"
    SCOPE = "scope"
    APPROXIMATION_OPERATOR = "approximation_operator"


class SuggestionStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    REQUEST_MORE_EVIDENCE = "request_more_evidence"
    SUPERSEDED = "superseded"                     # folded into a later suggestion


class EvidenceKind(str, Enum):
    DOI = "doi"
    PERSONAL_DATA = "personal_data"
    DOMAIN_ARGUMENT = "domain_argument"
    COUNTEREXAMPLE = "counterexample"


class Evidence(BaseModel):
    kind: EvidenceKind
    reference: str = Field(
        description="DOI, dataset handle, or free-text summary of the "
        "domain argument / counterexample."
    )
    note: str = ""


class Suggestion(BaseModel):
    """A structured community suggestion attached to a specific artifact
    in a meta-model."""

    id: str
    meta_model_id: str
    meta_model_version_seen: str = Field(
        description="Version of the meta-model the suggester saw when "
        "submitting. Later versions may moot the suggestion."
    )
    target_kind: SuggestionTargetKind
    target_id: str = Field(
        description="e.g. a parameter_id, a reconciliation key, a submodel_id."
    )
    target_context: dict[str, str] = Field(default_factory=dict)

    submitter_id: str
    submitted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    summary: str
    proposed_change: str = Field(
        description="What the suggester wants changed, concretely."
    )
    evidence: list[Evidence] = Field(default_factory=list)
    submitter_confidence: float = Field(ge=0.0, le=1.0, default=0.5)

    is_public: bool = True
    status: SuggestionStatus = SuggestionStatus.PENDING
    resolved_at: Optional[datetime] = None
    resolver_id: str = ""
    resolver_explanation: str = ""
    resulting_version: Optional[str] = None


class SuggestionLedger(BaseModel):
    """The full suggestion queue for one meta-model."""

    meta_model_id: str
    suggestions: list[Suggestion] = Field(default_factory=list)

    # --- submit --------------------------------------------------------------

    def submit(self, suggestion: Suggestion) -> None:
        if suggestion.meta_model_id != self.meta_model_id:
            raise ValueError(
                f"suggestion's meta_model_id '{suggestion.meta_model_id}' "
                f"does not match ledger's '{self.meta_model_id}'"
            )
        if any(s.id == suggestion.id for s in self.suggestions):
            raise ValueError(f"duplicate suggestion id: {suggestion.id}")
        self.suggestions.append(suggestion)

    # --- review --------------------------------------------------------------

    def accept(
        self,
        suggestion_id: str,
        reviewer_id: str,
        resulting_version: str,
        explanation: str = "",
    ) -> None:
        s = self._get(suggestion_id)
        s.status = SuggestionStatus.ACCEPTED
        s.resolved_at = datetime.now(timezone.utc)
        s.resolver_id = reviewer_id
        s.resolver_explanation = explanation
        s.resulting_version = resulting_version

    def reject(
        self, suggestion_id: str, reviewer_id: str, explanation: str
    ) -> None:
        if not explanation.strip():
            raise ValueError("rejection requires a non-empty explanation")
        s = self._get(suggestion_id)
        s.status = SuggestionStatus.REJECTED
        s.resolved_at = datetime.now(timezone.utc)
        s.resolver_id = reviewer_id
        s.resolver_explanation = explanation

    def request_more_evidence(
        self, suggestion_id: str, reviewer_id: str, note: str
    ) -> None:
        s = self._get(suggestion_id)
        s.status = SuggestionStatus.REQUEST_MORE_EVIDENCE
        s.resolver_id = reviewer_id
        s.resolver_explanation = note
        # Do NOT set resolved_at — the thread is still open.

    def supersede(
        self, suggestion_id: str, reviewer_id: str, superseder_id: str
    ) -> None:
        s = self._get(suggestion_id)
        if superseder_id not in {x.id for x in self.suggestions}:
            raise ValueError(f"superseder suggestion '{superseder_id}' not found")
        s.status = SuggestionStatus.SUPERSEDED
        s.resolved_at = datetime.now(timezone.utc)
        s.resolver_id = reviewer_id
        s.resolver_explanation = f"superseded by {superseder_id}"

    # --- queries -------------------------------------------------------------

    def pending(self) -> list[Suggestion]:
        return [s for s in self.suggestions if s.status == SuggestionStatus.PENDING]

    def accepted(self) -> list[Suggestion]:
        return [s for s in self.suggestions if s.status == SuggestionStatus.ACCEPTED]

    def for_target(
        self,
        target_kind: SuggestionTargetKind,
        target_id: str,
        include_resolved: bool = False,
    ) -> list[Suggestion]:
        """Every suggestion attached to a specific artifact.

        Default excludes resolved (accepted/rejected/superseded) — the UI
        usually wants the live thread for a given parameter/submodel.
        """
        alive = {
            SuggestionStatus.PENDING,
            SuggestionStatus.REQUEST_MORE_EVIDENCE,
        }
        out = []
        for s in self.suggestions:
            if s.target_kind != target_kind or s.target_id != target_id:
                continue
            if include_resolved or s.status in alive:
                out.append(s)
        return out

    def crediting_map(self) -> dict[str, list[str]]:
        """For each resulting_version, the list of submitter_ids whose
        accepted suggestions shaped that version. Feed to ChangelogEntry.credited_to."""
        out: dict[str, list[str]] = {}
        for s in self.accepted():
            if s.resulting_version is None:
                continue
            out.setdefault(s.resulting_version, []).append(s.submitter_id)
        return out

    # --- internals -----------------------------------------------------------

    def _get(self, suggestion_id: str) -> Suggestion:
        for s in self.suggestions:
            if s.id == suggestion_id:
                return s
        raise KeyError(f"unknown suggestion id: {suggestion_id}")

    @model_validator(mode="after")
    def _check_unique(self) -> "SuggestionLedger":
        ids = [s.id for s in self.suggestions]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate suggestion ids")
        return self
