"""Unit tests — community suggestions flow."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from simtool.metamodel import (
    Evidence,
    EvidenceKind,
    Suggestion,
    SuggestionLedger,
    SuggestionStatus,
    SuggestionTargetKind,
)


def _mk_suggestion(
    id_: str = "s1",
    target_id: str = "mu_max",
    meta_model_id: str = "nitrifying_biofilm",
    submitter: str = "alice@example.edu",
) -> Suggestion:
    return Suggestion(
        id=id_,
        meta_model_id=meta_model_id,
        meta_model_version_seen="1.2.0",
        target_kind=SuggestionTargetKind.PARAMETER,
        target_id=target_id,
        target_context={"species": "AOB"},
        submitter_id=submitter,
        summary="AOB mu_max should be lower",
        proposed_change="lower point estimate to 0.9 based on new data",
        evidence=[
            Evidence(kind=EvidenceKind.DOI, reference="10.1000/new_paper"),
        ],
        submitter_confidence=0.7,
    )


# --- Submit ---------------------------------------------------------------


def test_submit_adds_to_ledger() -> None:
    ledger = SuggestionLedger(meta_model_id="nitrifying_biofilm")
    ledger.submit(_mk_suggestion())
    assert len(ledger.suggestions) == 1
    assert ledger.pending()[0].submitter_id == "alice@example.edu"


def test_submit_different_meta_model_rejected() -> None:
    ledger = SuggestionLedger(meta_model_id="nitrifying_biofilm")
    bad = _mk_suggestion(meta_model_id="lithium_battery")
    with pytest.raises(ValueError, match="does not match"):
        ledger.submit(bad)


def test_duplicate_suggestion_id_rejected() -> None:
    ledger = SuggestionLedger(meta_model_id="nitrifying_biofilm")
    ledger.submit(_mk_suggestion("s1"))
    with pytest.raises(ValueError, match="duplicate suggestion id"):
        ledger.submit(_mk_suggestion("s1"))


# --- Accept ---------------------------------------------------------------


def test_accept_updates_state_and_credits() -> None:
    ledger = SuggestionLedger(meta_model_id="nitrifying_biofilm")
    ledger.submit(_mk_suggestion())
    ledger.accept(
        "s1",
        reviewer_id="maintainer@example.edu",
        resulting_version="1.3.0",
        explanation="accepted — value updated; new corpus entry.",
    )
    s = ledger.suggestions[0]
    assert s.status == SuggestionStatus.ACCEPTED
    assert s.resulting_version == "1.3.0"
    assert s.resolver_id == "maintainer@example.edu"
    assert s.resolved_at is not None
    # Credit map includes the accepted suggestion.
    credits = ledger.crediting_map()
    assert credits == {"1.3.0": ["alice@example.edu"]}


def test_crediting_map_groups_by_version() -> None:
    ledger = SuggestionLedger(meta_model_id="nitrifying_biofilm")
    ledger.submit(_mk_suggestion("s1", submitter="alice"))
    ledger.submit(_mk_suggestion("s2", target_id="K_s", submitter="bob"))
    ledger.submit(_mk_suggestion("s3", target_id="b", submitter="carol"))
    ledger.accept("s1", "maint", "1.3.0")
    ledger.accept("s2", "maint", "1.3.0")
    ledger.accept("s3", "maint", "1.4.0")
    credits = ledger.crediting_map()
    assert set(credits["1.3.0"]) == {"alice", "bob"}
    assert credits["1.4.0"] == ["carol"]


# --- Reject ---------------------------------------------------------------


def test_reject_requires_explanation() -> None:
    ledger = SuggestionLedger(meta_model_id="nitrifying_biofilm")
    ledger.submit(_mk_suggestion())
    with pytest.raises(ValueError, match="non-empty explanation"):
        ledger.reject("s1", "maint", "")


def test_reject_with_explanation_updates_state() -> None:
    ledger = SuggestionLedger(meta_model_id="nitrifying_biofilm")
    ledger.submit(_mk_suggestion())
    ledger.reject("s1", "maint", "evidence insufficient")
    s = ledger.suggestions[0]
    assert s.status == SuggestionStatus.REJECTED
    assert "insufficient" in s.resolver_explanation
    assert s.resolved_at is not None


# --- Request more evidence -------------------------------------------------


def test_request_more_evidence_keeps_thread_open() -> None:
    ledger = SuggestionLedger(meta_model_id="nitrifying_biofilm")
    ledger.submit(_mk_suggestion())
    ledger.request_more_evidence("s1", "maint", "need pH conditions")
    s = ledger.suggestions[0]
    assert s.status == SuggestionStatus.REQUEST_MORE_EVIDENCE
    # resolved_at must NOT be set — the thread is still live.
    assert s.resolved_at is None


# --- Supersede ------------------------------------------------------------


def test_supersede_requires_target_to_exist() -> None:
    ledger = SuggestionLedger(meta_model_id="nitrifying_biofilm")
    ledger.submit(_mk_suggestion("s1"))
    with pytest.raises(ValueError, match="not found"):
        ledger.supersede("s1", "maint", "ghost")


def test_supersede_updates_state() -> None:
    ledger = SuggestionLedger(meta_model_id="nitrifying_biofilm")
    ledger.submit(_mk_suggestion("s1"))
    ledger.submit(_mk_suggestion("s2", target_id="K_s"))
    ledger.supersede("s1", "maint", "s2")
    assert ledger.suggestions[0].status == SuggestionStatus.SUPERSEDED
    assert "s2" in ledger.suggestions[0].resolver_explanation


# --- Query ----------------------------------------------------------------


def test_for_target_filters_to_alive_by_default() -> None:
    ledger = SuggestionLedger(meta_model_id="nitrifying_biofilm")
    ledger.submit(_mk_suggestion("s_alive", target_id="mu_max"))
    ledger.submit(_mk_suggestion("s_dead", target_id="mu_max"))
    ledger.reject("s_dead", "maint", "not convincing")
    alive = ledger.for_target(SuggestionTargetKind.PARAMETER, "mu_max")
    assert len(alive) == 1
    assert alive[0].id == "s_alive"
    all_ = ledger.for_target(
        SuggestionTargetKind.PARAMETER, "mu_max", include_resolved=True
    )
    assert len(all_) == 2


def test_unknown_suggestion_id_raises_keyerror() -> None:
    ledger = SuggestionLedger(meta_model_id="nitrifying_biofilm")
    with pytest.raises(KeyError):
        ledger.accept("ghost", "maint", "1.3.0")


# --- Status enum coverage -------------------------------------------------


def test_status_enum_complete() -> None:
    assert {s.value for s in SuggestionStatus} == {
        "pending", "accepted", "rejected", "request_more_evidence", "superseded",
    }


def test_evidence_kind_enum_complete() -> None:
    assert {e.value for e in EvidenceKind} == {
        "doi", "personal_data", "domain_argument", "counterexample",
    }


# --- Submitter confidence bounds ------------------------------------------


def test_submitter_confidence_bounded() -> None:
    with pytest.raises(ValidationError):
        _mk_suggestion().__class__(
            id="s", meta_model_id="x", meta_model_version_seen="1.0.0",
            target_kind=SuggestionTargetKind.PARAMETER, target_id="p",
            submitter_id="u", summary="s", proposed_change="p",
            submitter_confidence=1.5,
        )
