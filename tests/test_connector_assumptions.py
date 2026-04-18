"""Assumption-ledger gating tests.

The ledger is the contract that prevents silent semantic failure: a run
is blocked until every assumption is user-approved. These tests pin that
contract.
"""

from __future__ import annotations

import pytest

from simtool.connector.assumptions import (
    Assumption,
    AssumptionCategory,
    AssumptionLedger,
    AssumptionSeverity,
    AssumptionStatus,
)


def _make_ledger() -> AssumptionLedger:
    ledger = AssumptionLedger(
        ir_id="nitrifying_biofilm_v0",
        framework="idynomics_2",
        framework_version="2.0.0",
    )
    ledger.add(
        Assumption(
            id="timestep_default",
            category=AssumptionCategory.NUMERICS,
            severity=AssumptionSeverity.MATERIAL,
            description="Fixed timestep of 60 s used for solute PDE solver.",
            justification="CFL stability given diffusivities; user hint was 60 s.",
            alternatives=["30 s", "adaptive"],
            surfaced_by="idynomics.timestep_estimator",
            affects=["NH4_diffusion", "O2_diffusion"],
        )
    )
    ledger.add(
        Assumption(
            id="bulk_well_mixed",
            category=AssumptionCategory.BOUNDARY,
            severity=AssumptionSeverity.CRITICAL,
            description="Bulk liquid above biofilm is well-mixed (Dirichlet).",
            justification="IR declared dirichlet BCs; no boundary-layer model.",
            alternatives=["Robin BC with mass-transfer coefficient"],
            surfaced_by="idynomics.boundary_lowering",
            affects=["NH4_bulk", "O2_bulk"],
        )
    )
    return ledger


def test_pending_blocks_run() -> None:
    ledger = _make_ledger()
    assert not ledger.is_ready_to_run()
    assert len(ledger.blocking_reasons()) == 2


def test_approve_all_unblocks() -> None:
    ledger = _make_ledger()
    ledger.approve("timestep_default")
    ledger.approve("bulk_well_mixed", user_note="acceptable for high-Re bulk")
    assert ledger.is_ready_to_run()
    approved = [a for a in ledger.assumptions if a.status == AssumptionStatus.APPROVED]
    assert len(approved) == 2
    assert all(a.resolved_at is not None for a in approved)


def test_rejection_still_blocks() -> None:
    ledger = _make_ledger()
    ledger.approve("timestep_default")
    ledger.reject("bulk_well_mixed", user_note="need mass-transfer coefficient")
    assert not ledger.is_ready_to_run()
    assert any("rejected" in r for r in ledger.blocking_reasons())


def test_duplicate_assumption_id_raises() -> None:
    ledger = _make_ledger()
    with pytest.raises(ValueError):
        ledger.add(
            Assumption(
                id="timestep_default",  # duplicate
                category=AssumptionCategory.NUMERICS,
                severity=AssumptionSeverity.ADVISORY,
                description="x",
                justification="x",
                surfaced_by="test",
            )
        )


def test_unknown_id_raises() -> None:
    ledger = _make_ledger()
    with pytest.raises(KeyError):
        ledger.approve("does_not_exist")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_ledger_is_ready_vacuously() -> None:
    """An empty ledger ``is_ready_to_run`` is True by vacuous truth. This is
    intentional — a plugin with zero implicit choices to surface should not
    block the user. Plugins MUST actually declare all their assumptions;
    an empty ledger from a non-trivial lowering is a plugin bug.
    """
    ledger = AssumptionLedger(
        ir_id="empty", framework="mock", framework_version="0"
    )
    assert ledger.is_ready_to_run()
    assert ledger.blocking_reasons() == []


def test_double_approve_is_idempotent() -> None:
    ledger = _make_ledger()
    ledger.approve("timestep_default", user_note="first")
    first_resolved_at = next(
        a.resolved_at for a in ledger.assumptions if a.id == "timestep_default"
    )
    ledger.approve("timestep_default", user_note="second")
    a = next(x for x in ledger.assumptions if x.id == "timestep_default")
    assert a.status == AssumptionStatus.APPROVED
    assert a.user_note == "second"
    # resolved_at is rewritten — that's fine, it's the latest decision time.
    assert a.resolved_at is not None
    assert first_resolved_at is not None


def test_approve_then_reject_flips_status() -> None:
    ledger = _make_ledger()
    ledger.approve("timestep_default")
    assert ledger.assumptions[0].status == AssumptionStatus.APPROVED
    ledger.reject("timestep_default", user_note="changed my mind")
    assert ledger.assumptions[0].status == AssumptionStatus.REJECTED
    assert not ledger.is_ready_to_run()


def test_reject_then_approve_flips_status() -> None:
    ledger = _make_ledger()
    ledger.reject("timestep_default", user_note="too coarse")
    assert ledger.assumptions[0].status == AssumptionStatus.REJECTED
    ledger.approve("timestep_default", user_note="ok, acceptable")
    assert ledger.assumptions[0].status == AssumptionStatus.APPROVED


def test_blocking_reasons_include_both_pending_and_rejected() -> None:
    ledger = _make_ledger()
    ledger.reject("timestep_default", user_note="needs 30 s")
    # bulk_well_mixed remains pending
    reasons = ledger.blocking_reasons()
    assert any(r.startswith("rejected") for r in reasons)
    assert any(r.startswith("pending") for r in reasons)


def test_ledger_json_round_trip() -> None:
    ledger = _make_ledger()
    ledger.approve("timestep_default", user_note="reviewed")
    payload = ledger.model_dump_json()
    restored = AssumptionLedger.model_validate_json(payload)
    assert restored.ir_id == ledger.ir_id
    assert restored.framework == ledger.framework
    assert len(restored.assumptions) == 2
    # Status survives round-trip.
    a = next(x for x in restored.assumptions if x.id == "timestep_default")
    assert a.status == AssumptionStatus.APPROVED
    assert a.user_note == "reviewed"
    assert a.resolved_at is not None


def test_pending_and_rejected_filters() -> None:
    ledger = _make_ledger()
    ledger.approve("timestep_default")
    ledger.reject("bulk_well_mixed", user_note="need mass transfer")
    assert ledger.pending() == []
    assert len(ledger.rejected()) == 1
    assert ledger.rejected()[0].id == "bulk_well_mixed"


def test_assumption_with_minimal_fields_constructs() -> None:
    """alternatives, affects, user_note are all optional/default-empty."""
    a = Assumption(
        id="x",
        category=AssumptionCategory.OTHER,
        severity=AssumptionSeverity.ADVISORY,
        description="a thing",
        justification="because",
        surfaced_by="test",
    )
    assert a.alternatives == []
    assert a.affects == []
    assert a.status == AssumptionStatus.PENDING


def test_unknown_category_rejected() -> None:
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        Assumption(
            id="x",
            category="gut_feeling",  # type: ignore[arg-type]
            severity=AssumptionSeverity.ADVISORY,
            description="x",
            justification="x",
            surfaced_by="test",
        )


def test_unknown_severity_rejected() -> None:
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        Assumption(
            id="x",
            category=AssumptionCategory.OTHER,
            severity="catastrophic",  # type: ignore[arg-type]
            description="x",
            justification="x",
            surfaced_by="test",
        )


def test_severity_ordering_is_documented() -> None:
    """Preserve the three-level ordering used by the spec:
    CRITICAL > MATERIAL > ADVISORY. This test encodes that ordering by
    asserting the enum values exist and are distinct."""
    levels = {s.value for s in AssumptionSeverity}
    assert levels == {"critical", "material", "advisory"}
