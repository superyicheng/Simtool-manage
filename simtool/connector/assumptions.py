"""Assumption ledger.

When translating an IR into framework code, the plugin necessarily makes
implicit choices the user did not specify — well-mixedness, boundary
behavior, time discretization, parameter temperature-independence,
numerical tolerance, dimensional downgrade, parallelism, and so on. The
ledger surfaces every such choice with its justification and the
alternatives that would change it. The user must approve the ledger
before the simulation executes.

This is the mechanism that turns silent semantic failure into visible,
auditable negotiation — a prerequisite for scientific trust.

The ledger is produced by the plugin (not written by the user), but its
*approval* is user-driven: each entry transitions pending → approved or
pending → rejected before execution is allowed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class AssumptionCategory(str, Enum):
    """Coarse grouping so UIs can lay the ledger out sensibly."""

    PHYSICS = "physics"
    NUMERICS = "numerics"
    DISCRETIZATION = "discretization"
    BOUNDARY = "boundary"
    PARAMETERIZATION = "parameterization"
    COMPUTE = "compute"
    OTHER = "other"


class AssumptionSeverity(str, Enum):
    """How much the choice could affect results.

    CRITICAL — default values could silently flip qualitative behavior
    (e.g. steady-state vs oscillation).
    MATERIAL — quantitative predictions may shift more than literature
    uncertainty.
    ADVISORY — unlikely to affect user-facing observables within tolerance,
    but surfaced for auditability.
    """

    CRITICAL = "critical"
    MATERIAL = "material"
    ADVISORY = "advisory"


class AssumptionStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class Assumption(BaseModel):
    """A single implicit choice made during lowering IR -> framework code.

    Fields:
      id:            stable within a ledger; the user's approval attaches to it.
      category:      rough grouping for presentation.
      severity:      how much this choice could move results.
      description:   one sentence — what was decided.
      justification: why the plugin chose this default.
      alternatives:  other defensible choices; each is a free-text summary.
      surfaced_by:   which lowering step raised this (e.g.
                     "idynomics.boundary_lowering" or "idynomics.timestep_estimator").
      affects:       IR node ids (entity / process / BC / IC / observable)
                     whose output depends on this choice.
      status:        pending / approved / rejected.
      user_note:     optional free-text from the user when approving/rejecting.
    """

    id: str
    category: AssumptionCategory
    severity: AssumptionSeverity
    description: str
    justification: str
    alternatives: list[str] = Field(default_factory=list)
    surfaced_by: str
    affects: list[str] = Field(default_factory=list)
    status: AssumptionStatus = AssumptionStatus.PENDING
    user_note: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: Optional[datetime] = None


class AssumptionLedger(BaseModel):
    """The full set of assumptions for one IR -> framework lowering.

    A ledger is bound to both an IR id and a framework plugin (name+version)
    — the same IR lowered against two frameworks produces two ledgers with
    non-overlapping entries, because the implicit choices are plugin-specific.
    """

    ir_id: str
    framework: str = Field(description="Plugin name, e.g. 'idynomics_2'.")
    framework_version: str
    assumptions: list[Assumption] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # --- Construction ------------------------------------------------------

    def add(self, assumption: Assumption) -> None:
        if any(a.id == assumption.id for a in self.assumptions):
            raise ValueError(f"duplicate assumption id: {assumption.id}")
        self.assumptions.append(assumption)

    # --- Approval flow -----------------------------------------------------

    def approve(self, assumption_id: str, user_note: str = "") -> None:
        self._set_status(assumption_id, AssumptionStatus.APPROVED, user_note)

    def reject(self, assumption_id: str, user_note: str = "") -> None:
        self._set_status(assumption_id, AssumptionStatus.REJECTED, user_note)

    def _set_status(
        self, assumption_id: str, status: AssumptionStatus, user_note: str
    ) -> None:
        for a in self.assumptions:
            if a.id == assumption_id:
                a.status = status
                a.user_note = user_note
                a.resolved_at = datetime.now(timezone.utc)
                return
        raise KeyError(f"unknown assumption id: {assumption_id}")

    # --- Gating ------------------------------------------------------------

    def pending(self) -> list[Assumption]:
        return [a for a in self.assumptions if a.status == AssumptionStatus.PENDING]

    def rejected(self) -> list[Assumption]:
        return [a for a in self.assumptions if a.status == AssumptionStatus.REJECTED]

    def is_ready_to_run(self) -> bool:
        """Every assumption must be approved before execution is allowed.

        Any rejected assumption also blocks — the user must either accept the
        default, pick an alternative (which the plugin would replay), or
        change the IR so the assumption is no longer required.
        """
        return all(a.status == AssumptionStatus.APPROVED for a in self.assumptions)

    def blocking_reasons(self) -> list[str]:
        reasons: list[str] = []
        for a in self.pending():
            reasons.append(f"pending: {a.id} — {a.description}")
        for a in self.rejected():
            reasons.append(f"rejected: {a.id} — {a.user_note or 'no note'}")
        return reasons
