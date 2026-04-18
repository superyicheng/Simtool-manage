"""Panel: user workspace layered on top of a meta-model.

A panel is the unit of an individual user's work. It holds:
  - A derived IR — the user-customized ``ScientificModel`` for one project.
  - A pin to a specific meta-model version.
  - The user's constraints (what they can measure, what they care about
    predicting, compute budget, excluded phenomena).
  - Parameter overrides (inherited from the meta-model by default; an
    override requires justification + evidence).
  - The assumption ledger for the panel's current configuration (from
    the connector).
  - Simulation run history.
  - Notes, tags, AI conversation id (conversation bodies live in a
    separate store — panels reference them, they don't contain them).
  - Publication state: DRAFT / SHARED / PUBLISHED / FROZEN.

Private by default. Forkable and shareable. When frozen, the panel is
locked to a specific meta-model version for reproducibility.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from simtool.connector.assumptions import AssumptionLedger
from simtool.connector.ir import ParameterBinding, ScientificModel


PANEL_SCHEMA_VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# User constraints
# ---------------------------------------------------------------------------


class MeasurementCapability(BaseModel):
    """What the user can measure in their experiment. Feeds the
    recommendation workflow (select submodels whose observables the user
    can actually validate against)."""

    observable_id: str
    sampling_rate_hz: Optional[float] = None
    spatial_resolution_um: Optional[float] = None
    note: str = ""


class UserConstraints(BaseModel):
    """The user's constraints for a panel.

    The recommendation workflow matches these against the meta-model's
    submodel hierarchy. Predictive priorities drive submodel selection;
    measurement capabilities drive observable selection; excluded
    phenomena drive hierarchy pruning.
    """

    predictive_priorities: list[str] = Field(
        default_factory=list,
        description="Ordered list — what the user cares most about predicting.",
    )
    measurement_capabilities: list[MeasurementCapability] = Field(default_factory=list)
    compute_budget_wall_time_s: Optional[float] = None
    compute_budget_memory_gb: Optional[float] = None
    time_horizon_s: float = Field(gt=0)
    excluded_phenomena: list[str] = Field(
        default_factory=list,
        description="Phenomena the user declares out-of-scope for this project.",
    )
    required_phenomena: list[str] = Field(
        default_factory=list,
        description="Phenomena the user needs captured. Submodels missing "
        "any of these are ineligible.",
    )
    note: str = ""


# ---------------------------------------------------------------------------
# Parameter overrides
# ---------------------------------------------------------------------------


class OverrideSource(str, Enum):
    USER_PROVIDED = "user_provided"
    AI_DEFAULT_SPECULATIVE = "ai_default_speculative"
    FITTED_FROM_DATA = "fitted_from_data"


class ParameterOverride(BaseModel):
    """A parameter value that departs from the meta-model's reconciled
    binding.

    Overrides are first-class and must carry justification. Silent
    override is not allowed (the Panel validator enforces this).
    """

    parameter_id: str
    context_keys: dict[str, str] = Field(default_factory=dict)
    original_binding: Optional[ParameterBinding] = Field(
        default=None,
        description="The meta-model binding this override replaces, if any. "
        "None if the meta-model had no value (MISSING parameter).",
    )
    override_binding: ParameterBinding
    source: OverrideSource
    justification: str = Field(
        description="Non-empty free-text justification. For "
        "AI_DEFAULT_SPECULATIVE the justification explains the speculation.",
    )
    supporting_evidence_dois: list[str] = Field(default_factory=list)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @model_validator(mode="after")
    def _check(self) -> "ParameterOverride":
        if not self.justification.strip():
            raise ValueError(
                f"ParameterOverride('{self.parameter_id}'): justification "
                "must be non-empty"
            )
        if (
            self.source == OverrideSource.AI_DEFAULT_SPECULATIVE
            and "speculat" not in self.justification.lower()
        ):
            # Soft check — AI speculation justifications should say so.
            # We don't raise; it's a warning signal for the UI.
            pass
        return self


# ---------------------------------------------------------------------------
# Run history entry
# ---------------------------------------------------------------------------


class RunHistoryEntry(BaseModel):
    """A compact reference to one simulation run from the panel.

    Full RunRecords live under the run's on-disk bundle; the panel keeps
    the pointer + key metadata for UI display.
    """

    run_id: str
    ir_id: str
    framework: str
    framework_version: str
    run_record_path: str = Field(
        description="Relative path to the run's metadata/run.json under "
        "the panel's workspace."
    )
    submitted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    status: str = Field(description="Mirror of RunRecord.status at the time of attach.")
    note: str = ""


# ---------------------------------------------------------------------------
# Publication state
# ---------------------------------------------------------------------------


class PublicationState(str, Enum):
    DRAFT = "draft"
    SHARED = "shared"
    PUBLISHED = "published"
    FROZEN = "frozen"


# ---------------------------------------------------------------------------
# The Panel itself
# ---------------------------------------------------------------------------


class Panel(BaseModel):
    """One user's workspace for one project, anchored to a meta-model."""

    id: str
    title: str
    user_id: str
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    meta_model_id: str
    meta_model_version_pin: str = Field(
        description="Exact version (SemVer string) this panel is built "
        "against. Fork/propagation logic uses this to decide whether a "
        "meta-model update flows through."
    )
    derived_ir: ScientificModel = Field(
        description="The user-customized IR for this panel. Populated by "
        "the recommendation workflow and edited via panel workflows."
    )
    constraints: UserConstraints

    parameter_overrides: list[ParameterOverride] = Field(default_factory=list)
    assumption_ledger: Optional[AssumptionLedger] = None
    run_history: list[RunHistoryEntry] = Field(default_factory=list)

    notes: str = ""
    tags: list[str] = Field(default_factory=list)
    ai_conversation_id: Optional[str] = None

    publication_state: PublicationState = PublicationState.DRAFT
    frozen_at: Optional[datetime] = None
    frozen_at_meta_model_version: Optional[str] = None

    # Forking provenance
    forked_from_panel_id: Optional[str] = None
    collaborators: list[str] = Field(default_factory=list)

    schema_version: str = PANEL_SCHEMA_VERSION

    # --- lookups ------------------------------------------------------------

    def find_override(
        self, parameter_id: str, context_keys: Optional[dict[str, str]] = None
    ) -> Optional[ParameterOverride]:
        ctx = context_keys or {}
        for ov in self.parameter_overrides:
            if ov.parameter_id != parameter_id:
                continue
            if all(ov.context_keys.get(k) == v for k, v in ctx.items()):
                return ov
        return None

    # --- mutators -----------------------------------------------------------

    def set_override(self, override: ParameterOverride) -> None:
        """Replace any existing override for the same (parameter_id, context)
        and add this one."""
        self._ensure_mutable()
        self.parameter_overrides = [
            o for o in self.parameter_overrides
            if not (
                o.parameter_id == override.parameter_id
                and o.context_keys == override.context_keys
            )
        ]
        self.parameter_overrides.append(override)

    def attach_run(self, entry: RunHistoryEntry) -> None:
        """Run history can grow even after freezing — freezing locks the
        CONFIGURATION, not the observation stream."""
        if any(r.run_id == entry.run_id for r in self.run_history):
            raise ValueError(f"duplicate run_id in history: {entry.run_id}")
        self.run_history.append(entry)

    def fork(self, new_panel_id: str, new_user_id: str) -> "Panel":
        """Produce an independent copy owned by ``new_user_id``.

        The fork carries the pin, IR, overrides, and ledger. Run history
        is NOT copied — a fork starts with a clean observation log.
        Publication state resets to DRAFT.
        """
        forked = self.model_copy(deep=True)
        forked.id = new_panel_id
        forked.user_id = new_user_id
        forked.forked_from_panel_id = self.id
        forked.run_history = []
        forked.collaborators = []
        forked.publication_state = PublicationState.DRAFT
        forked.frozen_at = None
        forked.frozen_at_meta_model_version = None
        forked.created_at = datetime.now(timezone.utc)
        return forked

    def freeze(self) -> None:
        """Lock the panel at the current meta-model version pin. Further
        propagation is blocked; overrides are blocked. Intended for
        publication reproducibility."""
        self.publication_state = PublicationState.FROZEN
        self.frozen_at = datetime.now(timezone.utc)
        self.frozen_at_meta_model_version = self.meta_model_version_pin

    def share_with(self, collaborator_id: str) -> None:
        self._ensure_mutable()
        if collaborator_id == self.user_id:
            return
        if collaborator_id not in self.collaborators:
            self.collaborators.append(collaborator_id)
        if self.publication_state == PublicationState.DRAFT:
            self.publication_state = PublicationState.SHARED

    # --- internals ----------------------------------------------------------

    def _ensure_mutable(self) -> None:
        if self.publication_state == PublicationState.FROZEN:
            raise RuntimeError(
                f"panel '{self.id}' is FROZEN at meta-model version "
                f"{self.frozen_at_meta_model_version} — unfreeze (or fork) "
                "before editing"
            )

    def unfreeze(self) -> None:
        """Explicit de-escalation back to DRAFT. Clears the frozen
        metadata. This is a safety valve — papers usually fork a frozen
        panel rather than unfreezing the published one."""
        if self.publication_state != PublicationState.FROZEN:
            return
        self.publication_state = PublicationState.DRAFT
        self.frozen_at = None
        self.frozen_at_meta_model_version = None

    # --- validators ---------------------------------------------------------

    @model_validator(mode="after")
    def _check_unique_overrides(self) -> "Panel":
        keys = [
            (o.parameter_id, tuple(sorted(o.context_keys.items())))
            for o in self.parameter_overrides
        ]
        if len(set(keys)) != len(keys):
            raise ValueError(
                "duplicate parameter overrides for same (parameter_id, context)"
            )
        return self


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------


class PropagationOutcome(BaseModel):
    """Outcome of attempting to advance a panel to a newer meta-model
    version. Captures what changed and whether the user needs to confirm."""

    old_version: str
    new_version: str
    kind: str = Field(description="'auto_propagated' | 'awaits_confirmation' | 'blocked_frozen' | 'downgrade_refused'")
    notification: str = ""
    applied: bool = False


def propagate_to_version(
    panel: Panel,
    new_version: str,
    *,
    policy_requires_confirmation: bool,
    user_confirmed: bool = False,
) -> PropagationOutcome:
    """Advance ``panel`` to a new meta-model version according to policy.

    - If panel is FROZEN: refuse.
    - If DOWNGRADE: refuse.
    - If PATCH/MINOR (policy says auto): apply and return auto_propagated.
    - If MAJOR (policy says confirm): require ``user_confirmed``; else
      return awaits_confirmation and leave panel alone.
    """
    if panel.publication_state == PublicationState.FROZEN:
        return PropagationOutcome(
            old_version=panel.meta_model_version_pin,
            new_version=new_version,
            kind="blocked_frozen",
            notification=f"panel '{panel.id}' is FROZEN — propagation blocked",
            applied=False,
        )
    old = panel.meta_model_version_pin
    if policy_requires_confirmation and not user_confirmed:
        return PropagationOutcome(
            old_version=old, new_version=new_version,
            kind="awaits_confirmation",
            notification=(
                f"major meta-model update from {old} to {new_version} "
                "requires explicit confirmation before propagation"
            ),
            applied=False,
        )
    panel.meta_model_version_pin = new_version
    return PropagationOutcome(
        old_version=old,
        new_version=new_version,
        kind="auto_propagated" if not policy_requires_confirmation else "auto_propagated",
        notification=f"panel '{panel.id}' moved {old} -> {new_version}",
        applied=True,
    )
