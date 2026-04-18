"""The MetaModel: shared scientific substrate.

A ``MetaModel`` is the field's reconciled knowledge about a specific
scientific system, populated from the primary literature. One meta-model
per system (e.g. Postia placenta growth, nitrifying biofilm, lithium-ion
cathode degradation). Community-maintained, versioned (SemVer), citable
(Zenodo DOI), updated on a recurring ingestion schedule.

Unity-with-IR note: a meta-model is essentially an IR populated from
literature. ``reconciled_parameters`` holds ``ParameterBinding``s keyed
by (parameter_id, context) — exactly the shape the connector's lower()
consumes. A panel's derived-IR is built by selecting a submodel and
materializing its parameters from this reconciled layer.

This module defines the META-MODEL as a shared artifact; it does NOT
define the reconciler internals (those live in the extractor/matcher/qc
pipeline owned by the main-body session). ``ReconciledParameter`` below
is the OUTPUT CONTRACT of the reconciler — independent of how the
reconciler computes it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from simtool.connector.ir import ParameterBinding
from simtool.metamodel.versioning import SemVer


# ---------------------------------------------------------------------------
# Reconciled parameter — the meta-model's payload for a (parameter_id, context)
# ---------------------------------------------------------------------------


class QualityRating(str, Enum):
    """Coarse quality signal attached to a reconciled parameter. The
    underlying GRADE rating per source record lives in ParameterRecord."""

    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    VERY_LOW = "very_low"


class ReconciledParameter(BaseModel):
    """A single reconciled parameter at the meta-model level.

    This is the RECONCILER'S OUTPUT contract — the shape every consumer
    (panels, connector, scope contract) depends on. The reconciler owns
    how it computes these; we own the shape.
    """

    parameter_id: str
    context_keys: dict[str, str] = Field(default_factory=dict)
    binding: ParameterBinding
    supporting_record_dois: list[str] = Field(default_factory=list)
    quality_rating: QualityRating = QualityRating.MODERATE
    conflict_flags: list[str] = Field(default_factory=list)
    last_updated: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ---------------------------------------------------------------------------
# Submodel hierarchy + approximation operators
# ---------------------------------------------------------------------------


class SubmodelEntry(BaseModel):
    """A node in the submodel hierarchy.

    ``complexity_rank`` orders simpler (lower) to richer (higher) models
    on a monotone scale. ``ir_template_ref`` is an opaque identifier the
    system's IR-template catalogue resolves to an actual ScientificModel
    skeleton with placeholder parameters; panels materialize placeholders
    from ``reconciled_parameters``.
    """

    id: str
    name: str
    description: str = ""
    complexity_rank: int = Field(
        ge=0,
        description="0 = simplest (lumped ODE); higher = richer (e.g. 3D agent-based).",
    )
    parent_id: Optional[str] = None
    ir_template_ref: str = Field(
        description="Catalogue key for the IR template this submodel materializes to."
    )
    required_parameter_ids: list[str] = Field(default_factory=list)
    excluded_phenomena: list[str] = Field(
        default_factory=list,
        description="Phenomena this submodel intentionally drops (e.g. "
        "'EPS_production', 'detachment'). Users asking for these must "
        "move up the hierarchy.",
    )
    covered_phenomena: list[str] = Field(
        default_factory=list,
        description="Phenomena this submodel captures.",
    )


class ApproximationOperatorKind(str, Enum):
    """Formal operators that bridge submodels of differing complexity."""

    AVERAGING = "averaging"
    ADIABATIC_ELIMINATION = "adiabatic_elimination"
    MEAN_FIELD_CLOSURE = "mean_field_closure"
    DIMENSIONAL_REDUCTION = "dimensional_reduction"
    LUMPING = "lumping"
    QUASI_STEADY_STATE = "quasi_steady_state"


class ApproximationOperator(BaseModel):
    """An operator that derives a simpler submodel from a richer one.

    When a user's compute budget doesn't fit any pre-enumerated submodel,
    the recommendation workflow composes approximation operators to
    synthesize a custom simplification on demand.
    """

    id: str
    kind: ApproximationOperatorKind
    from_submodel_id: str
    to_submodel_id: str
    description: str
    assumptions_introduced: list[str] = Field(
        default_factory=list,
        description="Assumptions this operator imposes. These feed the "
        "assumption ledger when the operator is applied in a panel.",
    )
    validity_conditions: list[str] = Field(
        default_factory=list,
        description="Regimes where the approximation is defensible "
        "(e.g. 'fast substrate relaxation vs growth timescale').",
    )


# ---------------------------------------------------------------------------
# Ingestion status snapshot (detailed types live in ingestion.py)
# ---------------------------------------------------------------------------


class IngestionCadence(str, Enum):
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"
    MANUAL = "manual"


class IngestionStatus(BaseModel):
    """Snapshot of the ingestion pipeline's state at last run."""

    last_ingestion_at: Optional[datetime] = None
    next_scheduled_at: Optional[datetime] = None
    cadence: IngestionCadence = IngestionCadence.WEEKLY
    papers_detected: int = 0
    papers_processed: int = 0
    papers_integrated: int = 0
    papers_flagged_for_review: int = 0


# ---------------------------------------------------------------------------
# Changelog
# ---------------------------------------------------------------------------


class ChangelogEntry(BaseModel):
    """One changelog entry attached to a meta-model version.

    Accepted suggestions credit the suggester via ``credited_to`` — that
    is how the community-contribution loop closes on the published side.
    """

    at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: SemVer
    kind: str = Field(description="'add' | 'update' | 'remove' | 'structural' | 'fix'.")
    summary: str
    affected_parameter_ids: list[str] = Field(default_factory=list)
    affected_submodel_ids: list[str] = Field(default_factory=list)
    credited_to: list[str] = Field(
        default_factory=list,
        description="User ids of suggesters whose submissions led to this "
        "change. Credited in the public changelog.",
    )
    suggestion_ids: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# The MetaModel itself
# ---------------------------------------------------------------------------


META_MODEL_SCHEMA_VERSION = "0.1.0"


class MetaModel(BaseModel):
    """A citable, versioned meta-model for one scientific system.

    Instances of this class ARE the shared scientific artifacts — they get
    Zenodo DOIs, are referenced in publications, and travel with the
    community rather than with individual users.
    """

    id: str
    title: str
    scientific_domain: str = Field(
        description="Free-text domain label, e.g. 'microbial_biofilm', "
        "'lithium_ion_cathode_degradation'."
    )
    version: SemVer
    zenodo_doi: Optional[str] = Field(
        default=None,
        description="DOI of the current published version. None until first release.",
    )
    maintainers: list[str] = Field(
        default_factory=list,
        description="User ids with review privileges for community suggestions.",
    )

    # Scientific content
    reconciled_parameters: list[ReconciledParameter] = Field(default_factory=list)
    submodels: list[SubmodelEntry] = Field(default_factory=list)
    approximation_operators: list[ApproximationOperator] = Field(default_factory=list)

    # Operational status
    ingestion: IngestionStatus = Field(default_factory=IngestionStatus)
    changelog: list[ChangelogEntry] = Field(default_factory=list)

    schema_version: str = META_MODEL_SCHEMA_VERSION
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # --- lookups ---------------------------------------------------------

    def find_parameter(
        self, parameter_id: str, context_keys: Optional[dict[str, str]] = None
    ) -> Optional[ReconciledParameter]:
        """Return the reconciled parameter matching (id, context), or None.

        A record matches if its parameter_id is equal AND every key in
        ``context_keys`` (if supplied) matches the record's value for
        that key. Extra context keys on the record do not prevent a match.
        """
        ctx = context_keys or {}
        for rp in self.reconciled_parameters:
            if rp.parameter_id != parameter_id:
                continue
            if all(rp.context_keys.get(k) == v for k, v in ctx.items()):
                return rp
        return None

    def get_submodel(self, submodel_id: str) -> Optional[SubmodelEntry]:
        for s in self.submodels:
            if s.id == submodel_id:
                return s
        return None

    def get_operator(
        self, from_id: str, to_id: str
    ) -> Optional[ApproximationOperator]:
        for op in self.approximation_operators:
            if op.from_submodel_id == from_id and op.to_submodel_id == to_id:
                return op
        return None

    # --- invariants ------------------------------------------------------

    @model_validator(mode="after")
    def _check_invariants(self) -> "MetaModel":
        # Unique submodel ids
        ids = [s.id for s in self.submodels]
        if len(set(ids)) != len(ids):
            dupes = sorted({x for x in ids if ids.count(x) > 1})
            raise ValueError(f"duplicate submodel ids: {dupes}")

        id_set = set(ids)
        # Parent references valid
        for s in self.submodels:
            if s.parent_id is not None and s.parent_id not in id_set:
                raise ValueError(
                    f"submodel '{s.id}' parent_id '{s.parent_id}' not found"
                )

        # Approximation operator endpoints valid
        for op in self.approximation_operators:
            if op.from_submodel_id not in id_set:
                raise ValueError(
                    f"approximation operator '{op.id}' from_submodel_id "
                    f"'{op.from_submodel_id}' not found"
                )
            if op.to_submodel_id not in id_set:
                raise ValueError(
                    f"approximation operator '{op.id}' to_submodel_id "
                    f"'{op.to_submodel_id}' not found"
                )
            # A simplification should go from higher to lower complexity rank.
            frm = self.get_submodel(op.from_submodel_id)
            to = self.get_submodel(op.to_submodel_id)
            if frm and to and frm.complexity_rank <= to.complexity_rank:
                raise ValueError(
                    f"approximation operator '{op.id}' must go from higher "
                    f"to lower complexity_rank (got "
                    f"{frm.complexity_rank} -> {to.complexity_rank})"
                )

        # Unique reconciled-parameter entries: same (id, context) shouldn't
        # appear twice.
        keys = [
            (rp.parameter_id, tuple(sorted(rp.context_keys.items())))
            for rp in self.reconciled_parameters
        ]
        if len(set(keys)) != len(keys):
            raise ValueError(
                "duplicate reconciled_parameters for same (parameter_id, context)"
            )
        return self
