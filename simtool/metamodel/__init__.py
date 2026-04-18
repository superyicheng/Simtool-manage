"""Meta-model — shared scientific substrate.

A meta-model is the field's reconciled knowledge about a specific
scientific system, populated from the primary literature. One meta-model
per system. Community-maintained, SemVer'd, citable (Zenodo DOI),
updated on a recurring ingestion schedule.

Unity-with-IR: a meta-model is essentially an IR populated from
literature. Panels derive user-customized IRs from it. The connector
lowers those IRs to specific simulation frameworks.

Submodule map:
    library       - MetaModel, ReconciledParameter, SubmodelEntry,
                    ApproximationOperator, ChangelogEntry
    versioning    - SemVer, VersionChangeKind, PropagationPolicy
    scope         - ScopeContract, ParameterStatus, evaluate_ir_against_scope
    ingestion     - IngestionTier, PaperCandidate, IngestionJob,
                    IntegrationDecision, staleness helpers
    suggestions   - Suggestion, SuggestionLedger (community flow)

Cooperation note: this layer defines the META-MODEL SHAPE as a shared
artifact. The reconciler internals (how parameter_record.ParameterRecord
lists get collapsed into ReconciledParameter) live elsewhere and
populate this shape.
"""

from simtool.metamodel.ingestion import (
    IngestionJob,
    IngestionTier,
    IntegrationDecision,
    IntegrationDecisionKind,
    PaperCandidate,
    PaperSource,
    is_stale,
    staleness_warning,
)
from simtool.metamodel.library import (
    META_MODEL_SCHEMA_VERSION,
    ApproximationOperator,
    ApproximationOperatorKind,
    ChangelogEntry,
    IngestionCadence,
    IngestionStatus,
    MetaModel,
    QualityRating,
    ReconciledParameter,
    SubmodelEntry,
)
from simtool.metamodel.scope import (
    ParameterStatus,
    ParameterStatusReport,
    ScopeContract,
    ScopeStatusReport,
    evaluate_ir_against_scope,
    parameter_status,
)
from simtool.metamodel.suggestions import (
    Evidence,
    EvidenceKind,
    Suggestion,
    SuggestionLedger,
    SuggestionStatus,
    SuggestionTargetKind,
)
from simtool.metamodel.versioning import (
    PropagationPolicy,
    SemVer,
    VersionChangeKind,
    classify_change,
)

__all__ = [
    "META_MODEL_SCHEMA_VERSION",
    "ApproximationOperator",
    "ApproximationOperatorKind",
    "ChangelogEntry",
    "Evidence",
    "EvidenceKind",
    "IngestionCadence",
    "IngestionJob",
    "IngestionStatus",
    "IngestionTier",
    "IntegrationDecision",
    "IntegrationDecisionKind",
    "MetaModel",
    "PaperCandidate",
    "PaperSource",
    "ParameterStatus",
    "ParameterStatusReport",
    "PropagationPolicy",
    "QualityRating",
    "ReconciledParameter",
    "ScopeContract",
    "ScopeStatusReport",
    "SemVer",
    "SubmodelEntry",
    "Suggestion",
    "SuggestionLedger",
    "SuggestionStatus",
    "SuggestionTargetKind",
    "VersionChangeKind",
    "classify_change",
    "evaluate_ir_against_scope",
    "is_stale",
    "parameter_status",
    "staleness_warning",
]
