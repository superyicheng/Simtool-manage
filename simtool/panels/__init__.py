"""Panel — user-specific workspace over a meta-model.

Each user's work happens in panels. A panel is a derived IR + user
constraints + parameter overrides + assumption ledger + run history,
pinned to a specific meta-model version. Private by default; forkable;
freezable for publication reproducibility.

Three workflows drive panel mutation:
    recommend_model(meta_model, constraints)  -> ModelRecommendation
    adjust_model(panel, meta_model, request)  -> AdjustmentProposal
    fit_data(panel, meta_model, datasets, fit_results) -> FitDataResult
"""

from simtool.panels.panel import (
    PANEL_SCHEMA_VERSION,
    MeasurementCapability,
    OverrideSource,
    Panel,
    ParameterOverride,
    PropagationOutcome,
    PublicationState,
    RunHistoryEntry,
    UserConstraints,
    propagate_to_version,
)
from simtool.panels.workflows import (
    AdjustmentProposal,
    AdjustmentRequest,
    ExperimentalDataset,
    FitCalibrationResult,
    FitDataResult,
    ModelRecommendation,
    RecommendationReasoningStep,
    SupportLevel,
    adjust_model,
    evaluate_panel_readiness,
    fit_data,
    recommend_model,
)

__all__ = [
    "PANEL_SCHEMA_VERSION",
    "AdjustmentProposal",
    "AdjustmentRequest",
    "ExperimentalDataset",
    "FitCalibrationResult",
    "FitDataResult",
    "MeasurementCapability",
    "ModelRecommendation",
    "OverrideSource",
    "Panel",
    "ParameterOverride",
    "PropagationOutcome",
    "PublicationState",
    "RecommendationReasoningStep",
    "RunHistoryEntry",
    "SupportLevel",
    "UserConstraints",
    "adjust_model",
    "evaluate_panel_readiness",
    "fit_data",
    "propagate_to_version",
    "recommend_model",
]
