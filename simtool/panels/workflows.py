"""The three panel workflows.

1. ``recommend_model`` — match user constraints against the meta-model
   submodel hierarchy. When no pre-enumerated submodel fits, compose
   approximation operators to derive a custom simplification.

2. ``adjust_model`` — the user asks for a modification. Trace which
   meta-model components the modification depends on. Return an
   AdjustmentProposal with SUPPORTED / PARTIALLY_SUPPORTED / SPECULATIVE.

3. ``fit_data`` — calibrate the panel's model against experimental data.
   Compare fitted parameters to the meta-model consensus; when fitted
   values fall outside the reconciled range, produce a
   ``Suggestion`` the user can submit back to the community.

These are user-visible entry points. They return structured outputs
describing WHAT THE SYSTEM DID so the UI can render reasoning traces,
not just final answers.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from simtool.connector.ir import Distribution, ParameterBinding, ScientificModel
from simtool.metamodel.library import (
    ApproximationOperator,
    MetaModel,
    SubmodelEntry,
)
from simtool.metamodel.scope import (
    ScopeStatusReport,
    evaluate_ir_against_scope,
)
from simtool.metamodel.suggestions import (
    Evidence,
    EvidenceKind,
    Suggestion,
    SuggestionTargetKind,
)
from simtool.panels.panel import (
    OverrideSource,
    Panel,
    ParameterOverride,
    UserConstraints,
)


# ---------------------------------------------------------------------------
# Workflow 1: recommend_model
# ---------------------------------------------------------------------------


class RecommendationReasoningStep(BaseModel):
    kind: str = Field(description="'eligible' | 'rejected' | 'applied_operator'")
    submodel_id: str = ""
    operator_id: str = ""
    note: str


class ModelRecommendation(BaseModel):
    """Output of ``recommend_model``.

    ``submodel_id`` names the selected node (possibly the result of
    applying an approximation operator). ``derived_via`` is the operator
    id when the submodel was synthesized on demand, else None.
    ``reasoning`` is the full trace the UI can expand.
    """

    meta_model_id: str
    meta_model_version: str
    submodel_id: str
    derived_via_operator_id: Optional[str] = None
    unmet_constraints: list[str] = Field(default_factory=list)
    assumptions_introduced: list[str] = Field(default_factory=list)
    reasoning: list[RecommendationReasoningStep] = Field(default_factory=list)


def recommend_model(
    meta_model: MetaModel, constraints: UserConstraints
) -> ModelRecommendation:
    """Select (or derive) a submodel that fits ``constraints``.

    Selection rules (in order):
      1. Submodel must cover every ``required_phenomena``.
      2. Submodel must NOT cover any ``excluded_phenomena`` the user
         explicitly rejected (unless covering them is cheap).
      3. Among eligible submodels, pick the one with the lowest
         ``complexity_rank`` — simplest model that still answers the
         question.
      4. If no eligible submodel exists, find one that ALMOST fits and
         apply an approximation operator from the meta-model to derive
         a custom simpler variant. Record the operator in
         ``derived_via_operator_id`` and its introduced assumptions.
    """
    reasoning: list[RecommendationReasoningStep] = []
    candidates: list[SubmodelEntry] = []

    for s in meta_model.submodels:
        missing = [p for p in constraints.required_phenomena if p not in s.covered_phenomena]
        if missing:
            reasoning.append(RecommendationReasoningStep(
                kind="rejected", submodel_id=s.id,
                note=f"missing required phenomena: {missing}",
            ))
            continue
        has_excluded = [
            p for p in s.covered_phenomena
            if p in constraints.excluded_phenomena
        ]
        if has_excluded:
            reasoning.append(RecommendationReasoningStep(
                kind="rejected", submodel_id=s.id,
                note=f"covers excluded phenomena: {has_excluded}",
            ))
            continue
        reasoning.append(RecommendationReasoningStep(
            kind="eligible", submodel_id=s.id,
            note=f"covers required {constraints.required_phenomena}, "
                 f"rank={s.complexity_rank}",
        ))
        candidates.append(s)

    if candidates:
        chosen = min(candidates, key=lambda s: s.complexity_rank)
        return ModelRecommendation(
            meta_model_id=meta_model.id,
            meta_model_version=str(meta_model.version),
            submodel_id=chosen.id,
            reasoning=reasoning,
        )

    # No eligible submodel — try deriving via an approximation operator.
    for op in meta_model.approximation_operators:
        parent = meta_model.get_submodel(op.from_submodel_id)
        target = meta_model.get_submodel(op.to_submodel_id)
        if parent is None or target is None:
            continue
        missing_parent = [
            p for p in constraints.required_phenomena if p not in parent.covered_phenomena
        ]
        if missing_parent:
            continue
        excluded_target = [
            p for p in target.covered_phenomena if p in constraints.excluded_phenomena
        ]
        if excluded_target:
            continue
        reasoning.append(RecommendationReasoningStep(
            kind="applied_operator",
            submodel_id=target.id,
            operator_id=op.id,
            note=f"parent {parent.id} fits phenomena; reduced to {target.id} "
                 f"via {op.kind.value}",
        ))
        return ModelRecommendation(
            meta_model_id=meta_model.id,
            meta_model_version=str(meta_model.version),
            submodel_id=target.id,
            derived_via_operator_id=op.id,
            assumptions_introduced=op.assumptions_introduced,
            reasoning=reasoning,
        )

    # Truly no fit.
    return ModelRecommendation(
        meta_model_id=meta_model.id,
        meta_model_version=str(meta_model.version),
        submodel_id="",
        unmet_constraints=[
            f"no submodel covers required phenomena {constraints.required_phenomena} "
            f"without including excluded phenomena {constraints.excluded_phenomena}",
        ],
        reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# Workflow 2: adjust_model
# ---------------------------------------------------------------------------


class SupportLevel(str, Enum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    SPECULATIVE = "speculative"


class AdjustmentRequest(BaseModel):
    """User-facing request for a panel modification.

    ``kind`` is one of:
      - add_process:     ``target_id`` unused; spec in ``spec`` dict.
      - remove_process:  ``target_id`` = process id to remove.
      - change_parameter:``target_id`` = parameter id; new value in ``spec``.
      - add_entity:      as add_process.
      - change_scope:    widen/narrow required_phenomena on constraints.
    """

    kind: str
    target_id: str = ""
    spec: dict = Field(default_factory=dict)
    user_note: str = ""


class AdjustmentProposal(BaseModel):
    """Output of ``adjust_model``.

    ``support_level`` tells the user how confident the system is:
      - SUPPORTED: every component the adjustment touches is present in
        the meta-model with reconciled parameters.
      - PARTIALLY_SUPPORTED: some components are present; others require
        user input or broader-literature search.
      - SPECULATIVE: the adjustment extends beyond the meta-model. The
        system MAY suggest candidate DOIs from broader search but MUST
        NOT present any value as authoritative.
    """

    request: AdjustmentRequest
    support_level: SupportLevel
    depends_on_parameter_ids: list[str] = Field(default_factory=list)
    depends_on_submodel_ids: list[str] = Field(default_factory=list)
    meta_model_has: list[str] = Field(default_factory=list)
    meta_model_missing: list[str] = Field(default_factory=list)
    speculative_candidate_dois: list[str] = Field(default_factory=list)
    reasoning: str = ""


def adjust_model(
    panel: Panel,
    meta_model: MetaModel,
    request: AdjustmentRequest,
) -> AdjustmentProposal:
    """Trace which meta-model components a requested modification depends on
    and flag support level honestly.
    """
    proposal_params: list[str] = []
    proposal_submodels: list[str] = []
    reasoning_parts: list[str] = []

    if request.kind in {"add_process", "add_entity"}:
        # The caller describes what they want to add in request.spec.
        required_params = list(request.spec.get("required_parameter_ids", []))
        proposal_params = required_params
        reasoning_parts.append(
            f"{request.kind} requires parameters: {required_params}"
        )
    elif request.kind == "change_parameter":
        proposal_params = [request.target_id]
        reasoning_parts.append(f"change_parameter targets {request.target_id}")
    elif request.kind == "remove_process":
        reasoning_parts.append(f"remove_process targets {request.target_id}")
    elif request.kind == "change_scope":
        required = request.spec.get("required_phenomena", [])
        reasoning_parts.append(f"change_scope adds required phenomena: {required}")
        # Which submodels cover these?
        for s in meta_model.submodels:
            if all(p in s.covered_phenomena for p in required):
                proposal_submodels.append(s.id)
    else:
        return AdjustmentProposal(
            request=request,
            support_level=SupportLevel.SPECULATIVE,
            reasoning=f"unknown adjustment kind '{request.kind}' — treating "
                      "as speculative",
        )

    has: list[str] = []
    missing: list[str] = []
    for pid in proposal_params:
        # Check against meta-model reconciled_parameters; context comes from
        # the panel's constraints where possible.
        rec = meta_model.find_parameter(pid)
        if rec is not None:
            has.append(pid)
        else:
            missing.append(pid)

    if not proposal_params and proposal_submodels:
        support = SupportLevel.SUPPORTED
    elif missing and has:
        support = SupportLevel.PARTIALLY_SUPPORTED
    elif missing:
        support = SupportLevel.SPECULATIVE
    else:
        support = SupportLevel.SUPPORTED

    return AdjustmentProposal(
        request=request,
        support_level=support,
        depends_on_parameter_ids=proposal_params,
        depends_on_submodel_ids=proposal_submodels,
        meta_model_has=has,
        meta_model_missing=missing,
        reasoning=" | ".join(reasoning_parts),
    )


# ---------------------------------------------------------------------------
# Workflow 3: fit_data
# ---------------------------------------------------------------------------


@dataclass
class ExperimentalDataset:
    """Minimal shape for user-supplied data. The real fitting would read
    time series / spatial fields; we only need enough here to compute
    per-parameter fitted values and their uncertainty."""

    name: str
    observable_id: str
    time_s: list[float]
    value: list[float]
    note: str = ""


@dataclass
class FitCalibrationResult:
    """Pretend output of a calibration routine — enough structure to test
    the meta-model comparison logic. Real fitting pipelines would produce
    richer objects; the panel-side shape is what matters here."""

    fitted_parameter_id: str
    context_keys: dict[str, str] = field(default_factory=dict)
    fitted_value: float = 0.0
    fitted_unit: str = ""
    fit_uncertainty: float = 0.0
    fit_n_iterations: int = 0


class FitDataResult(BaseModel):
    """Output of ``fit_data``.

    For each fitted parameter, the panel records an override (with
    source=FITTED_FROM_DATA). Fitted values falling outside the
    meta-model's reconciled range produce a ``Suggestion`` stub the user
    can review and submit.
    """

    panel_id: str
    meta_model_id: str
    fitted_overrides: list[ParameterOverride] = Field(default_factory=list)
    within_consensus: list[str] = Field(
        default_factory=list,
        description="parameter_ids whose fits fall within the reconciled range.",
    )
    outside_consensus: list[str] = Field(default_factory=list)
    suggestions: list[Suggestion] = Field(
        default_factory=list,
        description="Suggestion objects ready for the user to review and "
        "submit back to the community.",
    )
    notes: list[str] = Field(default_factory=list)


def fit_data(
    panel: Panel,
    meta_model: MetaModel,
    datasets: list[ExperimentalDataset],
    fit_results: list[FitCalibrationResult],
    *,
    reviewer_confidence: float = 0.7,
) -> FitDataResult:
    """Apply ``fit_results`` to ``panel`` as overrides and compare each
    fitted value to the meta-model consensus.

    ``datasets`` is carried for provenance only — the real calibration
    routine reads them to produce ``fit_results``. We compare fitted
    values to meta-model ranges and generate suggestions where they
    diverge.
    """
    out = FitDataResult(panel_id=panel.id, meta_model_id=meta_model.id)
    dataset_names = ", ".join(d.name for d in datasets) or "(no named datasets)"

    for fit in fit_results:
        rec = meta_model.find_parameter(fit.fitted_parameter_id, fit.context_keys)
        binding = ParameterBinding(
            parameter_id=fit.fitted_parameter_id,
            canonical_unit=fit.fitted_unit or (
                rec.binding.canonical_unit if rec else "dimensionless"
            ),
            context_keys=fit.context_keys,
            point_estimate=fit.fitted_value,
        )
        justification = (
            f"fitted to experimental datasets: {dataset_names}; "
            f"fit uncertainty ~{fit.fit_uncertainty}"
        )
        override = ParameterOverride(
            parameter_id=fit.fitted_parameter_id,
            context_keys=fit.context_keys,
            original_binding=rec.binding if rec else None,
            override_binding=binding,
            source=OverrideSource.FITTED_FROM_DATA,
            justification=justification,
        )
        panel.set_override(override)
        out.fitted_overrides.append(override)

        if rec is None:
            out.notes.append(
                f"{fit.fitted_parameter_id}: meta-model had no consensus "
                "(MISSING) — fit cannot be compared"
            )
            continue

        within = _is_within_reconciled_range(fit.fitted_value, rec.binding)
        if within:
            out.within_consensus.append(fit.fitted_parameter_id)
        else:
            out.outside_consensus.append(fit.fitted_parameter_id)
            # Generate a Suggestion stub — user reviews and optionally submits.
            out.suggestions.append(
                Suggestion(
                    id=f"suggestion-{uuid.uuid4().hex[:8]}",
                    meta_model_id=meta_model.id,
                    meta_model_version_seen=str(meta_model.version),
                    target_kind=SuggestionTargetKind.PARAMETER,
                    target_id=fit.fitted_parameter_id,
                    target_context=fit.context_keys,
                    submitter_id=panel.user_id,
                    summary=(
                        f"fitted {fit.fitted_parameter_id} = "
                        f"{fit.fitted_value} (unit {binding.canonical_unit}) "
                        "falls outside current meta-model consensus"
                    ),
                    proposed_change=(
                        "expand reconciled distribution or add this point as "
                        "a new supporting record"
                    ),
                    evidence=[
                        Evidence(
                            kind=EvidenceKind.PERSONAL_DATA,
                            reference=dataset_names,
                            note=(
                                f"fit uncertainty {fit.fit_uncertainty}; "
                                f"{fit.fit_n_iterations} iterations"
                            ),
                        ),
                    ],
                    submitter_confidence=reviewer_confidence,
                )
            )
    return out


def _is_within_reconciled_range(value: float, binding: ParameterBinding) -> bool:
    """Is ``value`` within the meta-model's binding's support?

    For a point estimate we require +/- 20% (sane default; the real
    reconciler will attach a proper uncertainty range to the binding,
    at which point this heuristic goes away).
    """
    if binding.point_estimate is not None:
        p = binding.point_estimate
        if p == 0.0:
            return value == 0.0
        return abs(value - p) / abs(p) <= 0.20
    d = binding.distribution
    if d is None:
        return False
    if d.shape == "empirical" and d.samples:
        return min(d.samples) <= value <= max(d.samples)
    if d.shape == "uniform":
        return d.params["low"] <= value <= d.params["high"]
    if d.shape == "triangular":
        return d.params["low"] <= value <= d.params["high"]
    if d.shape == "normal":
        mean, std = d.params["mean"], d.params["stddev"]
        return abs(value - mean) <= 2.0 * std
    if d.shape == "lognormal":
        import math
        mu, sigma = d.params["mu"], d.params["sigma"]
        return abs(math.log(max(value, 1e-30)) - mu) <= 2.0 * sigma
    return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def evaluate_panel_readiness(
    panel: Panel, meta_model: MetaModel
) -> ScopeStatusReport:
    """Convenience: the scope status report for the panel's derived IR."""
    return evaluate_ir_against_scope(panel.derived_ir, meta_model)
