"""Compare simulation outputs to meta-model ranges.

The basic question: does the final value of each observable fall within
a defensible range given the literature? We don't have per-observable
reconciled ranges (that would be a separate meta-model surface), so
this first pass compares observables to ranges the IR's parameter
bindings imply via their distributions, falling back to a percent-of-
consensus tolerance when only a point estimate is available.

This is intentionally conservative: when there's no meta-model coverage
for an observable, the comparison is marked UNKNOWN rather than being
silently reported as "within range".
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from simtool.connector.ir import Distribution, ParameterBinding, ScientificModel
from simtool.connector.runs import OutputBundle
from simtool.metamodel import MetaModel, ReconciledParameter


class ComparisonOutcome(str, Enum):
    WITHIN = "within"
    ABOVE = "above"
    BELOW = "below"
    UNKNOWN = "unknown"


class ObservableComparison(BaseModel):
    observable_id: str
    final_value: Optional[float] = None
    reference_low: Optional[float] = None
    reference_high: Optional[float] = None
    reference_source: str = ""
    outcome: ComparisonOutcome = ComparisonOutcome.UNKNOWN
    note: str = ""


class ComparisonReport(BaseModel):
    ir_id: str
    meta_model_id: str
    meta_model_version: str
    observables: list[ObservableComparison] = Field(default_factory=list)

    @property
    def any_outside_range(self) -> bool:
        return any(
            o.outcome in {ComparisonOutcome.ABOVE, ComparisonOutcome.BELOW}
            for o in self.observables
        )


def compare_outputs_to_metamodel(
    bundle: OutputBundle,
    ir: ScientificModel,
    meta_model: MetaModel,
) -> ComparisonReport:
    """Build a comparison report for every observable in the IR."""
    comparisons: list[ObservableComparison] = []
    for obs in ir.observables:
        series = bundle.scalar_time_series.get(obs.id)
        if not series:
            comparisons.append(ObservableComparison(
                observable_id=obs.id,
                note="no scalar time series found in outputs",
            ))
            continue
        final_value = series[-1][1]
        # Find a meta-model range matching the observable target.
        low, high, src = _reference_range(obs.target, meta_model)
        if low is None or high is None:
            comparisons.append(ObservableComparison(
                observable_id=obs.id,
                final_value=final_value,
                outcome=ComparisonOutcome.UNKNOWN,
                note=f"no meta-model reference for target '{obs.target}'",
            ))
            continue
        if final_value < low:
            outcome = ComparisonOutcome.BELOW
        elif final_value > high:
            outcome = ComparisonOutcome.ABOVE
        else:
            outcome = ComparisonOutcome.WITHIN
        comparisons.append(ObservableComparison(
            observable_id=obs.id,
            final_value=final_value,
            reference_low=low, reference_high=high,
            reference_source=src,
            outcome=outcome,
        ))
    return ComparisonReport(
        ir_id=ir.id,
        meta_model_id=meta_model.id,
        meta_model_version=str(meta_model.version),
        observables=comparisons,
    )


# ---------------------------------------------------------------------------
# Reference-range resolution
# ---------------------------------------------------------------------------


def _reference_range(
    target: str, meta_model: MetaModel,
) -> tuple[Optional[float], Optional[float], str]:
    """Pick a reference range for an observable target.

    Heuristic: split target on '_' or '@', look for a ReconciledParameter
    whose parameter_id matches one of the fragments. This handles e.g.
    ``biofilm_thickness`` falling back to no match, or ``mu_max@AOB``
    matching the mu_max reconciled entry.
    """
    fragments = {s for chunk in target.split("@") for s in chunk.split("_")}
    fragments |= {target}
    best: Optional[ReconciledParameter] = None
    for rp in meta_model.reconciled_parameters:
        if rp.parameter_id in fragments:
            best = rp
            break
    if best is None:
        return None, None, ""
    return _binding_range(best.binding)


def _binding_range(
    binding: ParameterBinding,
) -> tuple[Optional[float], Optional[float], str]:
    if binding.distribution is not None:
        lo, hi = _distribution_range(binding.distribution)
        return lo, hi, f"distribution:{binding.distribution.shape}"
    if binding.point_estimate is not None:
        p = binding.point_estimate
        # +/- 30% band around a point estimate as a crude "consensus" envelope.
        return p * 0.7, p * 1.3, "point_estimate +/-30%"
    return None, None, ""


def _distribution_range(d: Distribution) -> tuple[Optional[float], Optional[float]]:
    if d.shape == "empirical" and d.samples:
        return min(d.samples), max(d.samples)
    if d.shape == "uniform":
        return d.params["low"], d.params["high"]
    if d.shape == "triangular":
        return d.params["low"], d.params["high"]
    if d.shape == "normal":
        return (
            d.params["mean"] - 2 * d.params["stddev"],
            d.params["mean"] + 2 * d.params["stddev"],
        )
    if d.shape == "lognormal":
        import math
        center = math.exp(d.params["mu"])
        return (
            math.exp(d.params["mu"] - 2 * d.params["sigma"]),
            math.exp(d.params["mu"] + 2 * d.params["sigma"]),
        )
    return None, None
