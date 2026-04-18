"""Test-scaffolding helpers for the meta-model -> IR seam.

This file is TEST INFRASTRUCTURE, not production code. The real reconciler
lives elsewhere (currently being built by a different session). What's
here is the minimum shape needed so canonical integration tests can run
end-to-end: given a list of ``ParameterRecord`` objects (the extractor's
output), produce a ``ParameterBinding`` (the IR's input).

When the real reconciler lands, these helpers go away and the canonical
tests switch to it without changing shape.

The reduction rules here are deliberately simple:
  - 1 record  -> point estimate.
  - 2+ records that agree within 20% -> point estimate at the mean.
  - 2+ records disagreeing more than 20% -> empirical distribution with a
    ``conflict`` flag folded into ``source_note``.
  - No canonical-unit agreement -> skip record, warn.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Optional

from simtool.connector.ir import Distribution, ParameterBinding
from simtool.schema.idynomics_vocab import VOCAB, VocabEntry
from simtool.schema.parameter_record import (
    ExtractionModality,
    ExtractorAgreement,
    GradeRating,
    MeasurementMethod,
    ParameterRecord,
    SpanAnchor,
    StudyContext,
)


# ---------------------------------------------------------------------------
# Factory — build a ParameterRecord with sensible defaults
# ---------------------------------------------------------------------------


def make_record(
    *,
    parameter_id: str,
    value: float,
    unit: str,
    canonical_value: Optional[float] = None,
    doi: str,
    citation: str = "",
    species: Optional[str] = None,
    substrate: Optional[str] = None,
    temperature_c: Optional[float] = None,
    method: MeasurementMethod = MeasurementMethod.CHEMOSTAT,
    grade: GradeRating = GradeRating.MODERATE,
    dimensional_check_passed: bool = True,
    range_check_passed: bool = True,
    confidence: float = 0.8,
) -> ParameterRecord:
    """Minimal factory for test records — defaults cover the 90% case."""
    canonical = canonical_value if canonical_value is not None else value
    return ParameterRecord(
        parameter_id=parameter_id,
        value=value,
        unit=unit,
        canonical_value=canonical,
        citation=citation or f"Paper ({doi})",
        doi=doi,
        span=SpanAnchor(
            doi=doi, page=1, text_excerpt="...", modality=ExtractionModality.PROSE
        ),
        method=method,
        context=StudyContext(
            species=species, substrate=substrate, temperature_c=temperature_c
        ),
        extractor_agreement=ExtractorAgreement(n_extractors=2, n_agreed=2),
        dimensional_check_passed=dimensional_check_passed,
        range_check_passed=range_check_passed,
        extractor_confidence=confidence,
        grade=grade,
    )


# ---------------------------------------------------------------------------
# Reconciliation — a stub that mirrors the real reconciler's output shape
# ---------------------------------------------------------------------------


@dataclass
class ReconciliationResult:
    binding: Optional[ParameterBinding]
    """None when the records cannot be reconciled (e.g. failed dimensional
    checks, no canonical values). A structured 'I don't know', not garbage."""

    warnings: list[str] = field(default_factory=list)
    """Free-text warnings the reconciler raised. Surfaced to the user."""

    conflict_flags: list[str] = field(default_factory=list)
    """Specific conflicts detected between records."""


def reconcile_records(
    records: list[ParameterRecord],
    *,
    disagreement_threshold: float = 0.20,
) -> ReconciliationResult:
    """Reduce a list of ``ParameterRecord``s (all for the same logical
    parameter+context) to a single ``ParameterBinding``.

    This is the TEST stub. The real reconciler will be richer (GRADE
    weighting, context-aware stratification, method-bias correction) —
    but its *output shape* is the same ``ParameterBinding``, so canonical
    tests that depend on this shape will not need to change.
    """
    warnings: list[str] = []
    conflict_flags: list[str] = []

    if not records:
        return ReconciliationResult(
            binding=None, warnings=["no records supplied"]
        )

    pid = records[0].parameter_id
    if any(r.parameter_id != pid for r in records):
        return ReconciliationResult(
            binding=None,
            warnings=["records span multiple parameter_ids — split by id first"],
        )

    # Filter: records must have passed dimensional check + carry canonical value.
    usable = [
        r for r in records
        if r.dimensional_check_passed and r.canonical_value is not None
    ]
    skipped = len(records) - len(usable)
    if skipped:
        warnings.append(
            f"{skipped} of {len(records)} records dropped: failed "
            "dimensional check or missing canonical value"
        )

    if not usable:
        return ReconciliationResult(
            binding=None,
            warnings=warnings + [
                "no usable records after dimensional/canonical-unit filter"
            ],
        )

    vocab: VocabEntry = VOCAB[pid]
    provenance_dois = sorted({r.doi for r in usable})
    values = [r.canonical_value for r in usable]
    context_keys = _merge_context(usable)

    if len(values) == 1:
        return ReconciliationResult(
            binding=ParameterBinding(
                parameter_id=pid,
                canonical_unit=vocab.canonical_unit,
                context_keys=context_keys,
                point_estimate=values[0],
                provenance_dois=provenance_dois,
                source_note=f"single record ({provenance_dois[0]})",
            ),
            warnings=warnings,
        )

    mean = statistics.fmean(values)
    spread = (max(values) - min(values)) / mean if mean > 0 else float("inf")
    if spread <= disagreement_threshold:
        return ReconciliationResult(
            binding=ParameterBinding(
                parameter_id=pid,
                canonical_unit=vocab.canonical_unit,
                context_keys=context_keys,
                point_estimate=mean,
                provenance_dois=provenance_dois,
                source_note=(
                    f"mean of {len(values)} records, spread "
                    f"{spread * 100:.1f}% <= threshold"
                ),
            ),
            warnings=warnings,
        )

    # Disagreement beyond threshold — surface, do not collapse.
    conflict_flags.append(
        f"{pid}: {len(values)} records disagree "
        f"({min(values):.3g}..{max(values):.3g}, spread {spread * 100:.0f}%)"
    )
    return ReconciliationResult(
        binding=ParameterBinding(
            parameter_id=pid,
            canonical_unit=vocab.canonical_unit,
            context_keys=context_keys,
            distribution=Distribution(shape="empirical", samples=list(values)),
            provenance_dois=provenance_dois,
            source_note=(
                f"empirical distribution over {len(values)} conflicting "
                f"records; spread {spread * 100:.0f}%"
            ),
        ),
        warnings=warnings,
        conflict_flags=conflict_flags,
    )


def _merge_context(records: list[ParameterRecord]) -> dict[str, str]:
    """Keep only context keys that all records agree on."""
    context_keys: dict[str, str] = {}
    first_ctx = records[0].context
    for key in ("species", "substrate"):
        val = getattr(first_ctx, key)
        if val is None:
            continue
        if all(getattr(r.context, key) == val for r in records):
            context_keys[key] = val
    return context_keys
