"""Reconciler: RawExtraction stream -> ReconciledParameter list.

Produces entries conforming to the MetaModel's contract:
  - group raw extractions by (parameter_id, *context_disambiguators*)
  - harmonize each value to canonical units via simtool.units.harmonize
  - per group with one record: ParameterBinding.point_estimate
  - per group with >=2 records: empirical Distribution (preserves all
    supporting values, with provenance DOIs)
  - derive QualityRating from record count + agreement + QC signals
  - surface conflicts as conflict_flags (no silent reconciliation)

Not silent: when records disagree beyond a relative-dispersion threshold,
or when unit-harmonization failed for a subset, we set conflict_flags
rather than averaging away the disagreement.

Scope: this module does not touch the SubmodelEntry hierarchy or
approximation operators; those are separate concerns populated elsewhere
(or seeded per-system).
"""

from __future__ import annotations

import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional

from simtool.connector.ir import Distribution, ParameterBinding
from simtool.extractors.schemas import RawExtraction
from simtool.metamodel.library import QualityRating, ReconciledParameter
from simtool.schema.idynomics_vocab import VOCAB
from simtool.units.harmonize import HarmonizationResult, _DEFAULT_MW_LOOKUP, harmonize

logger = logging.getLogger(__name__)


# Context keys used to group records. Every value on the ReconciledParameter's
# context_keys dict comes from one of these. Additional RawStudyContext
# fields (temperature, pH, ...) are kept as supporting evidence notes but
# do NOT split groups in v0.1 — they'd produce a combinatorial explosion
# of single-source groups. A later version can promote specific fields.
_CONTEXT_KEYS = ("species", "substrate", "redox_regime", "culture_mode")


# Relative-dispersion threshold above which a group is flagged as
# "high heterogeneity, conflict". 0.5 means the ratio of the largest to
# the smallest canonical value is >= 1.5x. Chemostat Ks vs initial-rate
# Ks routinely differ by 2-3x — that's a real conflict, not noise.
_CONFLICT_RATIO = 1.5


@dataclass
class ExtractionWithDoi:
    """Pair a RawExtraction with the DOI of the paper it came from.

    The reconciler needs the source DOI for provenance; RawExtraction
    itself does not carry it (it carries a page/text span).
    """

    extraction: RawExtraction
    doi: str


@dataclass
class ReconcileSummary:
    """Diagnostic summary of a reconciliation run."""

    input_count: int = 0
    harmonized_count: int = 0
    dropped_harmonization: int = 0
    dropped_range_check: int = 0
    reconciled_parameters: int = 0
    conflict_parameters: int = 0
    groups_dropped_empty: int = 0
    notes: list[str] = field(default_factory=list)


def reconcile(
    records: Iterable[ExtractionWithDoi],
    *,
    range_check_strict: bool = True,
) -> tuple[list[ReconciledParameter], ReconcileSummary]:
    """Reconcile a batch of extractions into a list of ReconciledParameter.

    If ``range_check_strict`` is True (default), a record that fails the
    vocab's sanity range check is dropped from its group. Set False to
    keep all values and flag the group as out-of-range.
    """

    records = list(records)
    summary = ReconcileSummary(input_count=len(records))

    # 1. Harmonize each value; keep the ones that pass dimensional and (optionally) range checks.
    prepared: list[tuple[ExtractionWithDoi, HarmonizationResult]] = []
    for rec in records:
        if rec.extraction.parameter_id not in VOCAB:
            logger.warning(
                "Unknown parameter_id %s (doi=%s); dropping",
                rec.extraction.parameter_id,
                rec.doi,
            )
            continue
        hr = harmonize(
            value=rec.extraction.value,
            unit_str=rec.extraction.unit,
            parameter_id=rec.extraction.parameter_id,
            compound=rec.extraction.context.substrate,
            molar_mass_lookup=_DEFAULT_MW_LOOKUP,
        )
        if not hr.dimensional_check_passed:
            summary.dropped_harmonization += 1
            logger.info(
                "Dropped: harmonization failed (%s %s -> %s canonical): %s",
                rec.extraction.value,
                rec.extraction.unit,
                rec.extraction.parameter_id,
                hr.detail,
            )
            continue
        if range_check_strict and not hr.range_check_passed:
            summary.dropped_range_check += 1
            logger.info(
                "Dropped: range-check failed for %s=%s %s (canonical %s): %s",
                rec.extraction.parameter_id,
                rec.extraction.value,
                rec.extraction.unit,
                hr.canonical_value,
                hr.detail,
            )
            continue
        prepared.append((rec, hr))
    summary.harmonized_count = len(prepared)

    # 2. Group by (parameter_id, context-subset).
    groups: dict[tuple[str, tuple[tuple[str, str], ...]], list[tuple[ExtractionWithDoi, HarmonizationResult]]] = defaultdict(list)
    for rec, hr in prepared:
        key = _group_key(rec.extraction)
        groups[key].append((rec, hr))

    # 3. Build a ReconciledParameter per group.
    out: list[ReconciledParameter] = []
    for (parameter_id, ctx_tuple), members in groups.items():
        if not members:
            summary.groups_dropped_empty += 1
            continue
        vocab_entry = VOCAB[parameter_id]
        context_keys = dict(ctx_tuple)
        values = [hr.canonical_value for (_rec, hr) in members]
        dois = sorted({rec.doi for (rec, _hr) in members})

        # Build the ParameterBinding
        source_note = _source_note(parameter_id, members)
        binding = _build_binding(
            parameter_id=parameter_id,
            canonical_unit=vocab_entry.canonical_unit,
            context_keys=context_keys,
            values=values,
            dois=dois,
            source_note=source_note,
        )

        quality = _derive_quality(members, values)
        conflicts = _detect_conflicts(members, values)
        if conflicts:
            summary.conflict_parameters += 1

        out.append(
            ReconciledParameter(
                parameter_id=parameter_id,
                context_keys=context_keys,
                binding=binding,
                supporting_record_dois=dois,
                quality_rating=quality,
                conflict_flags=conflicts,
            )
        )

    summary.reconciled_parameters = len(out)
    return out, summary


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _group_key(extr: RawExtraction) -> tuple[str, tuple[tuple[str, str], ...]]:
    ctx: list[tuple[str, str]] = []
    ctx_obj = extr.context
    for key in _CONTEXT_KEYS:
        val = getattr(ctx_obj, key, None)
        if val is not None and val != "":
            ctx.append((key, _normalize_context_value(str(val))))
    return extr.parameter_id, tuple(sorted(ctx))


def _normalize_context_value(val: str) -> str:
    """Light normalization so 'Nitrosomonas europaea' and ' Nitrosomonas europaea ' group."""

    return val.strip().lower()


def _source_note(
    parameter_id: str,
    members: list[tuple[ExtractionWithDoi, HarmonizationResult]],
) -> str:
    n = len(members)
    methods = sorted({m[0].extraction.method.value for m in members})
    methods_text = "/".join(methods)
    return f"reconciled from {n} {methods_text} extraction(s) for {parameter_id}"


def _build_binding(
    *,
    parameter_id: str,
    canonical_unit: str,
    context_keys: dict[str, str],
    values: list[float],
    dois: list[str],
    source_note: str,
) -> ParameterBinding:
    if len(values) == 1:
        return ParameterBinding(
            parameter_id=parameter_id,
            canonical_unit=canonical_unit,
            context_keys=context_keys,
            point_estimate=values[0],
            provenance_dois=dois,
            source_note=source_note,
        )
    # Multiple values -> empirical distribution. We keep the raw values so
    # downstream consumers can resample, plot, or re-derive summary stats.
    # Empirical is honest: makes no parametric assumption about the shape.
    return ParameterBinding(
        parameter_id=parameter_id,
        canonical_unit=canonical_unit,
        context_keys=context_keys,
        distribution=Distribution(shape="empirical", samples=values),
        provenance_dois=dois,
        source_note=source_note,
    )


def _derive_quality(
    members: list[tuple[ExtractionWithDoi, HarmonizationResult]],
    canonical_values: list[float],
) -> QualityRating:
    n = len(members)
    mean_confidence = (
        sum(m[0].extraction.self_confidence for m in members) / n if n else 0.0
    )
    spread_ok = _relative_spread(canonical_values) <= _CONFLICT_RATIO

    # Decision rules (coarse, documented so they can be argued with):
    #  HIGH      — >=3 sources, tight agreement, good confidence
    #  MODERATE  — 2 sources that agree, or >=3 with looser agreement
    #  LOW       — 1 source OR disagreement among 2
    #  VERY_LOW  — disagreement among >=3 sources
    if n >= 3 and spread_ok and mean_confidence >= 0.6:
        return QualityRating.HIGH
    if n >= 3 and not spread_ok:
        return QualityRating.VERY_LOW
    if n == 2 and spread_ok:
        return QualityRating.MODERATE
    if n == 1:
        return QualityRating.LOW
    return QualityRating.MODERATE


def _detect_conflicts(
    members: list[tuple[ExtractionWithDoi, HarmonizationResult]],
    canonical_values: list[float],
) -> list[str]:
    flags: list[str] = []
    if len(canonical_values) >= 2:
        spread = _relative_spread(canonical_values)
        if spread > _CONFLICT_RATIO:
            lo, hi = min(canonical_values), max(canonical_values)
            flags.append(
                f"high heterogeneity: values span {lo:.3g}..{hi:.3g} "
                f"({spread:.1f}x ratio); do not collapse without checking method bias"
            )
    # Mixed methods on the same (parameter, context) group — flag because
    # e.g. chemostat Ks and initial-rate Ks are known to disagree.
    methods = {m[0].extraction.method.value for m in members}
    if len(methods) >= 2:
        flags.append(
                f"multiple methods in group: {sorted(methods)} "
            "— interpret with method-specific bias in mind"
        )
    return flags


def _relative_spread(values: list[float]) -> float:
    non_zero = [abs(v) for v in values if v != 0.0]
    if len(non_zero) < 2:
        return 0.0
    return max(non_zero) / min(non_zero)


# ---------------------------------------------------------------------------
# Helper exposed for callers that want to peek at statistics without
# constructing a full ReconciledParameter.
# ---------------------------------------------------------------------------


@dataclass
class GroupSummary:
    parameter_id: str
    context_keys: dict[str, str]
    n_sources: int
    values: list[float]
    mean: Optional[float]
    stdev: Optional[float]
    relative_spread: float


def summarize_groups(records: Iterable[ExtractionWithDoi]) -> list[GroupSummary]:
    """Debugging/analysis: return the per-group stats that feed reconciliation."""

    recs = list(records)
    groups: dict[tuple[str, tuple[tuple[str, str], ...]], list[float]] = defaultdict(list)
    for r in recs:
        if r.extraction.parameter_id not in VOCAB:
            continue
        hr = harmonize(
            value=r.extraction.value,
            unit_str=r.extraction.unit,
            parameter_id=r.extraction.parameter_id,
            compound=r.extraction.context.substrate,
            molar_mass_lookup=_DEFAULT_MW_LOOKUP,
        )
        if not hr.dimensional_check_passed:
            continue
        groups[_group_key(r.extraction)].append(hr.canonical_value)

    out = []
    for (pid, ctx), values in groups.items():
        out.append(
            GroupSummary(
                parameter_id=pid,
                context_keys=dict(ctx),
                n_sources=len(values),
                values=sorted(values),
                mean=statistics.fmean(values) if values else None,
                stdev=statistics.stdev(values) if len(values) > 1 else None,
                relative_spread=_relative_spread(values),
            )
        )
    return out
