"""Reconciler unit tests — pure logic, no network."""

from __future__ import annotations

import pytest

from simtool.extractors.schemas import RawExtraction, RawSpanAnchor, RawStudyContext
from simtool.metamodel.library import QualityRating
from simtool.metamodel.reconciler import (
    ExtractionWithDoi,
    reconcile,
    summarize_groups,
)
from simtool.schema.parameter_record import ExtractionModality, MeasurementMethod


def _ex(
    *,
    parameter_id: str = "mu_max",
    value: float = 1.0,
    unit: str = "1/day",
    method: MeasurementMethod = MeasurementMethod.CHEMOSTAT,
    species: str | None = "Nitrosomonas europaea",
    substrate: str | None = "NH4",
    page: int = 1,
    self_confidence: float = 0.85,
) -> RawExtraction:
    return RawExtraction(
        parameter_id=parameter_id,
        value=value,
        unit=unit,
        method=method,
        context=RawStudyContext(species=species, substrate=substrate),
        span=RawSpanAnchor(
            page=page,
            text_excerpt="excerpt",
            modality=ExtractionModality.PROSE,
        ),
        self_confidence=self_confidence,
    )


def _wrapped(ex: RawExtraction, doi: str) -> ExtractionWithDoi:
    return ExtractionWithDoi(extraction=ex, doi=doi)


def test_single_record_yields_point_estimate():
    recs = [_wrapped(_ex(value=1.2, unit="1/day"), "10.1234/a")]
    out, summary = reconcile(recs)
    assert len(out) == 1
    rp = out[0]
    assert rp.binding.point_estimate == 1.2
    assert rp.binding.distribution is None
    assert rp.supporting_record_dois == ["10.1234/a"]
    assert rp.quality_rating == QualityRating.LOW
    assert rp.conflict_flags == []
    assert summary.harmonized_count == 1


def test_multiple_records_yield_empirical_distribution():
    recs = [
        _wrapped(_ex(value=1.0), "10.1234/a"),
        _wrapped(_ex(value=1.1), "10.1234/b"),
        _wrapped(_ex(value=1.3), "10.1234/c"),
    ]
    out, _ = reconcile(recs)
    assert len(out) == 1
    rp = out[0]
    assert rp.binding.point_estimate is None
    assert rp.binding.distribution is not None
    assert rp.binding.distribution.shape == "empirical"
    assert sorted(rp.binding.distribution.samples) == [1.0, 1.1, 1.3]
    assert len(rp.supporting_record_dois) == 3
    assert rp.quality_rating == QualityRating.HIGH


def test_unit_harmonization_applies():
    # 0.05 /hr == 1.2 /day
    recs = [
        _wrapped(_ex(value=0.05, unit="1/hour"), "10.1234/a"),
        _wrapped(_ex(value=1.2, unit="1/day"), "10.1234/b"),
    ]
    out, _ = reconcile(recs)
    assert len(out) == 1
    rp = out[0]
    samples = rp.binding.distribution.samples
    assert all(abs(s - 1.2) < 1e-6 for s in samples)


def test_incompatible_unit_is_dropped():
    recs = [
        _wrapped(_ex(value=0.05, unit="1/hour"), "10.1234/a"),
        _wrapped(_ex(value=1.0, unit="kilogram"), "10.1234/b"),  # not a rate
    ]
    out, summary = reconcile(recs)
    # Dropped record should not poison the group — single valid record yields a point estimate
    assert len(out) == 1
    assert out[0].binding.point_estimate is not None
    assert summary.dropped_harmonization == 1


def test_range_check_drops_absurd_values_by_default():
    recs = [
        _wrapped(_ex(value=1.2, unit="1/day"), "10.1234/a"),
        _wrapped(_ex(value=9999.0, unit="1/day"), "10.1234/b"),  # outside sanity range
    ]
    out, summary = reconcile(recs)
    assert summary.dropped_range_check == 1
    assert len(out) == 1
    assert out[0].binding.point_estimate == 1.2


def test_conflict_flag_when_values_disagree_widely():
    # 3x spread between min and max — above the 1.5x conflict ratio
    recs = [
        _wrapped(_ex(value=1.0), "10.1234/a"),
        _wrapped(_ex(value=3.5), "10.1234/b"),
    ]
    out, summary = reconcile(recs)
    assert summary.conflict_parameters == 1
    rp = out[0]
    assert any("high heterogeneity" in f for f in rp.conflict_flags)


def test_conflict_flag_when_methods_differ():
    recs = [
        _wrapped(_ex(value=1.0, method=MeasurementMethod.CHEMOSTAT), "10.1234/a"),
        _wrapped(_ex(value=1.1, method=MeasurementMethod.BATCH_FIT), "10.1234/b"),
    ]
    out, _ = reconcile(recs)
    rp = out[0]
    # Values agree (no heterogeneity flag) but methods differ (method-bias flag)
    assert any("multiple methods" in f for f in rp.conflict_flags)
    assert not any("high heterogeneity" in f for f in rp.conflict_flags)


def test_grouping_by_species_and_substrate():
    recs = [
        _wrapped(_ex(species="Nitrosomonas", substrate="NH4", value=1.2), "10.1/a"),
        _wrapped(_ex(species="Nitrosomonas", substrate="NH4", value=1.3), "10.1/b"),
        _wrapped(_ex(species="Nitrobacter", substrate="NO2", value=0.8), "10.1/c"),
    ]
    out, _ = reconcile(recs)
    assert len(out) == 2
    by_species = {rp.context_keys["species"]: rp for rp in out}
    assert "nitrosomonas" in by_species
    assert "nitrobacter" in by_species
    assert by_species["nitrobacter"].binding.point_estimate == 0.8


def test_context_value_normalization_merges_whitespace_and_case():
    recs = [
        _wrapped(_ex(species="Nitrosomonas europaea", value=1.2), "10.1/a"),
        _wrapped(_ex(species=" nitrosomonas EUROPAEA ", value=1.3), "10.1/b"),
    ]
    out, _ = reconcile(recs)
    # Both records should group together despite whitespace/case differences
    assert len(out) == 1
    assert out[0].binding.distribution.shape == "empirical"


def test_quality_very_low_for_wide_disagreement_with_three_sources():
    recs = [
        _wrapped(_ex(value=0.5), "10.1/a"),
        _wrapped(_ex(value=2.0), "10.1/b"),
        _wrapped(_ex(value=5.0), "10.1/c"),
    ]
    out, _ = reconcile(recs)
    assert out[0].quality_rating == QualityRating.VERY_LOW


def test_unknown_parameter_id_dropped():
    # Build a RawExtraction with a bogus parameter_id via dict path
    ex = RawExtraction(
        parameter_id="not_a_real_param",
        value=1.0,
        unit="1/day",
        method=MeasurementMethod.CHEMOSTAT,
        context=RawStudyContext(),
        span=RawSpanAnchor(page=1, text_excerpt="x", modality=ExtractionModality.PROSE),
        self_confidence=0.5,
    )
    out, summary = reconcile([_wrapped(ex, "10.1/a")])
    assert out == []


def test_summarize_groups_reports_spread():
    recs = [
        _wrapped(_ex(value=1.0), "10.1/a"),
        _wrapped(_ex(value=2.0), "10.1/b"),
    ]
    summaries = summarize_groups(recs)
    assert len(summaries) == 1
    s = summaries[0]
    assert s.n_sources == 2
    assert s.values == [1.0, 2.0]
    assert s.relative_spread == pytest.approx(2.0)
