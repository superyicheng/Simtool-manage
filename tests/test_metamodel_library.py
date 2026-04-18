"""Unit tests — MetaModel invariants + lookups + submodel hierarchy."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from simtool.connector.ir import ParameterBinding
from simtool.metamodel import (
    ApproximationOperator,
    ApproximationOperatorKind,
    ChangelogEntry,
    MetaModel,
    QualityRating,
    ReconciledParameter,
    SemVer,
    SubmodelEntry,
)

from tests._metamodel_fixtures import make_nitrifying_biofilm_metamodel


# --- Lookups ---------------------------------------------------------------


def test_find_parameter_by_id_and_context() -> None:
    m = make_nitrifying_biofilm_metamodel()
    r = m.find_parameter("mu_max", {"species": "AOB"})
    assert r is not None
    assert r.binding.point_estimate == pytest.approx(1.0)


def test_find_parameter_returns_none_on_miss() -> None:
    m = make_nitrifying_biofilm_metamodel()
    assert m.find_parameter("ghost_param") is None
    assert m.find_parameter("mu_max", {"species": "ghost"}) is None


def test_find_parameter_partial_context_match() -> None:
    """Context keys in the query that match the record should succeed;
    extra keys on the record don't block matching."""
    m = make_nitrifying_biofilm_metamodel()
    r = m.find_parameter("K_s", {"species": "AOB"})
    assert r is not None


def test_get_submodel() -> None:
    m = make_nitrifying_biofilm_metamodel()
    s = m.get_submodel("continuum_pde_2d")
    assert s is not None
    assert s.complexity_rank == 2
    assert m.get_submodel("ghost") is None


def test_get_operator() -> None:
    m = make_nitrifying_biofilm_metamodel()
    op = m.get_operator("agent_based_3d", "continuum_pde_2d")
    assert op is not None
    assert op.kind == ApproximationOperatorKind.MEAN_FIELD_CLOSURE
    assert m.get_operator("a", "b") is None


# --- Invariant: unique submodel ids ---------------------------------------


def test_duplicate_submodel_ids_rejected() -> None:
    s1 = SubmodelEntry(id="x", name="x", complexity_rank=0, ir_template_ref="t")
    s2 = SubmodelEntry(id="x", name="x2", complexity_rank=1, ir_template_ref="t")
    with pytest.raises(ValidationError, match="duplicate submodel ids"):
        MetaModel(
            id="m", title="m", scientific_domain="d",
            version=SemVer(major=0, minor=1, patch=0),
            submodels=[s1, s2],
        )


# --- Invariant: bad parent reference --------------------------------------


def test_unknown_submodel_parent_rejected() -> None:
    s = SubmodelEntry(
        id="s", name="s", complexity_rank=0, ir_template_ref="t",
        parent_id="GHOST",
    )
    with pytest.raises(ValidationError, match="GHOST"):
        MetaModel(
            id="m", title="m", scientific_domain="d",
            version=SemVer(major=0, minor=1, patch=0),
            submodels=[s],
        )


# --- Invariant: bad operator endpoints ------------------------------------


def test_approximation_operator_unknown_submodel_rejected() -> None:
    s = SubmodelEntry(id="real", name="r", complexity_rank=0, ir_template_ref="t")
    op = ApproximationOperator(
        id="op", kind=ApproximationOperatorKind.LUMPING,
        from_submodel_id="real", to_submodel_id="GHOST",
        description="x",
    )
    with pytest.raises(ValidationError, match="GHOST"):
        MetaModel(
            id="m", title="m", scientific_domain="d",
            version=SemVer(major=0, minor=1, patch=0),
            submodels=[s],
            approximation_operators=[op],
        )


# --- Invariant: simplification must decrease complexity ---------------------


def test_approximation_operator_must_decrease_complexity() -> None:
    simple = SubmodelEntry(id="s0", name="s", complexity_rank=0, ir_template_ref="t")
    complex_m = SubmodelEntry(id="s2", name="s", complexity_rank=2, ir_template_ref="t")
    # op goes simple -> complex; should be rejected.
    op = ApproximationOperator(
        id="op", kind=ApproximationOperatorKind.AVERAGING,
        from_submodel_id="s0", to_submodel_id="s2",
        description="wrong direction",
    )
    with pytest.raises(ValidationError, match="higher to lower"):
        MetaModel(
            id="m", title="m", scientific_domain="d",
            version=SemVer(major=0, minor=1, patch=0),
            submodels=[simple, complex_m],
            approximation_operators=[op],
        )


# --- Invariant: unique reconciled-parameter keys ---------------------------


def test_duplicate_reconciled_parameters_rejected() -> None:
    b = ParameterBinding(parameter_id="mu_max", canonical_unit="1/day", point_estimate=1.0)
    r1 = ReconciledParameter(
        parameter_id="mu_max", context_keys={"species": "AOB"}, binding=b,
    )
    r2 = ReconciledParameter(
        parameter_id="mu_max", context_keys={"species": "AOB"},
        binding=ParameterBinding(
            parameter_id="mu_max", canonical_unit="1/day", point_estimate=2.0
        ),
    )
    with pytest.raises(ValidationError, match="duplicate reconciled_parameters"):
        MetaModel(
            id="m", title="m", scientific_domain="d",
            version=SemVer(major=0, minor=1, patch=0),
            reconciled_parameters=[r1, r2],
        )


# --- JSON round-trip --------------------------------------------------------


def test_metamodel_json_round_trip() -> None:
    m = make_nitrifying_biofilm_metamodel()
    restored = MetaModel.model_validate_json(m.model_dump_json())
    assert restored.id == m.id
    assert str(restored.version) == "1.2.0"
    assert len(restored.submodels) == len(m.submodels)
    assert len(restored.reconciled_parameters) == len(m.reconciled_parameters)
    aob_mu = restored.find_parameter("mu_max", {"species": "AOB"})
    assert aob_mu is not None


# --- Changelog -------------------------------------------------------------


def test_changelog_entry_crediting() -> None:
    entry = ChangelogEntry(
        version=SemVer.parse("1.2.0"),
        kind="update",
        summary="K_s(NO2) NOB updated from Spieck 2015",
        affected_parameter_ids=["K_s"],
        credited_to=["alice@example.edu"],
        suggestion_ids=["suggestion-abc"],
    )
    assert "alice@example.edu" in entry.credited_to


def test_quality_rating_enum_coverage() -> None:
    assert {q.value for q in QualityRating} == {"high", "moderate", "low", "very_low"}
