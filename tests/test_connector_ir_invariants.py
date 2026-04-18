"""IR invariant and malformed-input tests.

The IR is load-bearing — every downstream component (meta-model writer,
plugin lower(), comparator) trusts it. These tests pin the invariants so
a broken IR cannot silently flow past validation.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from simtool.connector.ir import (
    AgentPopulation,
    BoundaryCondition,
    ComputeBudget,
    CustomProcess,
    DiffusionProcess,
    Distribution,
    FirstOrderDecayProcess,
    InitialCondition,
    MaintenanceProcess,
    MonodGrowthProcess,
    Observable,
    ParameterBinding,
    ScientificModel,
    Solute,
    SpatialDomain,
    Surface,
)


# ---------------------------------------------------------------------------
# ParameterBinding — value-spec invariant
# ---------------------------------------------------------------------------


def test_parameter_binding_requires_point_or_distribution() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        ParameterBinding(parameter_id="mu_max", canonical_unit="1/day")


def test_parameter_binding_rejects_both_point_and_distribution() -> None:
    with pytest.raises(ValidationError, match="cannot have both"):
        ParameterBinding(
            parameter_id="mu_max",
            canonical_unit="1/day",
            point_estimate=1.0,
            distribution=Distribution(
                shape="normal", params={"mean": 1.0, "stddev": 0.1}
            ),
        )


def test_parameter_binding_requires_non_empty_canonical_unit() -> None:
    with pytest.raises(ValidationError, match="canonical_unit"):
        ParameterBinding(
            parameter_id="mu_max", canonical_unit="   ", point_estimate=1.0
        )


# ---------------------------------------------------------------------------
# Distribution — shape-specific required params
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape,params",
    [
        ("normal", {"mean": 1.0}),            # missing stddev
        ("normal", {"stddev": 0.1}),          # missing mean
        ("lognormal", {"mu": 0.0}),           # missing sigma
        ("uniform", {"low": 0.0}),            # missing high
        ("triangular", {"low": 0.0, "high": 1.0}),  # missing mode
    ],
)
def test_distribution_missing_params_rejected(
    shape: str, params: dict[str, float]
) -> None:
    with pytest.raises(ValidationError, match="missing"):
        Distribution(shape=shape, params=params)


def test_uniform_low_must_be_less_than_high() -> None:
    with pytest.raises(ValidationError, match="low < high"):
        Distribution(shape="uniform", params={"low": 1.0, "high": 0.5})


def test_triangular_mode_must_be_between_low_and_high() -> None:
    with pytest.raises(ValidationError, match="low <= mode <= high"):
        Distribution(
            shape="triangular", params={"low": 0.0, "mode": 2.0, "high": 1.0}
        )


def test_normal_stddev_must_be_positive() -> None:
    with pytest.raises(ValidationError, match="stddev > 0"):
        Distribution(shape="normal", params={"mean": 0.0, "stddev": 0.0})


def test_empirical_requires_samples() -> None:
    with pytest.raises(ValidationError, match="non-empty samples"):
        Distribution(shape="empirical")
    with pytest.raises(ValidationError, match="non-empty samples"):
        Distribution(shape="empirical", samples=[])


def test_empirical_with_samples_ok() -> None:
    d = Distribution(shape="empirical", samples=[0.5, 0.7, 0.9])
    assert d.shape == "empirical"


def test_unknown_distribution_shape_rejected() -> None:
    with pytest.raises(ValidationError):
        Distribution(shape="weibull", params={"k": 1.0})  # not in Literal


# ---------------------------------------------------------------------------
# SpatialDomain — geometry invariants
# ---------------------------------------------------------------------------


def test_extent_length_must_match_dimensionality() -> None:
    with pytest.raises(ValidationError, match="extent_um length"):
        SpatialDomain(id="d", dimensionality=3, extent_um=(10.0, 20.0))


def test_extent_must_be_positive() -> None:
    with pytest.raises(ValidationError, match="extent_um entries must be positive"):
        SpatialDomain(id="d", dimensionality=1, extent_um=(0.0,))


def test_periodic_axis_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError, match="periodic_axes"):
        SpatialDomain(
            id="d", dimensionality=2, extent_um=(10.0, 10.0), periodic_axes=[5]
        )


def test_periodic_axis_duplicates_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicates"):
        SpatialDomain(
            id="d", dimensionality=2, extent_um=(10.0, 10.0), periodic_axes=[0, 0]
        )


def test_resolution_must_be_positive() -> None:
    with pytest.raises(ValidationError, match="resolution_hint_um"):
        SpatialDomain(
            id="d",
            dimensionality=1,
            extent_um=(10.0,),
            resolution_hint_um=-1.0,
        )


def test_dimensionality_must_be_1_2_or_3() -> None:
    with pytest.raises(ValidationError):
        SpatialDomain(id="d", dimensionality=4, extent_um=(1.0, 1.0, 1.0, 1.0))


# ---------------------------------------------------------------------------
# ComputeBudget
# ---------------------------------------------------------------------------


def test_time_horizon_must_be_positive() -> None:
    with pytest.raises(ValidationError, match="time_horizon_s"):
        ComputeBudget(time_horizon_s=0.0)


def test_wall_time_budget_must_be_positive_if_set() -> None:
    with pytest.raises(ValidationError, match="wall_time_budget_s"):
        ComputeBudget(time_horizon_s=100.0, wall_time_budget_s=-1.0)


def test_parallel_hint_must_be_at_least_one() -> None:
    with pytest.raises(ValidationError, match="parallel_hint"):
        ComputeBudget(time_horizon_s=100.0, parallel_hint=0)


def test_compute_budget_defaults_ok() -> None:
    b = ComputeBudget(time_horizon_s=100.0)
    assert b.precision == "double"
    assert b.wall_time_budget_s is None


# ---------------------------------------------------------------------------
# ScientificModel — referential integrity
# ---------------------------------------------------------------------------


def _mk_point(pid: str, unit: str, v: float) -> ParameterBinding:
    return ParameterBinding(parameter_id=pid, canonical_unit=unit, point_estimate=v)


def _mk_valid_model(**overrides) -> ScientificModel:
    aob = AgentPopulation(id="AOB", name="AOB")
    nh4 = Solute(id="NH4", name="ammonium")
    dom = SpatialDomain(id="dom", dimensionality=2, extent_um=(100.0, 100.0))
    sub = Surface(id="bottom", name="bottom", axis=1, position="low")
    top = Surface(id="top", name="top", axis=1, position="high")
    bc = BoundaryCondition(
        id="nh4_top",
        target_entity="NH4",
        surface="top",
        kind="dirichlet",
        value=_mk_point("S_bulk_initial", "mg/L", 30.0),
    )
    base = dict(
        id="m",
        title="test",
        domain="biofilm",
        formalism="agent_based",
        entities=[aob, nh4, dom, sub, top],
        processes=[],
        boundary_conditions=[bc],
        initial_conditions=[],
        observables=[],
        compute=ComputeBudget(time_horizon_s=1000.0),
    )
    base.update(overrides)
    return ScientificModel(**base)


def test_valid_minimal_model_constructs() -> None:
    m = _mk_valid_model()
    assert "AOB" in m.entity_ids()


def test_duplicate_entity_ids_rejected() -> None:
    aob1 = AgentPopulation(id="AOB", name="AOB-1")
    aob2 = AgentPopulation(id="AOB", name="AOB-2")
    with pytest.raises(ValidationError, match="duplicate entity ids"):
        _mk_valid_model(entities=[aob1, aob2])


def test_duplicate_process_ids_rejected() -> None:
    decay_a = FirstOrderDecayProcess(
        id="decay", decaying_entity="AOB",
        parameters={"b": _mk_point("b", "1/day", 0.1)},
    )
    decay_b = FirstOrderDecayProcess(
        id="decay", decaying_entity="AOB",
        parameters={"b": _mk_point("b", "1/day", 0.2)},
    )
    with pytest.raises(ValidationError, match="duplicate process ids"):
        _mk_valid_model(processes=[decay_a, decay_b])


def test_duplicate_bc_ids_rejected() -> None:
    v = _mk_point("S_bulk_initial", "mg/L", 30.0)
    bc1 = BoundaryCondition(
        id="bc", target_entity="NH4", surface="top", kind="dirichlet", value=v
    )
    bc2 = BoundaryCondition(
        id="bc", target_entity="NH4", surface="top", kind="no_flux"
    )
    with pytest.raises(ValidationError, match="duplicate boundary condition ids"):
        _mk_valid_model(boundary_conditions=[bc1, bc2])


def test_bc_references_missing_entity_rejected() -> None:
    bc = BoundaryCondition(
        id="bad",
        target_entity="GHOST",
        surface="top",
        kind="dirichlet",
        value=_mk_point("S_bulk_initial", "mg/L", 30.0),
    )
    with pytest.raises(ValidationError, match="GHOST"):
        _mk_valid_model(boundary_conditions=[bc])


def test_bc_references_non_surface_as_surface_rejected() -> None:
    # NH4 is a Solute, not a Surface — using it as the surface slot should fail.
    bc = BoundaryCondition(
        id="bad",
        target_entity="NH4",
        surface="NH4",
        kind="no_flux",
    )
    with pytest.raises(ValidationError, match="not a Surface"):
        _mk_valid_model(boundary_conditions=[bc])


def test_periodic_bc_requires_paired_surface() -> None:
    bc = BoundaryCondition(
        id="per", target_entity="NH4", surface="top", kind="periodic"
    )
    with pytest.raises(ValidationError, match="periodic kind requires paired_surface"):
        _mk_valid_model(boundary_conditions=[bc])


def test_periodic_bc_with_valid_pair_ok() -> None:
    bc = BoundaryCondition(
        id="per",
        target_entity="NH4",
        surface="top",
        kind="periodic",
        paired_surface="bottom",
    )
    m = _mk_valid_model(boundary_conditions=[bc])
    assert m.boundary_conditions[0].kind == "periodic"


def test_robin_bc_requires_bulk_value() -> None:
    bc = BoundaryCondition(
        id="rob",
        target_entity="NH4",
        surface="top",
        kind="robin",
        value=_mk_point("mass_transfer", "m/s", 1e-4),
        # robin_bulk_value missing
    )
    with pytest.raises(ValidationError, match="robin kind requires robin_bulk_value"):
        _mk_valid_model(boundary_conditions=[bc])


def test_ic_references_missing_entity_rejected() -> None:
    ic = InitialCondition(id="ic_ghost", target_entity="NOT_THERE", kind="zero")
    with pytest.raises(ValidationError, match="NOT_THERE"):
        _mk_valid_model(initial_conditions=[ic])


def test_monod_growth_references_unknown_solute_rejected() -> None:
    p = MonodGrowthProcess(
        id="aob_growth",
        growing_entity="AOB",
        consumed_solutes=["PHANTOM"],
        parameters={
            "mu_max": _mk_point("mu_max", "1/day", 1.0),
        },
    )
    with pytest.raises(ValidationError, match="PHANTOM"):
        _mk_valid_model(processes=[p])


def test_monod_growth_unknown_growing_entity_rejected() -> None:
    p = MonodGrowthProcess(
        id="ghost_growth",
        growing_entity="GHOST",
        consumed_solutes=["NH4"],
        parameters={"mu_max": _mk_point("mu_max", "1/day", 1.0)},
    )
    with pytest.raises(ValidationError, match="GHOST"):
        _mk_valid_model(processes=[p])


def test_diffusion_unknown_solute_rejected() -> None:
    p = DiffusionProcess(
        id="ghost_diff",
        solute="GHOST",
        parameters={"D_bulk_liquid": _mk_point("D_liquid", "um^2/s", 1000.0)},
    )
    with pytest.raises(ValidationError, match="GHOST"):
        _mk_valid_model(processes=[p])


def test_maintenance_unknown_solute_rejected() -> None:
    p = MaintenanceProcess(
        id="maint",
        entity="AOB",
        consumed_solutes=["PHANTOM"],
        parameters={"m_s_NH4": _mk_point("m_s", "g/(g*day)", 0.1)},
    )
    with pytest.raises(ValidationError, match="PHANTOM"):
        _mk_valid_model(processes=[p])


def test_custom_process_unknown_actor_rejected() -> None:
    p = CustomProcess(
        id="c",
        description="mystery",
        actors=["GHOST"],
    )
    with pytest.raises(ValidationError, match="GHOST"):
        _mk_valid_model(processes=[p])


def test_surface_axis_out_of_range_for_domain_rejected() -> None:
    # 2D domain → axis 2 is invalid for a surface
    bad = Surface(id="bad", name="bad", axis=2, position="low")
    with pytest.raises(ValidationError, match="axis 2 out of range"):
        _mk_valid_model(
            entities=[
                AgentPopulation(id="AOB", name="AOB"),
                Solute(id="NH4", name="NH4"),
                SpatialDomain(id="dom", dimensionality=2, extent_um=(10.0, 10.0)),
                Surface(id="top", name="top", axis=1, position="high"),
                bad,
            ],
            boundary_conditions=[],
        )


# ---------------------------------------------------------------------------
# Enum / Literal coverage
# ---------------------------------------------------------------------------


def test_unknown_formalism_rejected() -> None:
    with pytest.raises(ValidationError):
        _mk_valid_model(formalism="telepathy")


def test_unknown_bc_kind_rejected() -> None:
    with pytest.raises(ValidationError):
        BoundaryCondition(
            id="x", target_entity="NH4", surface="top", kind="mystical"
        )


def test_unknown_observable_kind_rejected() -> None:
    with pytest.raises(ValidationError):
        Observable(id="o", name="o", kind="crystal_ball", target="NH4")


# ---------------------------------------------------------------------------
# Empty model edge cases — allowed but semantically thin
# ---------------------------------------------------------------------------


def test_empty_entities_and_processes_allowed() -> None:
    """An IR with no entities or processes is syntactically valid but
    semantically useless. Plugins should reject it in validate_ir. We do not
    enforce non-emptiness at schema level — that's a plugin-level concern."""
    m = ScientificModel(
        id="empty",
        title="empty",
        domain="nothing",
        formalism="ode",
        entities=[],
        processes=[],
        compute=ComputeBudget(time_horizon_s=1.0),
    )
    assert m.entity_ids() == set()
