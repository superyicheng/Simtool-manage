"""IR construction tests.

These exercise the connector IR against its first target system — a
two-species nitrifying biofilm (AOB oxidizes NH4 to NO2; NOB oxidizes
NO2 to NO3; both respire O2). If the IR cannot express this cleanly,
it is the wrong shape.
"""

from __future__ import annotations

import pytest

from simtool.connector.ir import (
    IR_SCHEMA_VERSION,
    AgentPopulation,
    AssumptionHint,
    BoundaryCondition,
    ComputeBudget,
    DiffusionProcess,
    Distribution,
    FirstOrderDecayProcess,
    InitialCondition,
    MonodGrowthProcess,
    Observable,
    ParameterBinding,
    SamplingSpec,
    ScientificModel,
    Solute,
    SpatialDomain,
    Surface,
)


def _binding(parameter_id: str, unit: str, value: float, **ctx: str) -> ParameterBinding:
    return ParameterBinding(
        parameter_id=parameter_id,
        canonical_unit=unit,
        context_keys=dict(ctx),
        point_estimate=value,
    )


@pytest.fixture
def nitrifying_biofilm() -> ScientificModel:
    # --- Entities ----------------------------------------------------------
    aob = AgentPopulation(
        id="AOB",
        name="Ammonia-oxidizing bacteria",
        morphology="coccus",
        parameters={
            "cell_density": _binding("cell_density", "pg/fL", 0.15, species="AOB"),
            "cell_division_mass": _binding("cell_division_mass", "pg", 0.4, species="AOB"),
            "cell_initial_mass": _binding("cell_initial_mass", "pg", 0.2, species="AOB"),
        },
    )
    nob = AgentPopulation(
        id="NOB",
        name="Nitrite-oxidizing bacteria",
        morphology="coccus",
        parameters={
            "cell_density": _binding("cell_density", "pg/fL", 0.15, species="NOB"),
            "cell_division_mass": _binding("cell_division_mass", "pg", 0.4, species="NOB"),
            "cell_initial_mass": _binding("cell_initial_mass", "pg", 0.2, species="NOB"),
        },
    )
    nh4 = Solute(id="NH4", name="ammonium", chemical_formula="NH4+")
    no2 = Solute(id="NO2", name="nitrite", chemical_formula="NO2-")
    no3 = Solute(id="NO3", name="nitrate", chemical_formula="NO3-")
    o2 = Solute(id="O2", name="oxygen", chemical_formula="O2")

    domain = SpatialDomain(
        id="biofilm_domain",
        dimensionality=2,
        extent_um=(256.0, 256.0),
        resolution_hint_um=4.0,
        periodic_axes=[0],
    )
    substratum = Surface(id="substratum", name="substratum", axis=1, position="low")
    bulk_interface = Surface(id="bulk_top", name="bulk_liquid_interface", axis=1, position="high")

    # --- Processes ---------------------------------------------------------
    aob_growth = MonodGrowthProcess(
        id="AOB_growth",
        growing_entity="AOB",
        consumed_solutes=["NH4", "O2"],
        produced_solutes=["NO2"],
        parameters={
            "mu_max": _binding("mu_max", "1/day", 1.0, species="AOB"),
            "K_s_NH4": _binding("K_s", "mg/L", 1.0, species="AOB", substrate="ammonium"),
            "K_s_O2": _binding("K_s", "mg/L", 0.5, species="AOB", substrate="oxygen"),
            "Y_XS_NH4": _binding(
                "Y_XS", "g_biomass/g_substrate", 0.12,
                species="AOB", substrate="ammonium",
            ),
        },
    )
    nob_growth = MonodGrowthProcess(
        id="NOB_growth",
        growing_entity="NOB",
        consumed_solutes=["NO2", "O2"],
        produced_solutes=["NO3"],
        parameters={
            "mu_max": _binding("mu_max", "1/day", 0.8, species="NOB"),
            "K_s_NO2": _binding("K_s", "mg/L", 1.5, species="NOB", substrate="nitrite"),
            "K_s_O2": _binding("K_s", "mg/L", 0.7, species="NOB", substrate="oxygen"),
            "Y_XS_NO2": _binding(
                "Y_XS", "g_biomass/g_substrate", 0.05,
                species="NOB", substrate="nitrite",
            ),
        },
    )

    aob_decay = FirstOrderDecayProcess(
        id="AOB_decay",
        decaying_entity="AOB",
        parameters={"b": _binding("b", "1/day", 0.1, species="AOB")},
    )
    nob_decay = FirstOrderDecayProcess(
        id="NOB_decay",
        decaying_entity="NOB",
        parameters={"b": _binding("b", "1/day", 0.08, species="NOB")},
    )

    def _diffusion(solute: str, d_liquid: float, d_biofilm: float) -> DiffusionProcess:
        return DiffusionProcess(
            id=f"{solute}_diffusion",
            solute=solute,
            regions=["bulk_liquid", "biofilm"],
            parameters={
                "D_bulk_liquid": _binding(
                    "D_liquid", "um^2/s", d_liquid, substrate=solute.lower(),
                ),
                "D_biofilm": _binding(
                    "D_biofilm", "um^2/s", d_biofilm, substrate=solute.lower(),
                ),
            },
        )

    processes = [
        aob_growth,
        nob_growth,
        aob_decay,
        nob_decay,
        _diffusion("NH4", 1800.0, 1440.0),
        _diffusion("NO2", 1700.0, 1360.0),
        _diffusion("NO3", 1700.0, 1360.0),
        _diffusion("O2", 2000.0, 1600.0),
    ]

    # --- BCs: Dirichlet at bulk top, no-flux at substratum ----------------
    bcs = [
        BoundaryCondition(
            id=f"{s}_bulk",
            target_entity=s,
            surface="bulk_top",
            kind="dirichlet",
            value=_binding("S_bulk_initial", "mg/L", conc, substrate=s.lower()),
        )
        for s, conc in [("NH4", 30.0), ("NO2", 0.0), ("NO3", 0.0), ("O2", 8.0)]
    ] + [
        BoundaryCondition(
            id=f"{s}_substratum",
            target_entity=s,
            surface="substratum",
            kind="no_flux",
        )
        for s in ("NH4", "NO2", "NO3", "O2")
    ]

    # --- ICs: uniform solutes, random agent placement at substratum -------
    ics: list[InitialCondition] = []
    for s, conc in [("NH4", 30.0), ("NO2", 0.0), ("NO3", 0.0), ("O2", 8.0)]:
        ics.append(
            InitialCondition(
                id=f"{s}_IC",
                target_entity=s,
                kind="uniform",
                parameters={
                    "value": _binding(
                        "S_bulk_initial", "mg/L", conc, substrate=s.lower(),
                    ),
                },
            )
        )
    for species in ("AOB", "NOB"):
        ics.append(
            InitialCondition(
                id=f"{species}_IC",
                target_entity=species,
                kind="random_discrete_placement",
                parameters={
                    "n_agents": ParameterBinding(
                        parameter_id="n_agents_initial",
                        canonical_unit="dimensionless",
                        point_estimate=50.0,
                    ),
                    "layer_thickness_um": ParameterBinding(
                        parameter_id="initial_layer_thickness",
                        canonical_unit="um",
                        point_estimate=4.0,
                    ),
                },
            )
        )

    # --- Observables ------------------------------------------------------
    observables = [
        Observable(
            id="thickness",
            name="biofilm thickness",
            kind="scalar_time_series",
            target="biofilm_thickness",
            sampling=SamplingSpec(interval_s=3600.0),
        ),
        Observable(
            id="species_fractions",
            name="AOB/NOB biomass fractions",
            kind="scalar_time_series",
            target="biomass_fractions",
            sampling=SamplingSpec(interval_s=3600.0),
        ),
        Observable(
            id="NH4_flux_top",
            name="NH4 flux into biofilm",
            kind="flux_through_surface",
            target="NH4@bulk_top",
            sampling=SamplingSpec(interval_s=3600.0),
        ),
        Observable(
            id="O2_field_final",
            name="O2 concentration field, final",
            kind="spatial_field_snapshot",
            target="O2",
            sampling=SamplingSpec(final_only=True),
        ),
    ]

    compute = ComputeBudget(
        time_horizon_s=10 * 24 * 3600.0,  # 10 simulated days
        wall_time_budget_s=3600.0,
        timestep_hint_s=60.0,
    )

    assumptions = [
        AssumptionHint(
            id="temperature_constant",
            description="Temperature held at 20 C throughout the run.",
            justification="Literature kinetic parameters here are pooled from "
            "20-25 C studies; explicit temperature dependence deferred.",
            alternatives=["Arrhenius-corrected parameters", "diurnal forcing"],
        ),
        AssumptionHint(
            id="well_mixed_bulk",
            description="Bulk liquid above biofilm is well-mixed (Dirichlet BC).",
            justification="Boundary-layer mass transfer neglected at this "
            "stage; Reynolds number assumed high.",
            alternatives=["Robin BC with explicit mass-transfer coefficient"],
        ),
    ]

    return ScientificModel(
        id="nitrifying_biofilm_v0",
        title="Two-species nitrifying biofilm (AOB + NOB)",
        domain="microbial_biofilm",
        formalism="agent_based",
        entities=[aob, nob, nh4, no2, no3, o2, domain, substratum, bulk_interface],
        processes=processes,
        boundary_conditions=bcs,
        initial_conditions=ics,
        observables=observables,
        compute=compute,
        assumption_hints=assumptions,
        metadata={
            "ir_schema_version": IR_SCHEMA_VERSION,
            "simtool_version": "0.1.0",
        },
    )


def test_construction(nitrifying_biofilm: ScientificModel) -> None:
    m = nitrifying_biofilm
    assert m.formalism == "agent_based"
    assert m.entity_ids() == {
        "AOB", "NOB", "NH4", "NO2", "NO3", "O2",
        "biofilm_domain", "substratum", "bulk_top",
    }
    assert m.surface_ids() == {"substratum", "bulk_top"}
    assert len(m.processes) == 8
    assert m.ir_schema_version == IR_SCHEMA_VERSION


def test_process_discriminator(nitrifying_biofilm: ScientificModel) -> None:
    growths = [p for p in nitrifying_biofilm.processes if p.kind == "monod_growth"]
    assert len(growths) == 2
    aob = next(p for p in growths if p.growing_entity == "AOB")
    assert "NH4" in aob.consumed_solutes
    assert "NO2" in aob.produced_solutes


def test_bc_references_valid_entities_and_surfaces(
    nitrifying_biofilm: ScientificModel,
) -> None:
    ids = nitrifying_biofilm.entity_ids()
    surfaces = nitrifying_biofilm.surface_ids()
    for bc in nitrifying_biofilm.boundary_conditions:
        assert bc.target_entity in ids, f"BC {bc.id}: unknown entity {bc.target_entity}"
        assert bc.surface in surfaces, f"BC {bc.id}: unknown surface {bc.surface}"


def test_parameter_bindings_declare_canonical_units(
    nitrifying_biofilm: ScientificModel,
) -> None:
    seen: list[tuple[str, str]] = []
    for p in nitrifying_biofilm.processes:
        params = getattr(p, "parameters", {}) or {}
        for binding in params.values():
            seen.append((binding.parameter_id, binding.canonical_unit))
            assert binding.canonical_unit  # non-empty
            assert binding.point_estimate is not None or binding.distribution is not None
    assert ("mu_max", "1/day") in seen
    assert ("K_s", "mg/L") in seen
    assert ("D_liquid", "um^2/s") in seen


def test_round_trip_json(nitrifying_biofilm: ScientificModel) -> None:
    payload = nitrifying_biofilm.model_dump_json()
    restored = ScientificModel.model_validate_json(payload)
    assert restored == nitrifying_biofilm


def test_distribution_binding() -> None:
    """A binding with a distribution rather than a point estimate still
    validates. This shape is what the meta-model emits after reconciliation
    over multiple literature values."""
    b = ParameterBinding(
        parameter_id="mu_max",
        canonical_unit="1/day",
        context_keys={"species": "AOB"},
        distribution=Distribution(
            shape="lognormal",
            params={"mu": 0.0, "sigma": 0.3},
        ),
        provenance_dois=["10.1128/AEM.70.5.3024-3040.2004", "10.1002/bit.28045"],
        source_note="reconciled over 12 chemostat AOB studies, 20-25 C",
    )
    assert b.distribution is not None
    assert b.point_estimate is None
    assert len(b.provenance_dois) == 2
