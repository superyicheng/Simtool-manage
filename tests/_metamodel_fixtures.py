"""Shared fixtures for meta-model + panel tests.

Builds a nitrifying-biofilm meta-model small enough to reason about end
to end, with three submodels at different complexity ranks and one
approximation operator between the two richer ones.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from simtool.connector.ir import (
    AgentPopulation,
    BoundaryCondition,
    ComputeBudget,
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
from simtool.metamodel import (
    ApproximationOperator,
    ApproximationOperatorKind,
    IngestionCadence,
    IngestionStatus,
    MetaModel,
    QualityRating,
    ReconciledParameter,
    ScopeContract,
    SemVer,
    SubmodelEntry,
)


def _binding(
    pid: str,
    unit: str,
    v: float,
    **ctx: str,
) -> ParameterBinding:
    return ParameterBinding(
        parameter_id=pid,
        canonical_unit=unit,
        context_keys=dict(ctx),
        point_estimate=v,
    )


def _dist_binding(
    pid: str,
    unit: str,
    samples: list[float],
    **ctx: str,
) -> ParameterBinding:
    return ParameterBinding(
        parameter_id=pid,
        canonical_unit=unit,
        context_keys=dict(ctx),
        distribution=Distribution(shape="empirical", samples=samples),
    )


def make_nitrifying_biofilm_metamodel(
    *,
    last_ingested_days_ago: int = 2,
) -> MetaModel:
    """Return a small but realistic meta-model for nitrifying biofilm."""
    now = datetime.now(timezone.utc)

    submodels = [
        SubmodelEntry(
            id="monod_chemostat_ode",
            name="Monod chemostat ODE",
            complexity_rank=0,
            ir_template_ref="templates/chemostat_ode",
            required_parameter_ids=["mu_max", "K_s", "Y_XS"],
            covered_phenomena=["growth", "substrate_limitation"],
            excluded_phenomena=["spatial_gradients", "biofilm_thickness"],
        ),
        SubmodelEntry(
            id="continuum_pde_2d",
            name="2D continuum PDE biofilm",
            complexity_rank=2,
            ir_template_ref="templates/continuum_pde_2d",
            required_parameter_ids=["mu_max", "K_s", "Y_XS", "b", "D_liquid", "D_biofilm"],
            covered_phenomena=[
                "growth", "substrate_limitation", "decay", "diffusion",
                "spatial_gradients", "biofilm_thickness",
            ],
            excluded_phenomena=["individual_cells", "detachment"],
        ),
        SubmodelEntry(
            id="agent_based_3d",
            name="3D agent-based biofilm",
            complexity_rank=4,
            ir_template_ref="templates/agent_based_3d",
            required_parameter_ids=[
                "mu_max", "K_s", "Y_XS", "b", "D_liquid", "D_biofilm",
                "cell_density", "cell_division_mass",
            ],
            covered_phenomena=[
                "growth", "substrate_limitation", "decay", "diffusion",
                "spatial_gradients", "biofilm_thickness",
                "individual_cells", "detachment", "eps_production",
            ],
        ),
    ]

    operators = [
        ApproximationOperator(
            id="abm3d_to_pde2d_mean_field",
            kind=ApproximationOperatorKind.MEAN_FIELD_CLOSURE,
            from_submodel_id="agent_based_3d",
            to_submodel_id="continuum_pde_2d",
            description=(
                "Mean-field closure collapses individual agents into a "
                "continuum biomass field; 3D structure averaged to 2D."
            ),
            assumptions_introduced=[
                "biomass can be treated as a continuum (scale-separation ok)",
                "vertical averaging is defensible",
            ],
            validity_conditions=[
                "biofilm thickness << horizontal extent",
                "number of agents per voxel >> 1",
            ],
        ),
    ]

    reconciled = [
        ReconciledParameter(
            parameter_id="mu_max",
            context_keys={"species": "AOB"},
            binding=_binding("mu_max", "1/day", 1.0, species="AOB"),
            supporting_record_dois=[
                "10.1038/nature23679",
                "10.1128/AEM.70.5.3024-3040.2004",
                "10.2166/wst.2006.221",
            ],
            quality_rating=QualityRating.HIGH,
        ),
        ReconciledParameter(
            parameter_id="mu_max",
            context_keys={"species": "NOB"},
            binding=_binding("mu_max", "1/day", 0.8, species="NOB"),
            supporting_record_dois=["10.1128/AEM.02734-14", "10.2166/wst.2006.221"],
            quality_rating=QualityRating.HIGH,
        ),
        ReconciledParameter(
            parameter_id="K_s",
            context_keys={"species": "AOB", "substrate": "ammonium"},
            binding=_binding("K_s", "mg/L", 0.7, species="AOB", substrate="ammonium"),
            supporting_record_dois=["10.1038/nature23679", "10.2166/wst.2006.221"],
            quality_rating=QualityRating.HIGH,
        ),
        ReconciledParameter(
            parameter_id="K_s",
            context_keys={"species": "AOB", "substrate": "oxygen"},
            binding=_dist_binding(
                "K_s", "mg/L", [0.3, 0.5, 0.8],
                species="AOB", substrate="oxygen",
            ),
            supporting_record_dois=[
                "10.1128/AEM.03163-15", "10.1002/bit.28045", "10.1128/AEM.00330-11",
            ],
            quality_rating=QualityRating.MODERATE,
            conflict_flags=["K_s(O2) AOB spans 0.3-0.8 mg/L across studies"],
        ),
        ReconciledParameter(
            parameter_id="K_s",
            context_keys={"species": "NOB", "substrate": "nitrite"},
            binding=_binding("K_s", "mg/L", 1.5, species="NOB", substrate="nitrite"),
            supporting_record_dois=["10.1128/AEM.02734-14", "10.1002/bit.28045"],
        ),
        ReconciledParameter(
            parameter_id="Y_XS",
            context_keys={"species": "AOB", "substrate": "ammonium"},
            binding=_binding(
                "Y_XS", "g_biomass/g_substrate", 0.12,
                species="AOB", substrate="ammonium",
            ),
            supporting_record_dois=["10.1128/AEM.70.5.3024-3040.2004"],
            quality_rating=QualityRating.MODERATE,
        ),
        ReconciledParameter(
            parameter_id="Y_XS",
            context_keys={"species": "NOB", "substrate": "nitrite"},
            binding=_binding(
                "Y_XS", "g_biomass/g_substrate", 0.05,
                species="NOB", substrate="nitrite",
            ),
            supporting_record_dois=["10.2166/wst.2006.221"],
        ),
        ReconciledParameter(
            parameter_id="b",
            context_keys={"species": "AOB"},
            binding=_binding("b", "1/day", 0.1, species="AOB"),
            supporting_record_dois=["10.2166/wst.2006.221", "10.1002/bit.28045"],
        ),
        ReconciledParameter(
            parameter_id="b",
            context_keys={"species": "NOB"},
            binding=_binding("b", "1/day", 0.08, species="NOB"),
            supporting_record_dois=["10.2166/wst.2006.221"],
        ),
    ]

    ingestion = IngestionStatus(
        cadence=IngestionCadence.WEEKLY,
        last_ingestion_at=now - timedelta(days=last_ingested_days_ago),
        next_scheduled_at=now + timedelta(days=7 - last_ingested_days_ago),
        papers_detected=42, papers_processed=18, papers_integrated=4,
        papers_flagged_for_review=2,
    )

    return MetaModel(
        id="nitrifying_biofilm",
        title="Nitrifying biofilm (AOB/NOB) meta-model",
        scientific_domain="microbial_biofilm",
        version=SemVer(major=1, minor=2, patch=0),
        zenodo_doi="10.5281/zenodo.example",
        maintainers=["maintainer@example.edu"],
        reconciled_parameters=reconciled,
        submodels=submodels,
        approximation_operators=operators,
        ingestion=ingestion,
    )


def make_nitrifying_biofilm_scope() -> ScopeContract:
    return ScopeContract(
        in_scope_parameter_ids=[
            "mu_max", "K_s", "Y_XS", "b", "D_liquid", "D_biofilm",
            "cell_density", "cell_division_mass",
        ],
        in_scope_phenomena=[
            "growth", "substrate_limitation", "decay", "diffusion",
            "spatial_gradients", "biofilm_thickness", "individual_cells",
            "detachment", "eps_production",
        ],
        out_of_scope_notes=["pH effects below 5.5 not modeled"],
        min_records_for_reconciliation=2,
    )


def make_small_panel_ir(
    *,
    aob_mu_max: ParameterBinding | None = None,
    aob_ks_nh4: ParameterBinding | None = None,
) -> ScientificModel:
    """Minimal two-species IR the panel workflows can exercise."""
    mu = aob_mu_max or _binding("mu_max", "1/day", 1.0, species="AOB")
    ks = aob_ks_nh4 or _binding(
        "K_s", "mg/L", 0.7, species="AOB", substrate="ammonium"
    )
    aob = AgentPopulation(id="AOB", name="AOB")
    nh4 = Solute(id="NH4", name="ammonium")
    o2 = Solute(id="O2", name="oxygen")
    dom = SpatialDomain(id="dom", dimensionality=1, extent_um=(64.0,))
    top = Surface(id="top", name="top", axis=0, position="high")
    bot = Surface(id="bot", name="bot", axis=0, position="low")
    growth = MonodGrowthProcess(
        id="AOB_growth",
        growing_entity="AOB",
        consumed_solutes=["NH4", "O2"],
        parameters={
            "mu_max": mu,
            "K_s_NH4": ks,
            "K_s_O2": _binding(
                "K_s", "mg/L", 0.5, species="AOB", substrate="oxygen"
            ),
            "Y_XS_NH4": _binding(
                "Y_XS", "g_biomass/g_substrate", 0.12,
                species="AOB", substrate="ammonium",
            ),
        },
    )
    decay = FirstOrderDecayProcess(
        id="AOB_decay", decaying_entity="AOB",
        parameters={"b": _binding("b", "1/day", 0.1, species="AOB")},
    )
    bcs = [
        BoundaryCondition(
            id="NH4_top", target_entity="NH4", surface="top", kind="dirichlet",
            value=_binding("S_bulk_initial", "mg/L", 30.0, substrate="ammonium"),
        ),
    ]
    ics: list[InitialCondition] = []
    return ScientificModel(
        id="small_panel_ir",
        title="small panel IR",
        domain="microbial_biofilm",
        formalism="agent_based",
        entities=[aob, nh4, o2, dom, top, bot],
        processes=[growth, decay],
        boundary_conditions=bcs,
        initial_conditions=ics,
        observables=[
            Observable(
                id="thickness", name="thickness",
                kind="scalar_time_series", target="biofilm_thickness",
                sampling=SamplingSpec(interval_s=3600.0),
            ),
        ],
        compute=ComputeBudget(time_horizon_s=7200.0),
    )
