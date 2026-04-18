"""Canonical REALISTIC example — multiple papers, some disagreement.

User story:

    "I want to simulate an AOB+NOB nitrifying biofilm. I've given the
    system 5 papers on AOB kinetics and 3 on NOB kinetics. The papers
    don't perfectly agree — AOB mu_max from chemostat studies runs
    lower than from initial-rate studies. I want an honest simulation
    that reflects what the literature actually says."

What the pipeline must deliver:
  1. Reconcile overlapping records per parameter. Agreement within
     tolerance collapses to a point estimate; beyond it becomes an
     explicit distribution.
  2. The IR reflects the distribution shape: a user can see that AOB
     mu_max is not known to three significant figures.
  3. Assumption ledger surfaces multiple choices (temperature
     normalization, method-bias handling, boundary treatment).
  4. Simulation runs end-to-end.
  5. Output bundle carries the requested observables.

This test exercises the reconciliation logic meaningfully. When this
test breaks, either the reconciler silently collapsed disagreement or
the connector silently dropped a distribution-shaped binding.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from simtool.connector.ir import (
    AgentPopulation,
    BoundaryCondition,
    ComputeBudget,
    DiffusionProcess,
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
from simtool.connector.plugin import DocSources
from simtool.connector.runs import RunLayout, RunRecord, RunStatus
from simtool.schema.parameter_record import GradeRating, MeasurementMethod

from tests._meta_model_fixtures import make_record, reconcile_records
from tests.test_connector_plugin import MockPlugin


# ---------------------------------------------------------------------------
# Corpus — 5 AOB papers (mu_max, K_s), 3 NOB papers. Values illustrative.
# ---------------------------------------------------------------------------


AOB_SPECIES = "Nitrosomonas europaea"
NOB_SPECIES = "Nitrobacter winogradskyi"


@pytest.fixture
def aob_mu_max_records() -> list:
    """5 AOB mu_max records spanning chemostat/initial-rate studies with
    real-world disagreement."""
    papers = [
        ("10.1038/nature23679",           1.03, MeasurementMethod.CHEMOSTAT, GradeRating.HIGH),
        ("10.1128/AEM.70.5.3024-3040.2004",  0.88, MeasurementMethod.CHEMOSTAT, GradeRating.HIGH),
        ("10.1002/bit.28045",              1.15, MeasurementMethod.BATCH_FIT,  GradeRating.MODERATE),
        ("10.1002/bit.260391202",          1.40, MeasurementMethod.INITIAL_RATE, GradeRating.MODERATE),
        ("10.2166/wst.2006.221",           0.95, MeasurementMethod.CHEMOSTAT, GradeRating.HIGH),
    ]
    return [
        make_record(
            parameter_id="mu_max", value=v, unit="1/day", canonical_value=v,
            doi=doi, species=AOB_SPECIES, temperature_c=25.0,
            method=method, grade=grade,
        )
        for doi, v, method, grade in papers
    ]


@pytest.fixture
def aob_ks_records() -> list:
    """AOB K_s(NH4) — tighter cluster. Values in mg N/L."""
    papers = [
        ("10.1038/nature23679",           0.68, GradeRating.HIGH),
        ("10.1128/AEM.70.5.3024-3040.2004",  0.75, GradeRating.HIGH),
        ("10.2166/wst.2006.221",           0.70, GradeRating.HIGH),
    ]
    return [
        make_record(
            parameter_id="K_s", value=v, unit="mg/L", canonical_value=v,
            doi=doi, species=AOB_SPECIES, substrate="ammonium", grade=grade,
        )
        for doi, v, grade in papers
    ]


@pytest.fixture
def nob_mu_max_records() -> list:
    """3 NOB mu_max records — mild disagreement."""
    papers = [
        ("10.1128/AEM.02734-14",  0.85, GradeRating.HIGH),
        ("10.1002/bit.28045",       0.78, GradeRating.MODERATE),
        ("10.2166/wst.2006.221",    0.82, GradeRating.HIGH),
    ]
    return [
        make_record(
            parameter_id="mu_max", value=v, unit="1/day", canonical_value=v,
            doi=doi, species=NOB_SPECIES, temperature_c=25.0, grade=grade,
        )
        for doi, v, grade in papers
    ]


# ---------------------------------------------------------------------------
# Reconciliation behavior — the user-visible contract
# ---------------------------------------------------------------------------


def test_aob_mu_max_disagreement_produces_distribution(
    aob_mu_max_records: list,
) -> None:
    """AOB mu_max across 5 papers spans ~0.88 to 1.40 — well above the 20%
    threshold. The user must see a distribution, NOT a spuriously-precise
    point estimate."""
    result = reconcile_records(aob_mu_max_records)
    assert result.binding is not None
    b = result.binding
    assert b.point_estimate is None, (
        "reconciliation silently collapsed conflicting records to a point "
        "estimate — the user would not see the uncertainty"
    )
    assert b.distribution is not None
    assert b.distribution.shape == "empirical"
    assert b.distribution.samples is not None
    assert len(b.distribution.samples) == 5
    assert result.conflict_flags, (
        "disagreement was not flagged to the user"
    )
    # All 5 DOIs must remain traceable.
    assert len(b.provenance_dois) == 5


def test_aob_ks_tight_cluster_collapses_to_point(aob_ks_records: list) -> None:
    """When records agree within threshold the reconciler returns a point
    estimate — adding noise isn't virtuous when the data genuinely agree."""
    result = reconcile_records(aob_ks_records)
    assert result.binding is not None
    assert result.binding.point_estimate is not None
    assert result.binding.distribution is None
    assert result.binding.point_estimate == pytest.approx((0.68 + 0.75 + 0.70) / 3)
    assert not result.conflict_flags


def test_nob_mu_max_reconciles_to_point(nob_mu_max_records: list) -> None:
    result = reconcile_records(nob_mu_max_records)
    assert result.binding is not None
    # The 3 NOB values span ~9% — under threshold, so point estimate.
    assert result.binding.point_estimate is not None


# ---------------------------------------------------------------------------
# User story: "give me a simulation that reflects the literature honestly"
# ---------------------------------------------------------------------------


def test_user_gets_multi_species_biofilm_from_multi_paper_corpus(
    aob_mu_max_records: list,
    aob_ks_records: list,
    nob_mu_max_records: list,
    tmp_path: Path,
) -> None:
    aob_mu = reconcile_records(aob_mu_max_records).binding
    aob_ks_nh4 = reconcile_records(aob_ks_records).binding
    nob_mu = reconcile_records(nob_mu_max_records).binding
    assert aob_mu and aob_ks_nh4 and nob_mu

    ir = _build_two_species_biofilm_ir(aob_mu, aob_ks_nh4, nob_mu)

    plugin = MockPlugin()
    skill = plugin.parse_docs(DocSources(sources=[]))
    report = plugin.validate_ir(ir, skill)
    assert report.ok, f"realistic IR did not validate: {report.errors()}"

    artifact = plugin.lower(ir, skill)
    # The reconciled AOB mu_max must travel through lowering as a distribution,
    # NOT be silently flattened.
    lowered_ir_json = next(
        content for name, content in artifact.files if name == "ir.json"
    )
    assert b'"distribution"' in lowered_ir_json, (
        "distribution-shaped AOB mu_max binding did not reach framework "
        "inputs — silent flattening would hide uncertainty from the user"
    )

    ledger = artifact.assumptions
    assert len(ledger.assumptions) >= 2

    # User reviews and approves.
    for a in ledger.assumptions:
        ledger.approve(a.id, user_note="reviewed for realistic case")

    layout = RunLayout.under(tmp_path / "nitrifying_biofilm_run")
    handle = plugin.execute(artifact, layout)
    reports = list(plugin.monitor(handle))
    assert reports

    bundle = plugin.parse_outputs(layout)
    assert "thickness" in bundle.scalar_time_series

    ledger_hash = (
        "sha256:"
        + hashlib.sha256(ledger.model_dump_json().encode()).hexdigest()
    )
    run = RunRecord(
        run_id=handle.run_id, ir_id=ir.id,
        framework=plugin.name, framework_version="0.1",
        skill_file_version=skill.skill_schema_version,
        assumption_ledger_hash=ledger_hash,
        layout_root=layout.root,
        status=RunStatus.SUCCEEDED,
        final_progress=reports[-1],
    )
    assert run.status == RunStatus.SUCCEEDED


def test_user_can_see_provenance_for_every_reconciled_parameter(
    aob_mu_max_records: list, aob_ks_records: list, nob_mu_max_records: list,
) -> None:
    """When asked 'which papers support this binding?' — the answer must
    be concrete."""
    for records, expected_n_dois in [
        (aob_mu_max_records, 5),
        (aob_ks_records, 3),
        (nob_mu_max_records, 3),
    ]:
        b = reconcile_records(records).binding
        assert b is not None
        assert len(b.provenance_dois) == expected_n_dois
        assert b.source_note


def test_context_keys_converge_on_shared_values(aob_mu_max_records: list) -> None:
    """All 5 AOB records name the same species; the reconciled binding
    must carry that context forward for downstream lookups."""
    b = reconcile_records(aob_mu_max_records).binding
    assert b is not None
    assert b.context_keys.get("species") == AOB_SPECIES


# ---------------------------------------------------------------------------
# IR builder (off the main narrative so the test stays readable)
# ---------------------------------------------------------------------------


def _fixed(pid: str, unit: str, v: float, **ctx: str) -> ParameterBinding:
    return ParameterBinding(
        parameter_id=pid, canonical_unit=unit, point_estimate=v, context_keys=dict(ctx)
    )


def _build_two_species_biofilm_ir(
    aob_mu: ParameterBinding,
    aob_ks_nh4: ParameterBinding,
    nob_mu: ParameterBinding,
) -> ScientificModel:
    aob = AgentPopulation(
        id="AOB", name=AOB_SPECIES,
        parameters={
            "cell_density": _fixed("cell_density", "pg/fL", 0.15),
            "cell_initial_mass": _fixed("cell_initial_mass", "pg", 0.2),
        },
    )
    nob = AgentPopulation(
        id="NOB", name=NOB_SPECIES,
        parameters={
            "cell_density": _fixed("cell_density", "pg/fL", 0.15),
            "cell_initial_mass": _fixed("cell_initial_mass", "pg", 0.2),
        },
    )
    nh4 = Solute(id="NH4", name="ammonium")
    no2 = Solute(id="NO2", name="nitrite")
    no3 = Solute(id="NO3", name="nitrate")
    o2 = Solute(id="O2", name="oxygen")
    dom = SpatialDomain(id="dom", dimensionality=2, extent_um=(128.0, 128.0))
    top = Surface(id="top", name="top", axis=1, position="high")
    bot = Surface(id="bottom", name="bottom", axis=1, position="low")

    aob_growth = MonodGrowthProcess(
        id="AOB_growth", growing_entity="AOB",
        consumed_solutes=["NH4", "O2"],
        produced_solutes=["NO2"],
        parameters={
            "mu_max": aob_mu,
            "K_s_NH4": aob_ks_nh4,
            "K_s_O2": _fixed("K_s", "mg/L", 0.5),
            "Y_XS_NH4": _fixed("Y_XS", "g_biomass/g_substrate", 0.12),
        },
    )
    nob_growth = MonodGrowthProcess(
        id="NOB_growth", growing_entity="NOB",
        consumed_solutes=["NO2", "O2"],
        produced_solutes=["NO3"],
        parameters={
            "mu_max": nob_mu,
            "K_s_NO2": _fixed("K_s", "mg/L", 1.5),
            "K_s_O2": _fixed("K_s", "mg/L", 0.7),
            "Y_XS_NO2": _fixed("Y_XS", "g_biomass/g_substrate", 0.05),
        },
    )
    decay_aob = FirstOrderDecayProcess(
        id="AOB_decay", decaying_entity="AOB",
        parameters={"b": _fixed("b", "1/day", 0.1)},
    )
    decay_nob = FirstOrderDecayProcess(
        id="NOB_decay", decaying_entity="NOB",
        parameters={"b": _fixed("b", "1/day", 0.08)},
    )

    def _diffusion(name: str) -> DiffusionProcess:
        return DiffusionProcess(
            id=f"{name}_diffusion",
            solute=name,
            regions=["bulk_liquid", "biofilm"],
            parameters={
                "D_bulk_liquid": _fixed("D_liquid", "um^2/s", 1800.0),
                "D_biofilm": _fixed("D_biofilm", "um^2/s", 1440.0),
            },
        )

    bcs = [
        BoundaryCondition(
            id=f"{s}_top", target_entity=s, surface="top", kind="dirichlet",
            value=_fixed("S_bulk_initial", "mg/L", conc, substrate=s.lower()),
        )
        for s, conc in [("NH4", 30.0), ("NO2", 0.0), ("NO3", 0.0), ("O2", 8.0)]
    ] + [
        BoundaryCondition(
            id=f"{s}_bot", target_entity=s, surface="bottom", kind="no_flux"
        )
        for s in ("NH4", "NO2", "NO3", "O2")
    ]

    ics = [
        InitialCondition(
            id=f"{sp}_IC", target_entity=sp,
            kind="random_discrete_placement",
            parameters={
                "n_agents": ParameterBinding(
                    parameter_id="n_agents_initial",
                    canonical_unit="dimensionless",
                    point_estimate=30.0,
                ),
                "layer_thickness_um": ParameterBinding(
                    parameter_id="initial_layer_thickness",
                    canonical_unit="um",
                    point_estimate=4.0,
                ),
            },
        )
        for sp in ("AOB", "NOB")
    ]

    return ScientificModel(
        id="realistic_nitrifying_biofilm",
        title="AOB+NOB biofilm reconciled from multi-paper corpus",
        domain="microbial_biofilm",
        formalism="agent_based",
        entities=[aob, nob, nh4, no2, no3, o2, dom, top, bot],
        processes=[
            aob_growth, nob_growth, decay_aob, decay_nob,
            _diffusion("NH4"), _diffusion("NO2"), _diffusion("NO3"), _diffusion("O2"),
        ],
        boundary_conditions=bcs,
        initial_conditions=ics,
        observables=[
            Observable(
                id="thickness", name="biofilm thickness",
                kind="scalar_time_series", target="biofilm_thickness",
                sampling=SamplingSpec(interval_s=3600.0),
            ),
        ],
        compute=ComputeBudget(time_horizon_s=86400.0 * 3, wall_time_budget_s=1800.0),
    )
