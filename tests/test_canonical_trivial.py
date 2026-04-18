"""Canonical TRIVIAL example — a single well-characterized paper.

User story (from the user's perspective, not the code's):

    "I have one paper on a pure AOB culture — Kits 2017 on
    Nitrosomonas europaea in a chemostat at 25 C. I want a simulation
    of that culture at the paper's reported conditions. Show me the
    trajectory."

What the pipeline must deliver to the user end-to-end:
  1. Ingest the paper's parameters (here: one ParameterRecord per
     canonical key).
  2. Reconcile (trivially — one record in, one binding out).
  3. Build a valid IR.
  4. Plugin validates + lowers the IR, producing framework inputs and
     an assumption ledger the user can approve.
  5. Run completes; output bundle has the declared observables.
  6. Reproducibility protocol is written.

This test should pass from day one and never regress. When it breaks,
the build is broken regardless of what the unit tests say.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from simtool.connector.ir import (
    AgentPopulation,
    BoundaryCondition,
    ComputeBudget,
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
# The "paper" — one ParameterRecord per parameter, all from Kits 2017.
# Values are illustrative of N. europaea oligotrophic AOB kinetics from the
# literature; exact numbers are not the point of this test.
# ---------------------------------------------------------------------------


KITS_2017_DOI = "10.1038/nature23679"


@pytest.fixture
def kits_2017_records() -> dict[str, list]:
    """The extractor's output if we fed it exactly Kits 2017.

    One record per canonical parameter. The real extractor would produce
    these with full provenance; we build them directly for this test.
    """
    species = "Nitrosomonas europaea"
    return {
        "mu_max": [
            make_record(
                parameter_id="mu_max",
                value=0.043, unit="1/h", canonical_value=1.03,
                doi=KITS_2017_DOI, species=species,
                temperature_c=25.0,
                method=MeasurementMethod.CHEMOSTAT,
                grade=GradeRating.HIGH,
            ),
        ],
        "K_s": [
            make_record(
                parameter_id="K_s",
                value=0.68, unit="mg/L", canonical_value=0.68,
                doi=KITS_2017_DOI, species=species, substrate="ammonium",
                temperature_c=25.0, grade=GradeRating.HIGH,
            ),
        ],
        "Y_XS": [
            make_record(
                parameter_id="Y_XS",
                value=0.12, unit="g_biomass/g_substrate", canonical_value=0.12,
                doi=KITS_2017_DOI, species=species, substrate="ammonium",
                grade=GradeRating.HIGH,
            ),
        ],
        "b": [
            make_record(
                parameter_id="b",
                value=0.1, unit="1/day", canonical_value=0.1,
                doi=KITS_2017_DOI, species=species, grade=GradeRating.MODERATE,
            ),
        ],
    }


# ---------------------------------------------------------------------------
# User story: "I want a simulation of this paper's pure culture."
# ---------------------------------------------------------------------------


def test_user_gets_working_simulation_from_single_paper(
    kits_2017_records: dict[str, list], tmp_path: Path
) -> None:
    # --- Step 1: reconcile the paper's records (trivial: 1 in -> 1 out) ---
    bindings = {}
    for pid, records in kits_2017_records.items():
        result = reconcile_records(records)
        assert result.binding is not None, (
            f"reconciliation failed for {pid} — this is the TRIVIAL case, "
            f"it must succeed"
        )
        assert result.binding.point_estimate is not None, (
            "single-record reconciliation should yield a point estimate, "
            "not a distribution"
        )
        assert result.binding.provenance_dois == [KITS_2017_DOI]
        bindings[pid] = result.binding

    # Sanity on the reconciled values the user will see.
    assert bindings["mu_max"].point_estimate == pytest.approx(1.03)
    assert bindings["K_s"].point_estimate == pytest.approx(0.68)

    # --- Step 2: user specifies what they want to simulate ---
    # "pure AOB culture, chemostat-like boundary with fresh feed"
    ir = _build_aob_pure_culture_ir(bindings)

    # --- Step 3: plugin validates; user's request must be expressible ---
    plugin = MockPlugin()
    skill = plugin.parse_docs(DocSources(sources=[]))
    report = plugin.validate_ir(ir, skill)
    assert report.ok, (
        f"plugin rejected a single-paper pure-culture IR — this is the "
        f"trivial case, it must validate. Errors: {report.errors()}"
    )

    # --- Step 4: lower; user gets artifact + assumption ledger to review --
    artifact = plugin.lower(ir, skill)
    assert artifact.entrypoint
    assert len(artifact.files) >= 1
    ledger = artifact.assumptions
    assert len(ledger.assumptions) >= 1, (
        "even the trivial case should surface at least one implicit "
        "assumption — silent lowering is a bug"
    )
    assert not ledger.is_ready_to_run(), (
        "fresh ledger must be pending; run must not be launchable yet"
    )

    # --- Step 5: user reviews and approves the ledger ---
    for a in ledger.assumptions:
        ledger.approve(a.id, user_note="reviewed for trivial case")
    assert ledger.is_ready_to_run()

    # --- Step 6: execute + monitor -----------------------------------------
    layout = RunLayout.under(tmp_path / "kits_2017_run")
    handle = plugin.execute(artifact, layout)
    reports = list(plugin.monitor(handle))
    assert len(reports) > 0, "monitor yielded no progress reports"
    final = reports[-1]
    assert final.sim_time_s is not None
    assert final.sim_time_s > 0

    # --- Step 7: user gets outputs + reproducibility protocol -------------
    bundle = plugin.parse_outputs(layout)
    assert "thickness" in bundle.scalar_time_series, (
        "requested observable must appear in the output bundle"
    )
    assert len(bundle.scalar_time_series["thickness"]) >= 2

    protocol_path = plugin.generate_protocol(ir, layout)
    assert protocol_path.exists()
    protocol_text = protocol_path.read_text()
    assert ir.id in protocol_text
    assert ir.formalism in protocol_text

    # --- Step 8: a citable RunRecord is produced --------------------------
    ledger_hash = (
        "sha256:"
        + hashlib.sha256(ledger.model_dump_json().encode()).hexdigest()
    )
    run = RunRecord(
        run_id=handle.run_id,
        ir_id=ir.id,
        framework=plugin.name,
        framework_version="0.1",
        skill_file_version=skill.skill_schema_version,
        assumption_ledger_hash=ledger_hash,
        layout_root=layout.root,
        status=RunStatus.SUCCEEDED,
        final_progress=final,
    )
    assert run.status == RunStatus.SUCCEEDED
    assert run.assumption_ledger_hash == ledger_hash


# ---------------------------------------------------------------------------
# Ancillary user-visible checks — the story pieces that aren't the main flow
# ---------------------------------------------------------------------------


def test_user_sees_provenance_on_every_binding(
    kits_2017_records: dict[str, list],
) -> None:
    """When the user asks 'where did this parameter come from?' — the
    answer must be reachable."""
    for pid, records in kits_2017_records.items():
        b = reconcile_records(records).binding
        assert b is not None
        assert b.provenance_dois, f"{pid} binding has no provenance DOIs"
        assert b.source_note, f"{pid} binding has no source_note"


def test_user_sees_single_doi_for_single_paper(
    kits_2017_records: dict[str, list],
) -> None:
    for pid, records in kits_2017_records.items():
        b = reconcile_records(records).binding
        assert b is not None
        assert len(b.provenance_dois) == 1


# ---------------------------------------------------------------------------
# Helpers — IR builder kept out of the main test body so the user story
# reads as a linear narrative.
# ---------------------------------------------------------------------------


def _fixed(pid: str, unit: str, v: float, **ctx: str) -> ParameterBinding:
    return ParameterBinding(
        parameter_id=pid, canonical_unit=unit, point_estimate=v, context_keys=dict(ctx)
    )


def _build_aob_pure_culture_ir(
    bindings: dict[str, ParameterBinding],
) -> ScientificModel:
    aob = AgentPopulation(
        id="AOB",
        name="N. europaea",
        parameters={
            "cell_density": _fixed("cell_density", "pg/fL", 0.15),
            "cell_initial_mass": _fixed("cell_initial_mass", "pg", 0.2),
        },
    )
    nh4 = Solute(id="NH4", name="ammonium")
    o2 = Solute(id="O2", name="oxygen")
    dom = SpatialDomain(id="dom", dimensionality=1, extent_um=(128.0,))
    top = Surface(id="top", name="top", axis=0, position="high")
    bot = Surface(id="bottom", name="bottom", axis=0, position="low")

    growth = MonodGrowthProcess(
        id="AOB_growth",
        growing_entity="AOB",
        consumed_solutes=["NH4", "O2"],
        parameters={
            "mu_max": bindings["mu_max"],
            "K_s_NH4": bindings["K_s"],
            "K_s_O2": _fixed("K_s", "mg/L", 0.5, species="AOB", substrate="oxygen"),
            "Y_XS_NH4": bindings["Y_XS"],
        },
    )
    decay = FirstOrderDecayProcess(
        id="AOB_decay",
        decaying_entity="AOB",
        parameters={"b": bindings["b"]},
    )

    bcs = [
        BoundaryCondition(
            id="NH4_feed", target_entity="NH4", surface="top",
            kind="dirichlet",
            value=_fixed("S_bulk_initial", "mg/L", 30.0, substrate="ammonium"),
        ),
        BoundaryCondition(
            id="O2_feed", target_entity="O2", surface="top",
            kind="dirichlet",
            value=_fixed("S_bulk_initial", "mg/L", 8.0, substrate="oxygen"),
        ),
        BoundaryCondition(id="NH4_bot", target_entity="NH4", surface="bottom", kind="no_flux"),
        BoundaryCondition(id="O2_bot", target_entity="O2", surface="bottom", kind="no_flux"),
    ]
    ics = [
        InitialCondition(
            id="AOB_IC",
            target_entity="AOB",
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
    ]
    observables = [
        Observable(
            id="thickness", name="biofilm thickness",
            kind="scalar_time_series", target="biofilm_thickness",
            sampling=SamplingSpec(interval_s=3600.0),
        ),
    ]
    return ScientificModel(
        id="kits_2017_pure_culture",
        title="N. europaea pure culture from Kits 2017",
        domain="microbial_biofilm",
        formalism="agent_based",
        entities=[aob, nh4, o2, dom, top, bot],
        processes=[growth, decay],
        boundary_conditions=bcs,
        initial_conditions=ics,
        observables=observables,
        compute=ComputeBudget(time_horizon_s=7200.0, wall_time_budget_s=600.0),
    )
