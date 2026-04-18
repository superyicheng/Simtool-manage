"""Canonical end-to-end test: drive a real iDynoMiCS 2 run.

Opt-in. Requires ``IDYNOMICS_2_JAR`` to point at a working jar. Skipped
otherwise. This is the test that catches 'the folder dropped into Ara
doesn't actually run' regressions.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from simtool.connector.ir import (
    AgentPopulation,
    BoundaryCondition,
    ComputeBudget,
    DiffusionProcess,
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
from simtool.connector.runs import RunStatus
from simtool.frameworks.idynomics_2 import IDynoMiCS2Plugin, resolve_jar_path
from simtool.panels import Panel, UserConstraints
from simtool.runner import run_panel


def _has_jar() -> bool:
    return resolve_jar_path() is not None


pytestmark = pytest.mark.skipif(
    not _has_jar(),
    reason="IDYNOMICS_2_JAR not set (or default paths don't resolve)",
)


def _b(pid, unit, v):
    return ParameterBinding(parameter_id=pid, canonical_unit=unit, point_estimate=v)


def _smoke_biofilm_ir() -> ScientificModel:
    aob = AgentPopulation(id="AOB", name="AOB", parameters={
        "cell_density": _b("cell_density", "pg/fL", 0.15),
        "cell_initial_mass": _b("cell_initial_mass", "pg", 0.2),
        "cell_division_mass": _b("cell_division_mass", "pg", 0.4),
    })
    nh4 = Solute(id="NH4", name="ammonium")
    o2 = Solute(id="O2", name="oxygen")
    dom = SpatialDomain(id="dom", dimensionality=2, extent_um=(32.0, 32.0),
                         periodic_axes=[0])
    top = Surface(id="top", name="top", axis=1, position="high")
    bot = Surface(id="bot", name="bot", axis=1, position="low")
    return ScientificModel(
        id="canonical_e2e_smoke",
        title="canonical E2E smoke",
        domain="microbial_biofilm",
        formalism="agent_based",
        entities=[aob, nh4, o2, dom, top, bot],
        processes=[
            MonodGrowthProcess(
                id="AOB_growth", growing_entity="AOB",
                consumed_solutes=["NH4", "O2"],
                parameters={
                    "mu_max": _b("mu_max", "1/day", 2.0),
                    "K_s_NH4": _b("K_s", "mg/L", 1.0),
                    "K_s_O2": _b("K_s", "mg/L", 0.5),
                    "Y_XS_NH4": _b("Y_XS", "g_biomass/g_substrate", 0.12),
                },
            ),
            DiffusionProcess(
                id="NH4_diff", solute="NH4", regions=["bulk_liquid", "biofilm"],
                parameters={
                    "D_bulk_liquid": _b("D_liquid", "um^2/s", 1800.0),
                    "D_biofilm": _b("D_biofilm", "um^2/s", 1440.0),
                },
            ),
            DiffusionProcess(
                id="O2_diff", solute="O2", regions=["bulk_liquid", "biofilm"],
                parameters={
                    "D_bulk_liquid": _b("D_liquid", "um^2/s", 2000.0),
                    "D_biofilm": _b("D_biofilm", "um^2/s", 1600.0),
                },
            ),
        ],
        boundary_conditions=[
            BoundaryCondition(id="NH4_top", target_entity="NH4",
                              surface="top", kind="dirichlet",
                              value=_b("S_bulk_initial", "mg/L", 30.0)),
            BoundaryCondition(id="O2_top", target_entity="O2",
                              surface="top", kind="dirichlet",
                              value=_b("S_bulk_initial", "mg/L", 8.0)),
        ],
        initial_conditions=[
            InitialCondition(
                id="AOB_IC", target_entity="AOB",
                kind="random_discrete_placement",
                parameters={
                    "n_agents": ParameterBinding(parameter_id="n", canonical_unit="dimensionless", point_estimate=10.0),
                    "layer_thickness_um": ParameterBinding(parameter_id="l", canonical_unit="um", point_estimate=2.0),
                },
            ),
        ],
        observables=[
            Observable(id="thickness", name="thickness",
                       kind="scalar_time_series", target="biofilm_thickness",
                       sampling=SamplingSpec(interval_s=60.0)),
        ],
        # Keep the run short — this test is a smoke test, not a benchmark.
        compute=ComputeBudget(time_horizon_s=300.0, timestep_hint_s=60.0),
    )


def test_drop_folder_on_ara_vm_and_run_idynomics(tmp_path):
    """User story: 'if I drop simtool-manage onto Ara with IDYNOMICS_2_JAR
    set, it should directly run iDynoMiCS 2 end to end.'

    The test kicks off a short biofilm simulation, confirms iDynoMiCS
    actually simulated (agent-count observables parsed), and confirms
    outputs were harvested into the run layout."""
    ir = _smoke_biofilm_ir()
    panel = Panel(
        id="canonical_e2e_panel",
        title="canonical e2e",
        user_id="ara-user",
        meta_model_id="",
        meta_model_version_pin="0.0.0",
        derived_ir=ir,
        constraints=UserConstraints(time_horizon_s=ir.compute.time_horizon_s),
    )
    summary = run_panel(
        panel, None,
        plugin=IDynoMiCS2Plugin(),
        run_root=tmp_path,
        auto_approve_assumptions=True,
    )
    assert summary.run_record.status == RunStatus.SUCCEEDED, (
        f"iDynoMiCS did not complete cleanly: {summary.run_record.failure_reason}"
    )
    # Monitor must have parsed real progress (not just the launch + exit frames).
    step_reports = [
        r for r in summary.progress_reports
        if r.timestep_index is not None
    ]
    assert step_reports, (
        "no structured progress reports parsed; monitor regex may be out of sync"
    )
    # Agent counts were observed.
    any_with_agents = [
        r for r in summary.progress_reports
        if r.observables.get("n_agents") is not None
    ]
    assert any_with_agents, "n_agents observable never surfaced"
    # Outputs were harvested (iDynoMiCS writes snapshots as XML).
    xml_files = list(summary.run_record.layout_root.rglob("*.xml"))
    assert len(xml_files) > 1, (
        "only the input protocol.xml exists — outputs weren't harvested"
    )
    # ODD protocol document was produced.
    assert Path(summary.protocol_doc_path).is_file()
