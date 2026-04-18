"""Unit tests — iDynoMiCS 2 lowering, config, monitor regex, ODD."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from unittest.mock import MagicMock

import pytest

from simtool.connector.ir import (
    AgentPopulation,
    BoundaryCondition,
    ComputeBudget,
    CustomProcess,
    DiffusionProcess,
    Distribution,
    FirstOrderDecayProcess,
    InitialCondition,
    MonodGrowthProcess,
    ParameterBinding,
    ScientificModel,
    Solute,
    SpatialDomain,
    Surface,
)
from simtool.connector.runs import ProgressReport, RunLayout
from simtool.frameworks.idynomics_2.config import (
    IDynoMiCS2Config,
    IDynoMiCS2NotAvailable,
    require_runtime,
    resolve_jar_path,
    resolve_java_bin,
)
from simtool.frameworks.idynomics_2.lower import lower_ir
from simtool.frameworks.idynomics_2.monitor import _parse_line
from simtool.frameworks.idynomics_2.odd import generate_odd
from simtool.frameworks.idynomics_2.outputs import (
    extract_simulation_name,
    harvest_jar_results,
)


# ---------------------------------------------------------------------------
# Config / jar resolution
# ---------------------------------------------------------------------------


def test_resolve_jar_path_from_env(tmp_path, monkeypatch):
    fake_jar = tmp_path / "iDynoMiCS-2.0.jar"
    fake_jar.write_bytes(b"not really a jar")
    monkeypatch.setenv("IDYNOMICS_2_JAR", str(fake_jar))
    assert resolve_jar_path() == fake_jar


def test_resolve_jar_path_from_config_file(tmp_path, monkeypatch):
    fake_jar = tmp_path / "iDyno.jar"
    fake_jar.write_bytes(b"x")
    cfg_path = tmp_path / "idynomics_2.json"
    cfg_path.write_text(f'{{"jar_path": "{fake_jar}"}}')
    monkeypatch.delenv("IDYNOMICS_2_JAR", raising=False)
    # Point directly at the fake config path.
    assert resolve_jar_path(config_path=cfg_path) == fake_jar


def test_resolve_jar_path_none_when_nothing(monkeypatch, tmp_path):
    monkeypatch.delenv("IDYNOMICS_2_JAR", raising=False)
    missing_cfg = tmp_path / "noconfig.json"
    # Can't easily mock _DEFAULT_SEARCH_PATHS hits, but unless /opt/idynomics
    # or ./vendor exist, this will return None on the CI path.
    result = resolve_jar_path(config_path=missing_cfg)
    # accept None or a real jar if the dev has one installed; the function
    # must at least not raise.
    assert result is None or result.is_file()


def test_require_runtime_raises_when_jar_missing(monkeypatch, tmp_path):
    # Force every resolution path to miss.
    monkeypatch.setenv("IDYNOMICS_2_JAR", str(tmp_path / "nope.jar"))
    from simtool.frameworks.idynomics_2 import config as cfg_mod

    def _fake_resolve(config_path=None):
        return None

    monkeypatch.setattr(cfg_mod, "resolve_jar_path", _fake_resolve)
    with pytest.raises(IDynoMiCS2NotAvailable, match="iDynoMiCS 2 jar not found"):
        require_runtime()


def test_as_command_includes_protocol_flag(tmp_path):
    jar = tmp_path / "j.jar"
    jar.write_bytes(b"x")
    cfg = IDynoMiCS2Config(jar_path=jar, java_bin="/usr/bin/java")
    cmd = cfg.as_command(Path("/tmp/p.xml"))
    assert "-protocol" in cmd
    assert "/tmp/p.xml" in cmd
    # -protocol must come immediately before the protocol path.
    assert cmd.index("-protocol") == cmd.index("/tmp/p.xml") - 1


# ---------------------------------------------------------------------------
# Lowering — biofilm path
# ---------------------------------------------------------------------------


def _b(pid, unit, v, **ctx):
    return ParameterBinding(parameter_id=pid, canonical_unit=unit,
                            point_estimate=v, context_keys=dict(ctx))


def _make_biofilm_ir() -> ScientificModel:
    aob = AgentPopulation(id="AOB", name="AOB", parameters={
        "cell_density": _b("cell_density", "pg/fL", 0.15),
        "cell_initial_mass": _b("cell_initial_mass", "pg", 0.2),
        "cell_division_mass": _b("cell_division_mass", "pg", 0.4),
    })
    nh4 = Solute(id="NH4", name="NH4")
    o2 = Solute(id="O2", name="O2")
    dom = SpatialDomain(id="dom", dimensionality=2, extent_um=(32.0, 32.0),
                         periodic_axes=[0])
    top = Surface(id="top", name="top", axis=1, position="high")
    bot = Surface(id="bot", name="bot", axis=1, position="low")
    return ScientificModel(
        id="test_biofilm", title="t", domain="d", formalism="agent_based",
        entities=[aob, nh4, o2, dom, top, bot],
        processes=[
            MonodGrowthProcess(
                id="g", growing_entity="AOB", consumed_solutes=["NH4", "O2"],
                parameters={
                    "mu_max": _b("mu_max", "1/day", 1.0),
                    "K_s_NH4": _b("K_s", "mg/L", 1.0),
                    "K_s_O2": _b("K_s", "mg/L", 0.5),
                    "Y_XS_NH4": _b("Y_XS", "g_biomass/g_substrate", 0.12),
                },
            ),
            FirstOrderDecayProcess(
                id="d", decaying_entity="AOB",
                parameters={"b": _b("b", "1/day", 0.1)},
            ),
            DiffusionProcess(
                id="diff_nh4", solute="NH4", regions=["bulk_liquid", "biofilm"],
                parameters={
                    "D_bulk_liquid": _b("D_liquid", "um^2/s", 1800.0),
                    "D_biofilm": _b("D_biofilm", "um^2/s", 1440.0),
                },
            ),
        ],
        boundary_conditions=[
            BoundaryCondition(
                id="NH4_top", target_entity="NH4", surface="top", kind="dirichlet",
                value=_b("S_bulk_initial", "mg/L", 30.0),
            ),
        ],
        initial_conditions=[
            InitialCondition(
                id="ic", target_entity="AOB", kind="random_discrete_placement",
                parameters={
                    "n_agents": ParameterBinding(parameter_id="n", canonical_unit="dimensionless", point_estimate=10.0),
                    "layer_thickness_um": ParameterBinding(parameter_id="l", canonical_unit="um", point_estimate=2.0),
                },
            ),
        ],
        observables=[],
        compute=ComputeBudget(time_horizon_s=3600.0, timestep_hint_s=60.0),
    )


def test_lower_biofilm_produces_valid_xml_headers():
    ir = _make_biofilm_ir()
    res = lower_ir(ir)
    xml = res.protocol_xml.decode("utf-8")
    assert xml.startswith('<?xml')
    assert "<simulation" in xml
    assert 'name="test_biofilm"' in xml
    assert "<compartment" in xml
    assert '<shape class="Rectangle"' in xml


def test_lower_biofilm_includes_growth_reaction():
    ir = _make_biofilm_ir()
    xml = lower_ir(ir).protocol_xml.decode("utf-8")
    assert 'reaction name="g"' in xml
    assert "mass*mumax*(NH4/(NH4+Ks_NH4))*(O2/(O2+Ks_O2))" in xml
    assert 'name="mumax" value="1[d-1]"' in xml
    assert 'name="Ks_NH4" value="1[mg/l]"' in xml


def test_lower_biofilm_inserts_decay_reaction():
    ir = _make_biofilm_ir()
    xml = lower_ir(ir).protocol_xml.decode("utf-8")
    assert 'reaction name="d"' in xml
    assert "mass*b" in xml
    assert 'name="b" value="0.1[d-1]"' in xml
    # Decay insertion surfaces an advisory assumption.
    assert any(a.id.startswith("decay_reaction_inserted") for a in lower_ir(ir).ledger.assumptions)


def test_lower_biofilm_solute_diffusivities_attached():
    ir = _make_biofilm_ir()
    xml = lower_ir(ir).protocol_xml.decode("utf-8")
    assert 'defaultDiffusivity="1800[um+2/s]"' in xml
    assert 'biofilmDiffusivity="1440[um+2/s]"' in xml


def test_lower_biofilm_dirichlet_bc_emitted():
    ir = _make_biofilm_ir()
    xml = lower_ir(ir).protocol_xml.decode("utf-8")
    assert 'class="FixedBoundary"' in xml
    assert 'concentration="30[mg/l]"' in xml


def test_lower_biofilm_spawn_block_emitted():
    ir = _make_biofilm_ir()
    xml = lower_ir(ir).protocol_xml.decode("utf-8")
    assert "randomSpawner" in xml
    assert 'number="10"' in xml


def test_lower_distribution_binding_flattens_with_assumption():
    aob = AgentPopulation(id="AOB", name="AOB")
    dom = SpatialDomain(id="dom", dimensionality=1, extent_um=(32.0,))
    sol = Solute(id="S", name="S")
    o2 = Solute(id="O2", name="O2")
    top = Surface(id="top", name="top", axis=0, position="high")
    bot = Surface(id="bot", name="bot", axis=0, position="low")
    ir = ScientificModel(
        id="dist_test", title="t", domain="d", formalism="agent_based",
        entities=[aob, sol, o2, dom, top, bot],
        processes=[
            MonodGrowthProcess(
                id="g", growing_entity="AOB", consumed_solutes=["S", "O2"],
                parameters={
                    "mu_max": ParameterBinding(
                        parameter_id="mu_max", canonical_unit="1/day",
                        distribution=Distribution(
                            shape="empirical", samples=[0.8, 1.0, 1.2],
                        ),
                    ),
                    "K_s_S": _b("K_s", "mg/L", 1.0),
                    "K_s_O2": _b("K_s", "mg/L", 0.5),
                },
            ),
        ],
        compute=ComputeBudget(time_horizon_s=3600.0),
    )
    result = lower_ir(ir)
    ledger_ids = [a.id for a in result.ledger.assumptions]
    assert any("flatten_distribution" in aid for aid in ledger_ids)
    xml = result.protocol_xml.decode("utf-8")
    # Central value of empirical [0.8,1.0,1.2] is 1.0.
    assert 'name="mumax" value="1[d-1]"' in xml


def test_lower_chemostat_path():
    # ODE formalism should use Dimensionless shape + ChemostatSolver.
    aob = AgentPopulation(id="AOB", name="AOB")
    s = Solute(id="S", name="S")
    ir = ScientificModel(
        id="cs", title="cs", domain="d", formalism="ode",
        entities=[aob, s],
        processes=[
            MonodGrowthProcess(
                id="g", growing_entity="AOB", consumed_solutes=["S"],
                parameters={
                    "mu_max": _b("mu_max", "1/day", 1.0),
                    "K_s_S": _b("K_s", "mg/L", 1.0),
                },
            ),
        ],
        compute=ComputeBudget(time_horizon_s=3600.0),
    )
    xml = lower_ir(ir).protocol_xml.decode("utf-8")
    assert '<shape class="Dimensionless"' in xml
    assert "ChemostatSolver" in xml


def test_lower_unsupported_formalism_raises():
    ir = ScientificModel(
        id="md", title="x", domain="x", formalism="molecular_dynamics",
        entities=[], processes=[],
        compute=ComputeBudget(time_horizon_s=1.0),
    )
    with pytest.raises(ValueError, match="does not support formalism"):
        lower_ir(ir)


# ---------------------------------------------------------------------------
# Monitor parsing
# ---------------------------------------------------------------------------


def _make_handle():
    from simtool.connector.plugin import RunHandle
    backend = MagicMock()
    backend.started_at = time.time() - 10
    return RunHandle(run_id="r", layout=MagicMock(), backend=backend)


def test_monitor_parses_step_progress_line():
    last = ProgressReport(run_id="r")
    line = "[16:07] #75 time: 13320.0 step: 180.0 end: 14400.0"
    report = _parse_line(_make_handle(), MagicMock(started_at=time.time()), line, last)
    assert report is not None
    assert report.timestep_index == 75
    assert report.sim_time_s == pytest.approx(13320.0)
    assert report.sim_time_horizon_s == pytest.approx(14400.0)
    assert report.timestep_total == 80  # 14400 / 180


def test_monitor_parses_agent_count_line():
    last = ProgressReport(run_id="r")
    line = "[16:07] biofilm-compartment contains 1042 agents"
    report = _parse_line(_make_handle(), MagicMock(started_at=time.time()), line, last)
    assert report is not None
    assert report.observables.get("n_agents") == 1042.0


def test_monitor_ignores_unrecognized_line():
    last = ProgressReport(run_id="r")
    line = "some debug line with no numbers"
    report = _parse_line(_make_handle(), MagicMock(started_at=time.time()), line, last)
    assert report is None


# ---------------------------------------------------------------------------
# ODD generation
# ---------------------------------------------------------------------------


def test_odd_written_under_path(tmp_path):
    ir = _make_biofilm_ir()
    out = tmp_path / "protocol" / "ODD.md"
    result = generate_odd(ir, out)
    assert result == out
    text = out.read_text()
    assert "ODD Protocol" in text
    assert ir.id in text
    assert ir.formalism in text
    assert "Monod growth" in text  # process description rendered


# ---------------------------------------------------------------------------
# Output harvesting
# ---------------------------------------------------------------------------


def test_harvest_jar_results_copies_matching_dir(tmp_path):
    jar_dir = tmp_path / "idynomics"
    (jar_dir / "results" / "2026.04.18_mysim_biofilm").mkdir(parents=True)
    (jar_dir / "results" / "2026.04.18_mysim_biofilm" / "snapshot_00001.xml").write_text("<x/>")
    layout = RunLayout.under(tmp_path / "run")
    layout.ensure()
    result = harvest_jar_results(jar_dir, "mysim_biofilm", layout)
    assert result is not None
    assert (result / "snapshot_00001.xml").is_file()


def test_harvest_jar_results_no_match(tmp_path):
    jar_dir = tmp_path / "idynomics"
    (jar_dir / "results").mkdir(parents=True)
    layout = RunLayout.under(tmp_path / "run")
    layout.ensure()
    assert harvest_jar_results(jar_dir, "anything", layout) is None


def test_extract_simulation_name_from_protocol(tmp_path):
    p = tmp_path / "protocol.xml"
    p.write_text('<?xml version="1.0"?><document><simulation name="my_sim" log="NORMAL"/></document>')
    assert extract_simulation_name(p) == "my_sim"


def test_extract_simulation_name_missing_file_returns_none(tmp_path):
    assert extract_simulation_name(tmp_path / "nope.xml") is None
