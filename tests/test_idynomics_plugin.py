"""Unit tests — IDynoMiCS2Plugin conformance + validate_ir rules."""

from __future__ import annotations

import pytest

from simtool.connector.ir import (
    AgentPopulation,
    ComputeBudget,
    CustomProcess,
    FirstOrderDecayProcess,
    MonodGrowthProcess,
    ParameterBinding,
    ScientificModel,
    Solute,
    SpatialDomain,
    Surface,
)
from simtool.connector.plugin import DocSources, FrameworkPlugin
from simtool.frameworks.idynomics_2 import IDynoMiCS2Plugin


def _b(pid, unit, v):
    return ParameterBinding(parameter_id=pid, canonical_unit=unit, point_estimate=v)


def _trivial_ir(**overrides) -> ScientificModel:
    aob = AgentPopulation(id="AOB", name="AOB")
    s = Solute(id="S", name="S")
    dom = SpatialDomain(id="dom", dimensionality=1, extent_um=(32.0,))
    top = Surface(id="top", name="top", axis=0, position="high")
    bot = Surface(id="bot", name="bot", axis=0, position="low")
    base = dict(
        id="t", title="t", domain="d", formalism="agent_based",
        entities=[aob, s, dom, top, bot],
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
    base.update(overrides)
    return ScientificModel(**base)


def test_plugin_is_framework_plugin():
    assert isinstance(IDynoMiCS2Plugin(), FrameworkPlugin)


def test_parse_docs_returns_populated_skill():
    plugin = IDynoMiCS2Plugin()
    skill = plugin.parse_docs(DocSources(sources=[]))
    assert skill.framework == "idynomics_2"
    assert skill.grammar.elements
    assert any(e.name == "simulation" for e in skill.grammar.elements)
    assert len(skill.stage_reports) >= 2


def test_validate_accepts_supported_ir():
    plugin = IDynoMiCS2Plugin()
    skill = plugin.parse_docs(DocSources(sources=[]))
    report = plugin.validate_ir(_trivial_ir(), skill)
    assert report.ok
    assert report.issues == []


def test_validate_rejects_unsupported_formalism():
    plugin = IDynoMiCS2Plugin()
    skill = plugin.parse_docs(DocSources(sources=[]))
    report = plugin.validate_ir(_trivial_ir(formalism="molecular_dynamics"), skill)
    assert not report.ok
    assert any("formalism" in i.message for i in report.errors())


def test_validate_rejects_custom_process_with_suggestion():
    custom = CustomProcess(id="mystery", description="?", actors=["AOB"])
    ir = _trivial_ir(processes=[
        MonodGrowthProcess(
            id="g", growing_entity="AOB", consumed_solutes=["S"],
            parameters={
                "mu_max": _b("mu_max", "1/day", 1.0),
                "K_s_S": _b("K_s", "mg/L", 1.0),
            },
        ),
        custom,
    ])
    plugin = IDynoMiCS2Plugin()
    skill = plugin.parse_docs(DocSources(sources=[]))
    report = plugin.validate_ir(ir, skill)
    assert not report.ok
    assert any("CustomProcess" in i.message for i in report.errors())
    assert any(i.suggestion for i in report.errors())


def test_validate_flags_missing_mu_max():
    aob = AgentPopulation(id="AOB", name="AOB")
    s = Solute(id="S", name="S")
    dom = SpatialDomain(id="dom", dimensionality=1, extent_um=(32.0,))
    top = Surface(id="top", name="top", axis=0, position="high")
    bot = Surface(id="bot", name="bot", axis=0, position="low")
    ir = ScientificModel(
        id="t", title="t", domain="d", formalism="agent_based",
        entities=[aob, s, dom, top, bot],
        processes=[
            MonodGrowthProcess(
                id="g", growing_entity="AOB", consumed_solutes=["S"],
                parameters={"K_s_S": _b("K_s", "mg/L", 1.0)},
            ),
        ],
        compute=ComputeBudget(time_horizon_s=3600.0),
    )
    plugin = IDynoMiCS2Plugin()
    skill = plugin.parse_docs(DocSources(sources=[]))
    report = plugin.validate_ir(ir, skill)
    assert not report.ok
    assert any("mu_max" in i.message for i in report.errors())


def test_validate_flags_missing_k_s_per_solute():
    ir = _trivial_ir()
    # Remove the K_s_S binding from the single growth process.
    ir.processes[0].parameters = {"mu_max": _b("mu_max", "1/day", 1.0)}
    plugin = IDynoMiCS2Plugin()
    skill = plugin.parse_docs(DocSources(sources=[]))
    report = plugin.validate_ir(ir, skill)
    assert not report.ok
    assert any("K_s_S" in i.message for i in report.errors())


def test_lower_produces_protocol_and_ledger():
    plugin = IDynoMiCS2Plugin()
    skill = plugin.parse_docs(DocSources(sources=[]))
    artifact = plugin.lower(_trivial_ir(), skill)
    assert artifact.entrypoint == "protocol.xml"
    assert any(name == "protocol.xml" for name, _ in artifact.files)
    assert artifact.assumptions.assumptions
    # Every lowering surfaces at least the default_process_managers
    # assumption the plugin adds post-hoc.
    assert any(a.id == "default_process_managers"
               for a in artifact.assumptions.assumptions)


def test_parse_outputs_on_empty_layout(tmp_path):
    from simtool.connector.runs import RunLayout
    layout = RunLayout.under(tmp_path / "run")
    layout.ensure()
    plugin = IDynoMiCS2Plugin()
    bundle = plugin.parse_outputs(layout)
    # Empty but well-formed.
    assert bundle.scalar_time_series == {}
    assert bundle.spatial_field_paths == {}


def test_generate_protocol_writes_odd(tmp_path):
    from simtool.connector.runs import RunLayout
    plugin = IDynoMiCS2Plugin()
    layout = RunLayout.under(tmp_path / "run")
    path = plugin.generate_protocol(_trivial_ir(), layout)
    assert path.is_file()
    assert "ODD Protocol" in path.read_text()


def test_execute_raises_when_ledger_not_approved(tmp_path, monkeypatch):
    # Force jar to resolve to a fake existing file so execute reaches the
    # ledger check rather than bailing on missing jar.
    fake_jar = tmp_path / "jar" / "iDyno.jar"
    fake_jar.parent.mkdir()
    fake_jar.write_bytes(b"x")
    monkeypatch.setenv("IDYNOMICS_2_JAR", str(fake_jar))
    plugin = IDynoMiCS2Plugin()
    skill = plugin.parse_docs(DocSources(sources=[]))
    artifact = plugin.lower(_trivial_ir(), skill)
    from simtool.connector.runs import RunLayout
    layout = RunLayout.under(tmp_path / "run")
    with pytest.raises(RuntimeError, match="not fully approved"):
        plugin.execute(artifact, layout)
