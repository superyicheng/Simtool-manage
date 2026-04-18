"""Plugin Protocol conformance + end-to-end lifecycle smoke.

These tests pin the contract the core orchestrator depends on. A mock
plugin exercises every method on FrameworkPlugin, we verify that the
pieces (IR, skill file, ledger, run layout, output bundle, protocol doc)
flow through end to end.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pytest

from simtool.connector.assumptions import (
    Assumption,
    AssumptionCategory,
    AssumptionLedger,
    AssumptionSeverity,
)
from simtool.connector.ir import (
    AgentPopulation,
    ComputeBudget,
    FirstOrderDecayProcess,
    MonodGrowthProcess,
    ParameterBinding,
    ScientificModel,
    Solute,
    SpatialDomain,
    Surface,
)
from simtool.connector.plugin import (
    DocSources,
    FrameworkPlugin,
    LoweredArtifact,
    RunHandle,
    ValidationIssue,
    ValidationReport,
)
from simtool.connector.runs import (
    OutputBundle,
    ProgressReport,
    ResourceUsage,
    RunLayout,
    RunRecord,
    RunStatus,
)
from simtool.connector.skill import (
    Grammar,
    GrammarElement,
    GrammarField,
    GrammarType,
    PipelineStage,
    SkillFile,
    SourceKind,
    SourceRef,
    StageReport,
)


# ---------------------------------------------------------------------------
# Mock plugin — the minimum viable implementation
# ---------------------------------------------------------------------------


class MockPlugin:
    """A deliberately small FrameworkPlugin implementation.

    Only supports ``formalism == 'agent_based'`` and only understands
    ``MonodGrowthProcess`` + ``FirstOrderDecayProcess``. Everything else is
    a validation error. That's representative of how a real plugin would
    behave: narrow, declarative, honest about what it cannot do.
    """

    name = "mock_plugin"
    version = "0.1.0"
    supported_framework_versions = ("0.1",)

    # Internal bookkeeping (not part of the Protocol).
    terminate_count: int

    def __init__(self) -> None:
        self.terminate_count = 0

    def parse_docs(self, sources: DocSources) -> SkillFile:
        return SkillFile(
            framework=self.name,
            framework_version="0.1",
            sources=list(sources.sources),
            grammar=Grammar(
                elements=[
                    GrammarElement(
                        name="agent",
                        fields=[
                            GrammarField(
                                name="id", type=GrammarType.STRING, required=True
                            ),
                        ],
                    ),
                ],
                root_element="agent",
            ),
            stage_reports=[
                StageReport(
                    stage=PipelineStage.CLASSIFY_SOURCES,
                    ok=True,
                    summary="mock",
                ),
                StageReport(
                    stage=PipelineStage.EXTRACT_GRAMMAR,
                    ok=True,
                    summary="mock",
                ),
            ],
        )

    def validate_ir(
        self, ir: ScientificModel, skill: SkillFile
    ) -> ValidationReport:
        issues: list[ValidationIssue] = []
        if ir.formalism != "agent_based":
            issues.append(
                ValidationIssue(
                    severity="error",
                    message=f"mock plugin only supports agent_based "
                    f"(got '{ir.formalism}')",
                    ir_path="formalism",
                )
            )
        supported_process_kinds = {
            "monod_growth",
            "first_order_decay",
            "diffusion",
            "maintenance",
            "eps_production",
        }
        for idx, p in enumerate(ir.processes):
            if p.kind not in supported_process_kinds:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        message=f"mock plugin does not support process kind "
                        f"'{p.kind}'",
                        ir_path=f"processes[{idx}]",
                        suggestion="rewrite as monod_growth or first_order_decay",
                    )
                )
        return ValidationReport(
            ok=all(i.severity != "error" for i in issues), issues=issues
        )

    def lower(
        self, ir: ScientificModel, skill: SkillFile
    ) -> LoweredArtifact:
        # Fake "code": a JSON dump of the IR + a trivial entrypoint.
        payload = ir.model_dump_json(indent=2).encode("utf-8")
        entry = b"# mock entrypoint - points at ir.json\n"
        files = [("ir.json", payload), ("run.sh", entry)]
        ledger = AssumptionLedger(
            ir_id=ir.id,
            framework=self.name,
            framework_version="0.1",
        )
        # Every real plugin surfaces at least timestep + boundary choices.
        ledger.add(
            Assumption(
                id="timestep",
                category=AssumptionCategory.NUMERICS,
                severity=AssumptionSeverity.MATERIAL,
                description="Fixed 60 s timestep.",
                justification="Default when IR carries no timestep_hint_s.",
                alternatives=["30 s", "adaptive"],
                surfaced_by="mock.lower.timestep_estimator",
                affects=[p.id for p in ir.processes],
            )
        )
        ledger.add(
            Assumption(
                id="well_mixed_bulk",
                category=AssumptionCategory.BOUNDARY,
                severity=AssumptionSeverity.CRITICAL,
                description="Dirichlet BCs -> well-mixed bulk assumed.",
                justification="No mass-transfer coefficient declared.",
                alternatives=["Robin BC with k_L"],
                surfaced_by="mock.lower.boundary_lowering",
                affects=[bc.id for bc in ir.boundary_conditions],
            )
        )
        return LoweredArtifact(
            entrypoint="run.sh",
            files=files,
            assumptions=ledger,
            extra={"ir_bytes": len(payload)},
        )

    def execute(
        self, artifact: LoweredArtifact, layout: RunLayout
    ) -> RunHandle:
        # Refuse to run if ledger not approved — real plugins delegate this
        # to the harness, but belt-and-suspenders is fine for the mock.
        if not artifact.assumptions.is_ready_to_run():
            raise RuntimeError(
                "cannot execute: assumption ledger not fully approved "
                f"({artifact.assumptions.blocking_reasons()})"
            )
        layout.ensure()
        # Materialize files under inputs/.
        for rel, content in artifact.files:
            (layout.inputs / rel).write_bytes(content)
        # Fake some "outputs" so parse_outputs has something to project.
        (layout.outputs / "thickness.csv").write_text(
            "t_s,thickness_um\n0,0.0\n3600,5.0\n7200,9.8\n"
        )
        return RunHandle(
            run_id=f"run-{artifact.extra.get('ir_bytes', 0)}",
            layout=layout,
            backend={"status": "started"},
        )

    def monitor(self, handle: RunHandle) -> Iterator[ProgressReport]:
        for i, (t, v) in enumerate([(0.0, 0.0), (3600.0, 5.0), (7200.0, 9.8)]):
            yield ProgressReport(
                run_id=handle.run_id,
                sim_time_s=t,
                sim_time_horizon_s=7200.0,
                timestep_index=i,
                timestep_total=2,
                wall_time_elapsed_s=float(i) * 2.0,
                observables={"thickness": v},
                message=f"tick {i}",
            )

    def terminate(self, handle: RunHandle, reason: str = "") -> None:
        self.terminate_count += 1

    def parse_outputs(self, layout: RunLayout) -> OutputBundle:
        series_path = layout.outputs / "thickness.csv"
        series: list[tuple[float, float]] = []
        if series_path.exists():
            lines = series_path.read_text().splitlines()[1:]
            for ln in lines:
                t, v = ln.split(",")
                series.append((float(t), float(v)))
        return OutputBundle(
            run_id="run-parse",
            scalar_time_series={"thickness": series},
        )

    def generate_protocol(
        self, ir: ScientificModel, layout: RunLayout
    ) -> Path:
        layout.ensure()
        doc = layout.protocol / "odd.md"
        doc.write_text(
            f"# ODD protocol for {ir.id}\n\n"
            f"Formalism: {ir.formalism}\n"
            f"Entities: {sorted(ir.entity_ids())}\n"
        )
        return doc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _binding(pid: str, unit: str, v: float) -> ParameterBinding:
    return ParameterBinding(parameter_id=pid, canonical_unit=unit, point_estimate=v)


@pytest.fixture
def tiny_ir() -> ScientificModel:
    return ScientificModel(
        id="tiny",
        title="tiny AOB-only biofilm",
        domain="microbial_biofilm",
        formalism="agent_based",
        entities=[
            AgentPopulation(id="AOB", name="AOB"),
            Solute(id="NH4", name="ammonium"),
            Solute(id="O2", name="oxygen"),
            SpatialDomain(id="dom", dimensionality=2, extent_um=(64.0, 64.0)),
            Surface(id="top", name="top", axis=1, position="high"),
            Surface(id="bottom", name="bottom", axis=1, position="low"),
        ],
        processes=[
            MonodGrowthProcess(
                id="AOB_growth",
                growing_entity="AOB",
                consumed_solutes=["NH4", "O2"],
                parameters={
                    "mu_max": _binding("mu_max", "1/day", 1.0),
                    "K_s_NH4": _binding("K_s", "mg/L", 1.0),
                    "K_s_O2": _binding("K_s", "mg/L", 0.5),
                    "Y_XS_NH4": _binding("Y_XS", "g/g", 0.12),
                },
            ),
            FirstOrderDecayProcess(
                id="AOB_decay",
                decaying_entity="AOB",
                parameters={"b": _binding("b", "1/day", 0.1)},
            ),
        ],
        boundary_conditions=[],
        initial_conditions=[],
        observables=[],
        compute=ComputeBudget(time_horizon_s=7200.0),
    )


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_mock_plugin_is_framework_plugin() -> None:
    assert isinstance(MockPlugin(), FrameworkPlugin)


def test_incomplete_plugin_is_not_framework_plugin() -> None:
    class BrokenPlugin:
        name = "broken"
        version = "0.0.1"
        # missing every method

    assert not isinstance(BrokenPlugin(), FrameworkPlugin)


def test_plugin_missing_one_method_is_not_framework_plugin() -> None:
    """Build a class that implements every FrameworkPlugin method EXCEPT
    generate_protocol — runtime_checkable Protocol must flag it."""

    class AlmostPlugin:
        name = "almost"
        version = "0.0.1"
        supported_framework_versions = ("0.1",)

        def parse_docs(self, sources):  # type: ignore[no-untyped-def]
            raise NotImplementedError

        def validate_ir(self, ir, skill):  # type: ignore[no-untyped-def]
            raise NotImplementedError

        def lower(self, ir, skill):  # type: ignore[no-untyped-def]
            raise NotImplementedError

        def execute(self, artifact, layout):  # type: ignore[no-untyped-def]
            raise NotImplementedError

        def monitor(self, handle):  # type: ignore[no-untyped-def]
            raise NotImplementedError

        def terminate(self, handle, reason=""):  # type: ignore[no-untyped-def]
            raise NotImplementedError

        def parse_outputs(self, layout):  # type: ignore[no-untyped-def]
            raise NotImplementedError

        # generate_protocol deliberately omitted.

    assert not isinstance(AlmostPlugin(), FrameworkPlugin)


# ---------------------------------------------------------------------------
# validate_ir logic
# ---------------------------------------------------------------------------


def test_validate_rejects_unsupported_formalism(tiny_ir: ScientificModel) -> None:
    # Swap formalism to something the plugin doesn't support, via a fresh build.
    ode_ir = ScientificModel(
        id="ode",
        title="ode",
        domain="x",
        formalism="ode",
        entities=tiny_ir.entities,
        processes=tiny_ir.processes,
        boundary_conditions=[],
        initial_conditions=[],
        observables=[],
        compute=ComputeBudget(time_horizon_s=100.0),
    )
    plugin = MockPlugin()
    skill = plugin.parse_docs(DocSources(sources=[]))
    report = plugin.validate_ir(ode_ir, skill)
    assert not report.ok
    assert any("agent_based" in i.message for i in report.errors())


def test_validate_accepts_supported_ir(tiny_ir: ScientificModel) -> None:
    plugin = MockPlugin()
    skill = plugin.parse_docs(DocSources(sources=[]))
    report = plugin.validate_ir(tiny_ir, skill)
    assert report.ok
    assert report.issues == []


def test_validate_is_read_only(tiny_ir: ScientificModel) -> None:
    """validate_ir MUST NOT mutate the IR. We assert by JSON-round-trip
    equivalence before and after."""
    plugin = MockPlugin()
    skill = plugin.parse_docs(DocSources(sources=[]))
    before = tiny_ir.model_dump_json()
    plugin.validate_ir(tiny_ir, skill)
    after = tiny_ir.model_dump_json()
    assert before == after


# ---------------------------------------------------------------------------
# lower — ledger is populated, unapproved blocks execute
# ---------------------------------------------------------------------------


def test_lower_produces_artifact_and_pending_ledger(
    tiny_ir: ScientificModel,
) -> None:
    plugin = MockPlugin()
    skill = plugin.parse_docs(DocSources(sources=[]))
    artifact = plugin.lower(tiny_ir, skill)
    assert artifact.entrypoint == "run.sh"
    assert any(name == "ir.json" for name, _ in artifact.files)
    assert len(artifact.assumptions.assumptions) >= 2
    assert not artifact.assumptions.is_ready_to_run()


def test_execute_blocks_while_assumptions_pending(
    tiny_ir: ScientificModel, tmp_path: Path
) -> None:
    plugin = MockPlugin()
    skill = plugin.parse_docs(DocSources(sources=[]))
    artifact = plugin.lower(tiny_ir, skill)
    layout = RunLayout.under(tmp_path / "run")
    with pytest.raises(RuntimeError, match="not fully approved"):
        plugin.execute(artifact, layout)


def test_execute_allowed_after_approval(
    tiny_ir: ScientificModel, tmp_path: Path
) -> None:
    plugin = MockPlugin()
    skill = plugin.parse_docs(DocSources(sources=[]))
    artifact = plugin.lower(tiny_ir, skill)
    for a in artifact.assumptions.assumptions:
        artifact.assumptions.approve(a.id)
    layout = RunLayout.under(tmp_path / "run")
    handle = plugin.execute(artifact, layout)
    assert handle.run_id.startswith("run-")
    # Files were materialized.
    assert (layout.inputs / "ir.json").exists()
    assert (layout.inputs / "run.sh").exists()
    # Round-trip the IR JSON from disk so we know the artifact survived.
    restored = ScientificModel.model_validate_json(
        (layout.inputs / "ir.json").read_text()
    )
    assert restored.id == tiny_ir.id


# ---------------------------------------------------------------------------
# monitor + terminate + parse_outputs + generate_protocol
# ---------------------------------------------------------------------------


def test_monitor_emits_structured_progress(
    tiny_ir: ScientificModel, tmp_path: Path
) -> None:
    plugin, artifact, handle = _launch(tiny_ir, tmp_path)
    reports = list(plugin.monitor(handle))
    assert len(reports) == 3
    assert reports[0].sim_time_s == 0.0
    assert reports[-1].observables["thickness"] == pytest.approx(9.8)
    # Every report carries the run id.
    assert all(r.run_id == handle.run_id for r in reports)


def test_terminate_is_idempotent(
    tiny_ir: ScientificModel, tmp_path: Path
) -> None:
    plugin, _, handle = _launch(tiny_ir, tmp_path)
    plugin.terminate(handle)
    plugin.terminate(handle, reason="user cancel")
    assert plugin.terminate_count == 2  # mock counts each call; must not raise


def test_parse_outputs_projects_to_bundle(
    tiny_ir: ScientificModel, tmp_path: Path
) -> None:
    plugin, _, handle = _launch(tiny_ir, tmp_path)
    # Consume monitor so the "run" is "done".
    list(plugin.monitor(handle))
    bundle = plugin.parse_outputs(handle.layout)
    assert "thickness" in bundle.scalar_time_series
    assert bundle.scalar_time_series["thickness"][-1][1] == pytest.approx(9.8)


def test_parse_outputs_tolerates_missing_outputs(tmp_path: Path) -> None:
    """Failed runs may produce no output files — parse_outputs must return
    an (empty) bundle, not raise."""
    plugin = MockPlugin()
    layout = RunLayout.under(tmp_path / "run")
    layout.ensure()
    bundle = plugin.parse_outputs(layout)
    assert bundle.scalar_time_series["thickness"] == []


def test_generate_protocol_writes_odd_doc(
    tiny_ir: ScientificModel, tmp_path: Path
) -> None:
    plugin = MockPlugin()
    layout = RunLayout.under(tmp_path / "run")
    layout.ensure()
    doc = plugin.generate_protocol(tiny_ir, layout)
    assert doc.exists()
    text = doc.read_text()
    assert "ODD protocol" in text
    assert tiny_ir.id in text


# ---------------------------------------------------------------------------
# End-to-end lifecycle + RunRecord assembly
# ---------------------------------------------------------------------------


def test_full_lifecycle_produces_run_record(
    tiny_ir: ScientificModel, tmp_path: Path
) -> None:
    """Happy path: parse_docs → validate → lower → approve → execute →
    monitor → parse_outputs → generate_protocol → write RunRecord.

    This is what the harness (not yet built) will orchestrate."""
    plugin = MockPlugin()
    skill = plugin.parse_docs(DocSources(sources=[]))
    report = plugin.validate_ir(tiny_ir, skill)
    assert report.ok

    artifact = plugin.lower(tiny_ir, skill)
    # User would review each assumption here.
    for a in artifact.assumptions.assumptions:
        artifact.assumptions.approve(a.id, user_note="reviewed")
    assert artifact.assumptions.is_ready_to_run()

    layout = RunLayout.under(tmp_path / "run-e2e")
    handle = plugin.execute(artifact, layout)

    final_progress: ProgressReport | None = None
    for r in plugin.monitor(handle):
        final_progress = r
    assert final_progress is not None

    bundle = plugin.parse_outputs(layout)
    protocol_doc = plugin.generate_protocol(tiny_ir, layout)

    # Assemble a RunRecord from the pieces.
    import hashlib
    ledger_bytes = artifact.assumptions.model_dump_json().encode("utf-8")
    ledger_hash = "sha256:" + hashlib.sha256(ledger_bytes).hexdigest()

    run = RunRecord(
        run_id=handle.run_id,
        ir_id=tiny_ir.id,
        framework=plugin.name,
        framework_version="0.1",
        skill_file_version=skill.skill_schema_version,
        assumption_ledger_hash=ledger_hash,
        layout_root=layout.root,
        status=RunStatus.SUCCEEDED,
        final_progress=final_progress,
        resource_usage=ResourceUsage(wall_time_s=final_progress.wall_time_elapsed_s),
    )

    # Persist metadata, verify round-trip.
    meta_path = layout.metadata / "run.json"
    layout.metadata.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(run.model_dump_json())
    restored = RunRecord.model_validate_json(meta_path.read_text())
    assert restored.status == RunStatus.SUCCEEDED
    assert restored.assumption_ledger_hash == ledger_hash

    # Artifacts are present on disk in the expected locations.
    assert (layout.inputs / "ir.json").exists()
    assert (layout.outputs / "thickness.csv").exists()
    assert protocol_doc.exists()
    assert json.loads((layout.inputs / "ir.json").read_text())["id"] == tiny_ir.id

    # And the output bundle projects to the comparator-friendly shape.
    assert len(bundle.scalar_time_series["thickness"]) == 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _launch(
    ir: ScientificModel, tmp_path: Path
) -> tuple[MockPlugin, LoweredArtifact, RunHandle]:
    plugin = MockPlugin()
    skill = plugin.parse_docs(DocSources(sources=[]))
    artifact = plugin.lower(ir, skill)
    for a in artifact.assumptions.assumptions:
        artifact.assumptions.approve(a.id)
    layout = RunLayout.under(tmp_path / "run")
    handle = plugin.execute(artifact, layout)
    return plugin, artifact, handle
