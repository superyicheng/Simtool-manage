"""End-to-end run orchestration.

``run_panel`` is the top-level Ara-facing entry point:

    summary = run_panel(panel, meta_model, plugin=IDynoMiCS2Plugin(),
                        run_root=Path("~/simtool/runs/xyz"),
                        auto_approve_assumptions=False)

It validates the panel's derived IR against the plugin, lowers it,
optionally auto-approves assumptions (with a loud audit trail in the
summary), materializes inputs and launches the run, streams progress
reports into a collected list, parses outputs, generates the ODD
protocol, and writes a RunRecord to disk.

The orchestrator also feeds the resulting RunHistoryEntry back to the
panel so the panel's run log stays current.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from pydantic import BaseModel, Field

from simtool.connector.assumptions import AssumptionLedger
from simtool.connector.ir import ScientificModel
from simtool.connector.plugin import (
    DocSources,
    FrameworkPlugin,
    LoweredArtifact,
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
from simtool.metamodel import MetaModel
from simtool.panels import Panel, RunHistoryEntry


class RunExecutionSummary(BaseModel):
    """What the orchestrator returns to the caller.

    Includes the structured outputs the caller needs to render replies
    (progress reports, output bundle, protocol path) and the RunRecord
    path for persistence.
    """

    run_record: RunRecord
    progress_reports: list[ProgressReport] = Field(default_factory=list)
    output_bundle: OutputBundle
    protocol_doc_path: str
    validation_report: ValidationReport
    auto_approved_assumption_ids: list[str] = Field(default_factory=list)
    ledger_snapshot: AssumptionLedger


def run_panel(
    panel: Panel,
    meta_model: Optional[MetaModel],
    *,
    plugin: FrameworkPlugin,
    run_root: Path,
    auto_approve_assumptions: bool = False,
    doc_sources: Optional[DocSources] = None,
) -> RunExecutionSummary:
    """Orchestrate a full run of ``panel.derived_ir`` through ``plugin``.

    When ``auto_approve_assumptions`` is True, every pending assumption
    is approved with a ``user_note="auto-approved by run_panel"``. This
    is intended for scripted Ara sessions where the agent has already
    presented the ledger to the user and received blanket confirmation.
    The approved assumption ids are returned in the summary so the audit
    trail is complete.

    ``meta_model`` is optional; when supplied, it's used to tag the
    RunRecord metadata with the meta-model version.
    """
    run_id = f"run-{panel.id}-{uuid.uuid4().hex[:8]}"
    layout = RunLayout.under(run_root / run_id)
    layout.ensure()

    skill = plugin.parse_docs(doc_sources or DocSources(sources=[]))
    validation = plugin.validate_ir(panel.derived_ir, skill)
    if not validation.ok:
        return _failed_before_launch(
            run_id, panel, meta_model, plugin, layout, skill,
            validation, reason="IR failed plugin validation",
        )

    artifact = plugin.lower(panel.derived_ir, skill)
    auto_approved: list[str] = []
    if auto_approve_assumptions:
        for a in artifact.assumptions.pending():
            artifact.assumptions.approve(a.id, user_note="auto-approved by run_panel")
            auto_approved.append(a.id)

    if not artifact.assumptions.is_ready_to_run():
        return _failed_before_launch(
            run_id, panel, meta_model, plugin, layout, skill, validation,
            reason=(
                "assumption ledger not fully approved; "
                f"blocking: {artifact.assumptions.blocking_reasons()}"
            ),
            artifact=artifact,
            auto_approved=auto_approved,
        )

    record = _new_run_record(
        run_id, panel, meta_model, plugin, layout, skill, artifact,
    )
    record.status = RunStatus.RUNNING
    record.started_at = datetime.now(timezone.utc)

    try:
        handle = plugin.execute(artifact, layout)
    except Exception as exc:
        record.status = RunStatus.FAILED
        record.ended_at = datetime.now(timezone.utc)
        record.failure_reason = f"execute() raised: {exc!r}"
        _persist(record, layout)
        return RunExecutionSummary(
            run_record=record,
            output_bundle=OutputBundle(run_id=run_id),
            protocol_doc_path="",
            validation_report=validation,
            auto_approved_assumption_ids=auto_approved,
            ledger_snapshot=artifact.assumptions,
        )

    reports: list[ProgressReport] = []
    try:
        for r in plugin.monitor(handle):
            reports.append(r)
    except Exception as exc:
        record.status = RunStatus.FAILED
        record.failure_reason = f"monitor() raised: {exc!r}"
        plugin.terminate(handle, reason="monitor failure")
    finally:
        record.ended_at = datetime.now(timezone.utc)
        if record.status == RunStatus.RUNNING:
            # Determine final status from the process exit.
            exit_code = getattr(handle.backend, "process", None)
            if exit_code is not None and hasattr(exit_code, "returncode"):
                record.exit_code = exit_code.returncode
                record.status = (
                    RunStatus.SUCCEEDED
                    if exit_code.returncode == 0
                    else RunStatus.FAILED
                )
            else:
                record.status = RunStatus.SUCCEEDED

    if reports:
        record.final_progress = reports[-1]
        wall = reports[-1].wall_time_elapsed_s
        record.resource_usage = ResourceUsage(wall_time_s=wall)

    bundle = plugin.parse_outputs(layout)
    protocol_path = plugin.generate_protocol(panel.derived_ir, layout)

    _persist(record, layout)

    panel.attach_run(RunHistoryEntry(
        run_id=run_id,
        ir_id=panel.derived_ir.id,
        framework=plugin.name,
        framework_version=plugin.supported_framework_versions[0],
        run_record_path=str(
            (layout.metadata / "run.json").relative_to(layout.root.parent)
        ) if layout.root.parent in (layout.metadata / "run.json").parents
        else "metadata/run.json",
        status=record.status.value,
    ))

    return RunExecutionSummary(
        run_record=record,
        progress_reports=reports,
        output_bundle=bundle,
        protocol_doc_path=str(protocol_path),
        validation_report=validation,
        auto_approved_assumption_ids=auto_approved,
        ledger_snapshot=artifact.assumptions,
    )


def run_scientific_model(
    ir: ScientificModel,
    *,
    plugin: FrameworkPlugin,
    run_root: Path,
    auto_approve_assumptions: bool = False,
) -> RunExecutionSummary:
    """Convenience wrapper for running an IR directly (no panel context)."""
    minimal_panel = Panel(
        id=f"adhoc-{uuid.uuid4().hex[:8]}",
        title=f"ad-hoc run of {ir.id}",
        user_id="system",
        meta_model_id="",
        meta_model_version_pin="0.0.0",
        derived_ir=ir,
        constraints=__import__(
            "simtool.panels", fromlist=["UserConstraints"]
        ).UserConstraints(time_horizon_s=ir.compute.time_horizon_s),
    )
    return run_panel(
        minimal_panel, None,
        plugin=plugin, run_root=run_root,
        auto_approve_assumptions=auto_approve_assumptions,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _new_run_record(
    run_id: str,
    panel: Panel,
    meta_model: Optional[MetaModel],
    plugin: FrameworkPlugin,
    layout: RunLayout,
    skill,
    artifact: LoweredArtifact,
) -> RunRecord:
    ledger_bytes = artifact.assumptions.model_dump_json().encode("utf-8")
    ledger_hash = "sha256:" + hashlib.sha256(ledger_bytes).hexdigest()
    extra = {}
    if meta_model is not None:
        extra["meta_model_id"] = meta_model.id
        extra["meta_model_version"] = str(meta_model.version)
    return RunRecord(
        run_id=run_id,
        ir_id=panel.derived_ir.id,
        framework=plugin.name,
        framework_version=plugin.supported_framework_versions[0],
        skill_file_version=skill.skill_schema_version,
        assumption_ledger_hash=ledger_hash,
        layout_root=layout.root,
        extra=extra,
    )


def _failed_before_launch(
    run_id: str,
    panel: Panel,
    meta_model: Optional[MetaModel],
    plugin: FrameworkPlugin,
    layout: RunLayout,
    skill,
    validation: ValidationReport,
    *,
    reason: str,
    artifact: Optional[LoweredArtifact] = None,
    auto_approved: Optional[list[str]] = None,
) -> RunExecutionSummary:
    if artifact is not None:
        ledger = artifact.assumptions
    else:
        ledger = AssumptionLedger(
            ir_id=panel.derived_ir.id,
            framework=plugin.name,
            framework_version=plugin.supported_framework_versions[0],
        )
    record = RunRecord(
        run_id=run_id,
        ir_id=panel.derived_ir.id,
        framework=plugin.name,
        framework_version=plugin.supported_framework_versions[0],
        skill_file_version=skill.skill_schema_version,
        assumption_ledger_hash="sha256:(not_launched)",
        layout_root=layout.root,
        status=RunStatus.FAILED,
        failure_reason=reason,
        ended_at=datetime.now(timezone.utc),
        extra={"meta_model_id": meta_model.id} if meta_model else {},
    )
    _persist(record, layout)
    return RunExecutionSummary(
        run_record=record,
        output_bundle=OutputBundle(run_id=run_id),
        protocol_doc_path="",
        validation_report=validation,
        auto_approved_assumption_ids=auto_approved or [],
        ledger_snapshot=ledger,
    )


def _persist(record: RunRecord, layout: RunLayout) -> None:
    layout.metadata.mkdir(parents=True, exist_ok=True)
    (layout.metadata / "run.json").write_text(record.model_dump_json(indent=2))
