"""Run-harness datatype tests.

Covers: RunLayout directory contract, ProgressReport/OutputBundle
JSON round-trip fidelity (tuples survive), RunRecord lifecycle states,
Path handling.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from simtool.connector.runs import (
    Artifact,
    OutputBundle,
    ProgressReport,
    ResourceUsage,
    RunLayout,
    RunRecord,
    RunStatus,
)


# ---------------------------------------------------------------------------
# RunLayout
# ---------------------------------------------------------------------------


def test_run_layout_under_assembles_standard_dirs(tmp_path: Path) -> None:
    root = tmp_path / "run-001"
    layout = RunLayout.under(root)
    assert layout.root == root
    assert layout.inputs == root / "inputs"
    assert layout.outputs == root / "outputs"
    assert layout.logs == root / "logs"
    assert layout.snapshots == root / "snapshots"
    assert layout.protocol == root / "protocol"
    assert layout.metadata == root / "metadata"


def test_run_layout_ensure_creates_every_dir(tmp_path: Path) -> None:
    layout = RunLayout.under(tmp_path / "r")
    layout.ensure()
    for p in (
        layout.root, layout.inputs, layout.outputs,
        layout.logs, layout.snapshots, layout.protocol, layout.metadata,
    ):
        assert p.is_dir()


def test_run_layout_ensure_is_idempotent(tmp_path: Path) -> None:
    layout = RunLayout.under(tmp_path / "r")
    layout.ensure()
    # Drop a file into inputs to confirm ensure() doesn't wipe it.
    sentinel = layout.inputs / "sentinel.txt"
    sentinel.write_text("keep-me")
    layout.ensure()  # call twice
    assert sentinel.read_text() == "keep-me"


def test_run_layout_ensure_on_existing_tree_no_error(tmp_path: Path) -> None:
    (tmp_path / "pre-existing" / "inputs").mkdir(parents=True)
    layout = RunLayout.under(tmp_path / "pre-existing")
    layout.ensure()  # should not raise


# ---------------------------------------------------------------------------
# ProgressReport
# ---------------------------------------------------------------------------


def test_progress_report_defaults_minimal() -> None:
    r = ProgressReport(run_id="r-1")
    assert r.run_id == "r-1"
    assert r.wall_time_elapsed_s == 0.0
    assert r.observables == {}
    assert r.ts.tzinfo is not None  # timezone-aware


def test_progress_report_round_trip() -> None:
    r = ProgressReport(
        run_id="r-2",
        sim_time_s=3600.0,
        sim_time_horizon_s=86400.0,
        timestep_index=60,
        timestep_total=1440,
        wall_time_elapsed_s=12.5,
        wall_time_estimated_remaining_s=287.5,
        memory_rss_gb=1.2,
        cpu_percent=95.0,
        observables={"biofilm_thickness": 12.4, "N_flux": 0.028},
        message="steady state reached",
    )
    restored = ProgressReport.model_validate_json(r.model_dump_json())
    assert restored == r
    assert restored.observables["biofilm_thickness"] == pytest.approx(12.4)


def test_progress_report_only_scalar_observables() -> None:
    """ProgressReport MUST NOT carry spatial fields or distributions —
    those belong in snapshots / OutputBundle. Schema-level check: the
    observables field is dict[str, float]."""
    # A list/dict value should be rejected by pydantic.
    with pytest.raises(ValidationError):
        ProgressReport(
            run_id="r",
            observables={"field": [1.0, 2.0, 3.0]},  # type: ignore[dict-item]
        )


# ---------------------------------------------------------------------------
# OutputBundle
# ---------------------------------------------------------------------------


def test_output_bundle_defaults_empty() -> None:
    b = OutputBundle(run_id="r")
    assert b.scalar_time_series == {}
    assert b.flux_time_series == {}
    assert b.spatial_field_paths == {}
    assert b.distributions == {}


def test_output_bundle_round_trip_preserves_tuples() -> None:
    """Tuples are JSON-serialized as arrays; pydantic must re-tuple them
    on load (or keep them structurally equivalent)."""
    b = OutputBundle(
        run_id="r",
        scalar_time_series={
            "thickness": [(0.0, 0.0), (3600.0, 5.0), (7200.0, 9.8)],
            "n_agents_AOB": [(0.0, 50.0), (3600.0, 72.0)],
        },
        flux_time_series={"NH4_flux_top": [(3600.0, -0.02), (7200.0, -0.05)]},
        spatial_field_paths={"O2_final": "snapshots/O2_t_final.nc"},
        distributions={"agent_masses": [0.1, 0.12, 0.15, 0.09]},
    )
    payload = b.model_dump_json()
    restored = OutputBundle.model_validate_json(payload)
    # Pydantic decodes (float, float) tuples back — the typing is tuple,
    # so values should be comparable.
    assert len(restored.scalar_time_series["thickness"]) == 3
    t0, v0 = restored.scalar_time_series["thickness"][0]
    assert t0 == 0.0 and v0 == 0.0
    assert restored.scalar_time_series["thickness"][-1][1] == pytest.approx(9.8)
    assert restored.distributions["agent_masses"][2] == pytest.approx(0.15)


def test_output_bundle_tolerates_missing_channels() -> None:
    """Partial outputs from a failed run should not raise."""
    b = OutputBundle(run_id="r", scalar_time_series={"only_this": [(0.0, 1.0)]})
    assert "only_this" in b.scalar_time_series
    assert b.flux_time_series == {}


# ---------------------------------------------------------------------------
# ResourceUsage + Artifact
# ---------------------------------------------------------------------------


def test_resource_usage_minimal() -> None:
    ru = ResourceUsage(wall_time_s=12.5)
    assert ru.peak_memory_gb is None


def test_artifact_round_trip() -> None:
    a = Artifact(
        relative_path="outputs/t_final.nc",
        kind="output",
        content_type="application/x-netcdf",
        size_bytes=1_048_576,
    )
    restored = Artifact.model_validate_json(a.model_dump_json())
    assert restored == a


# ---------------------------------------------------------------------------
# RunRecord + lifecycle
# ---------------------------------------------------------------------------


def _mk_run_record(tmp_path: Path, status: RunStatus = RunStatus.PENDING) -> RunRecord:
    return RunRecord(
        run_id="run-001",
        ir_id="nitrifying_biofilm_v0",
        framework="idynomics_2",
        framework_version="2.0.0",
        skill_file_version="0.1.0",
        assumption_ledger_hash="sha256:deadbeef",
        status=status,
        layout_root=tmp_path / "run-001",
    )


def test_run_record_default_pending(tmp_path: Path) -> None:
    r = _mk_run_record(tmp_path)
    assert r.status == RunStatus.PENDING
    assert r.started_at is None
    assert r.ended_at is None
    assert r.exit_code is None


def test_run_record_lifecycle_transitions(tmp_path: Path) -> None:
    r = _mk_run_record(tmp_path)
    now = datetime.now(timezone.utc)
    r.status = RunStatus.RUNNING
    r.started_at = now
    assert r.status == RunStatus.RUNNING

    r.status = RunStatus.SUCCEEDED
    r.ended_at = now
    r.exit_code = 0
    r.resource_usage = ResourceUsage(wall_time_s=120.0, peak_memory_gb=1.5)
    r.final_progress = ProgressReport(run_id=r.run_id, sim_time_s=86400.0)
    assert r.status == RunStatus.SUCCEEDED
    assert r.exit_code == 0


def test_run_record_failure_path(tmp_path: Path) -> None:
    r = _mk_run_record(tmp_path, RunStatus.FAILED)
    r.exit_code = 1
    r.failure_reason = "solver diverged at t=360 s"
    assert r.status == RunStatus.FAILED


def test_run_record_terminated_status(tmp_path: Path) -> None:
    r = _mk_run_record(tmp_path, RunStatus.TERMINATED)
    r.failure_reason = "wall_time_budget_s exceeded"
    assert r.status == RunStatus.TERMINATED


def test_run_record_artifacts_relative_paths(tmp_path: Path) -> None:
    r = _mk_run_record(tmp_path)
    r.artifacts.append(Artifact(relative_path="inputs/protocol.xml", kind="input"))
    r.artifacts.append(Artifact(relative_path="outputs/t_final.nc", kind="output"))
    # Relocate the bundle: relative paths remain valid.
    r.layout_root = tmp_path / "elsewhere"
    assert all(not Path(a.relative_path).is_absolute() for a in r.artifacts)


def test_run_record_round_trip(tmp_path: Path) -> None:
    r = _mk_run_record(tmp_path, RunStatus.SUCCEEDED)
    r.artifacts.append(Artifact(relative_path="inputs/a.xml", kind="input"))
    r.final_progress = ProgressReport(run_id=r.run_id, sim_time_s=3600.0)
    payload = r.model_dump_json()
    restored = RunRecord.model_validate_json(payload)
    assert restored.run_id == r.run_id
    assert restored.status == RunStatus.SUCCEEDED
    assert restored.artifacts[0].kind == "input"
    assert restored.final_progress is not None


def test_unknown_run_status_rejected() -> None:
    with pytest.raises(ValidationError):
        RunRecord(
            run_id="r",
            ir_id="ir",
            framework="f",
            framework_version="v",
            skill_file_version="sv",
            assumption_ledger_hash="h",
            layout_root=Path("/tmp/x"),
            status="exploded",  # type: ignore[arg-type]
        )


def test_ledger_hash_required(tmp_path: Path) -> None:
    """assumption_ledger_hash is required — locks the run to the approved
    assumptions. Omitting it is not allowed."""
    with pytest.raises(ValidationError):
        RunRecord(  # type: ignore[call-arg]
            run_id="r",
            ir_id="ir",
            framework="f",
            framework_version="v",
            skill_file_version="sv",
            layout_root=tmp_path,
        )
