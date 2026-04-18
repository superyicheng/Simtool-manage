"""Unit tests — runner orchestrator with the MockPlugin.

The orchestrator's job is to sequence the plugin contract correctly.
We drive it with MockPlugin (from tests/test_connector_plugin.py) so the
tests run without any real framework dependencies.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from simtool.connector.ir import (
    AgentPopulation,
    ComputeBudget,
    MonodGrowthProcess,
    ParameterBinding,
    ScientificModel,
    Solute,
    SpatialDomain,
    Surface,
)
from simtool.connector.runs import RunStatus
from simtool.panels import MeasurementCapability, Panel, UserConstraints
from simtool.runner import run_panel, run_scientific_model

from tests.test_connector_plugin import MockPlugin


def _b(pid, unit, v):
    return ParameterBinding(parameter_id=pid, canonical_unit=unit, point_estimate=v)


def _ir() -> ScientificModel:
    aob = AgentPopulation(id="AOB", name="AOB")
    s = Solute(id="S", name="S")
    o2 = Solute(id="O2", name="O2")
    dom = SpatialDomain(id="dom", dimensionality=1, extent_um=(32.0,))
    top = Surface(id="top", name="top", axis=0, position="high")
    bot = Surface(id="bot", name="bot", axis=0, position="low")
    return ScientificModel(
        id="ir", title="t", domain="d", formalism="agent_based",
        entities=[aob, s, o2, dom, top, bot],
        processes=[
            MonodGrowthProcess(
                id="g", growing_entity="AOB", consumed_solutes=["S", "O2"],
                parameters={
                    "mu_max": _b("mu_max", "1/day", 1.0),
                    "K_s_S": _b("K_s", "mg/L", 1.0),
                    "K_s_O2": _b("K_s", "mg/L", 0.5),
                    "Y_XS_S": _b("Y_XS", "g_biomass/g_substrate", 0.12),
                },
            ),
        ],
        compute=ComputeBudget(time_horizon_s=3600.0),
    )


def _panel(ir=None) -> Panel:
    ir = ir or _ir()
    return Panel(
        id="p", title="p", user_id="alice",
        meta_model_id="", meta_model_version_pin="0.0.0",
        derived_ir=ir,
        constraints=UserConstraints(time_horizon_s=3600.0),
    )


def test_run_panel_happy_path_with_auto_approve(tmp_path):
    summary = run_panel(
        _panel(), None, plugin=MockPlugin(), run_root=tmp_path,
        auto_approve_assumptions=True,
    )
    assert summary.validation_report.ok
    assert summary.auto_approved_assumption_ids  # MockPlugin adds 2
    assert summary.run_record.status == RunStatus.SUCCEEDED
    assert summary.progress_reports
    # RunRecord is persisted to disk.
    meta_path = Path(summary.run_record.layout_root) / "metadata" / "run.json"
    assert meta_path.is_file()
    # Panel's run history was appended.


def test_run_panel_fails_when_not_auto_approved(tmp_path):
    summary = run_panel(
        _panel(), None, plugin=MockPlugin(), run_root=tmp_path,
        auto_approve_assumptions=False,
    )
    assert summary.run_record.status == RunStatus.FAILED
    assert "not fully approved" in (summary.run_record.failure_reason or "")


def test_run_panel_fails_on_invalid_ir(tmp_path):
    # Use a formalism MockPlugin doesn't accept.
    ir = _ir()
    bad_ir = ir.model_copy(update={"formalism": "molecular_dynamics"})
    summary = run_panel(
        _panel(bad_ir), None, plugin=MockPlugin(), run_root=tmp_path,
        auto_approve_assumptions=True,
    )
    assert summary.run_record.status == RunStatus.FAILED
    assert "IR failed plugin validation" in (summary.run_record.failure_reason or "")
    assert not summary.validation_report.ok


def test_run_scientific_model_builds_adhoc_panel(tmp_path):
    summary = run_scientific_model(
        _ir(), plugin=MockPlugin(), run_root=tmp_path,
        auto_approve_assumptions=True,
    )
    assert summary.run_record.status == RunStatus.SUCCEEDED
    assert summary.run_record.ir_id == "ir"


def test_run_panel_attaches_to_run_history(tmp_path):
    panel = _panel()
    assert len(panel.run_history) == 0
    run_panel(
        panel, None, plugin=MockPlugin(), run_root=tmp_path,
        auto_approve_assumptions=True,
    )
    assert len(panel.run_history) == 1
    assert panel.run_history[0].status == "succeeded"
