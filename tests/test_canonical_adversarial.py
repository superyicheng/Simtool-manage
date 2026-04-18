"""Canonical ADVERSARIAL example — graceful failure modes.

User stories where the system MUST produce structured "I don't know" or
"I can't do that" rather than silently proceeding with garbage:

  1. Two papers disagree by 10x on the same parameter. The user must be
     told, not presented with a confidently-wrong number.
  2. A paper's extraction failed dimensional checks. The record must be
     dropped AND the user must be told which paper was dropped.
  3. Every paper for a parameter failed QC. The binding must be None and
     the user must receive a structured "no usable evidence".
  4. User requests an unsupported formalism (e.g. molecular dynamics)
     from a plugin that only does agent-based. The plugin returns a
     ValidationReport with ok=False and specific errors — it does not
     crash and does not silently produce wrong output.
  5. User tries to execute a run without approving the assumption
     ledger. The harness must refuse, naming the specific pending
     assumptions.
  6. A run crashed partway through. ``parse_outputs`` must tolerate
     missing output files and return empty channels rather than raising.
  7. IR referential integrity violations must be caught at IR
     construction with a message naming the offending id — not silently
     passed to the plugin to crash on.

Every failure here is tested for STRUCTURE: the user gets a typed
object carrying the failure reason, with provenance where applicable.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from simtool.connector.ir import (
    AgentPopulation,
    BoundaryCondition,
    ComputeBudget,
    CustomProcess,
    Distribution,
    FirstOrderDecayProcess,
    MonodGrowthProcess,
    ParameterBinding,
    ScientificModel,
    Solute,
    SpatialDomain,
    Surface,
)
from simtool.connector.plugin import DocSources
from simtool.connector.runs import OutputBundle, RunLayout
from simtool.schema.parameter_record import GradeRating

from tests._meta_model_fixtures import make_record, reconcile_records
from tests.test_connector_plugin import MockPlugin


# ---------------------------------------------------------------------------
# Story 1 — strong disagreement must surface, not collapse
# ---------------------------------------------------------------------------


def test_ten_x_disagreement_surfaces_as_distribution_with_conflict_flag() -> None:
    """Two papers report K_s(O2) values 10x apart. The pipeline must NOT
    silently pick one. It must emit an empirical distribution AND raise a
    conflict flag the user will see."""
    records = [
        make_record(
            parameter_id="K_s", value=0.2, unit="mg/L", canonical_value=0.2,
            doi="10.1000/low", species="AOB", substrate="oxygen",
        ),
        make_record(
            parameter_id="K_s", value=2.0, unit="mg/L", canonical_value=2.0,
            doi="10.1000/high", species="AOB", substrate="oxygen",
        ),
    ]
    result = reconcile_records(records)
    assert result.binding is not None
    assert result.binding.point_estimate is None, (
        "10x disagreement must NOT collapse to a point estimate"
    )
    assert result.binding.distribution is not None
    assert result.binding.distribution.shape == "empirical"
    assert set(result.binding.distribution.samples or []) == {0.2, 2.0}
    assert any("disagree" in f for f in result.conflict_flags), (
        "disagreement must surface as a structured conflict flag"
    )


# ---------------------------------------------------------------------------
# Story 2 — partial QC failure drops records but warns
# ---------------------------------------------------------------------------


def test_failed_dimensional_check_record_dropped_with_warning() -> None:
    good = make_record(
        parameter_id="mu_max", value=1.0, unit="1/day", canonical_value=1.0,
        doi="10.1000/good", species="AOB",
    )
    bad = make_record(
        parameter_id="mu_max", value=3.0, unit="1/day", canonical_value=None,
        doi="10.1000/bad", species="AOB",
        dimensional_check_passed=False,
    )
    result = reconcile_records([good, bad])
    assert result.binding is not None
    assert result.binding.point_estimate == pytest.approx(1.0), (
        "bad record must not influence the binding"
    )
    assert result.binding.provenance_dois == ["10.1000/good"], (
        "bad record's DOI must not appear as provenance"
    )
    assert result.warnings, "dropping a record must produce a user-visible warning"
    assert any("dropped" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Story 3 — everything failed QC → structured "I don't know"
# ---------------------------------------------------------------------------


def test_all_records_fail_qc_returns_structured_none() -> None:
    records = [
        make_record(
            parameter_id="mu_max", value=1.0, unit="wrong_unit",
            canonical_value=None, doi=f"10.1000/{i}", species="AOB",
            dimensional_check_passed=False,
        )
        for i in range(3)
    ]
    result = reconcile_records(records)
    assert result.binding is None, (
        "with zero usable records the binding must be None — a structured "
        "'I don't know', not a guess"
    )
    assert result.warnings, "user must see why the reconciliation failed"


def test_empty_input_returns_structured_none() -> None:
    result = reconcile_records([])
    assert result.binding is None
    assert result.warnings == ["no records supplied"]


def test_mismatched_parameter_ids_return_structured_none() -> None:
    mixed = [
        make_record(parameter_id="mu_max", value=1.0, unit="1/day",
                    doi="10.1000/a", species="AOB"),
        make_record(parameter_id="K_s", value=1.0, unit="mg/L",
                    doi="10.1000/b", species="AOB", substrate="ammonium"),
    ]
    result = reconcile_records(mixed)
    assert result.binding is None
    assert any("parameter_ids" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Story 4 — unsupported formalism is rejected with structured issues
# ---------------------------------------------------------------------------


def test_unsupported_formalism_returns_structured_validation_errors() -> None:
    ir = _minimal_valid_ir(formalism="molecular_dynamics")
    plugin = MockPlugin()
    skill = plugin.parse_docs(DocSources(sources=[]))
    report = plugin.validate_ir(ir, skill)
    assert not report.ok, "plugin must reject unsupported formalism"
    assert report.errors(), "failure must be reported as typed errors"
    assert any("agent_based" in e.message for e in report.errors()), (
        "error message must tell the user what IS supported"
    )


def test_custom_process_returns_structured_validation_error() -> None:
    """A CustomProcess is the IR's escape hatch; plugins MAY reject it,
    and the rejection must be typed, with a suggestion."""
    custom = CustomProcess(
        id="mystery", description="???", actors=["AOB"],
    )
    ir = _minimal_valid_ir(extra_processes=[custom])
    plugin = MockPlugin()
    skill = plugin.parse_docs(DocSources(sources=[]))
    report = plugin.validate_ir(ir, skill)
    assert not report.ok
    errs = report.errors()
    assert errs
    assert any("custom" in e.message.lower() for e in errs)
    assert any(e.suggestion for e in errs), (
        "rejected CustomProcess error must carry a suggestion"
    )


# ---------------------------------------------------------------------------
# Story 5 — execute refuses to launch on a pending ledger
# ---------------------------------------------------------------------------


def test_execute_without_approval_raises_named_runtime_error(
    tmp_path: Path,
) -> None:
    ir = _minimal_valid_ir()
    plugin = MockPlugin()
    skill = plugin.parse_docs(DocSources(sources=[]))
    artifact = plugin.lower(ir, skill)
    layout = RunLayout.under(tmp_path / "run")
    with pytest.raises(RuntimeError) as exc_info:
        plugin.execute(artifact, layout)
    msg = str(exc_info.value)
    assert "not fully approved" in msg
    # The specific pending assumption ids must be in the error message so
    # the user knows what to act on.
    for a in artifact.assumptions.assumptions:
        assert a.id in msg, (
            f"RuntimeError must name pending assumption '{a.id}' to guide "
            f"the user; got: {msg}"
        )


def test_rejected_ledger_also_blocks_execute(tmp_path: Path) -> None:
    ir = _minimal_valid_ir()
    plugin = MockPlugin()
    skill = plugin.parse_docs(DocSources(sources=[]))
    artifact = plugin.lower(ir, skill)
    # Approve one, reject the other.
    a_ids = [a.id for a in artifact.assumptions.assumptions]
    artifact.assumptions.approve(a_ids[0])
    artifact.assumptions.reject(a_ids[1], user_note="need a different default")
    layout = RunLayout.under(tmp_path / "run")
    with pytest.raises(RuntimeError, match="not fully approved"):
        plugin.execute(artifact, layout)


# ---------------------------------------------------------------------------
# Story 6 — crash-mid-run tolerance
# ---------------------------------------------------------------------------


def test_parse_outputs_on_empty_run_returns_empty_bundle(tmp_path: Path) -> None:
    plugin = MockPlugin()
    layout = RunLayout.under(tmp_path / "crashed_run")
    layout.ensure()
    # No thickness.csv was ever written (run crashed before producing it).
    bundle = plugin.parse_outputs(layout)
    assert isinstance(bundle, OutputBundle)
    assert bundle.scalar_time_series["thickness"] == [], (
        "missing outputs must return an empty series, not raise"
    )


def test_parse_outputs_on_truncated_file_does_not_silently_drop_present_rows(
    tmp_path: Path,
) -> None:
    plugin = MockPlugin()
    layout = RunLayout.under(tmp_path / "truncated_run")
    layout.ensure()
    # Simulate a run that wrote only 1 of 3 expected rows before crashing.
    (layout.outputs / "thickness.csv").write_text(
        "t_s,thickness_um\n0,0.0\n"
    )
    bundle = plugin.parse_outputs(layout)
    assert bundle.scalar_time_series["thickness"] == [(0.0, 0.0)], (
        "present rows must still be recovered from a truncated output"
    )


# ---------------------------------------------------------------------------
# Story 7 — IR referential integrity errors are caught at construction
# ---------------------------------------------------------------------------


def test_ir_with_ghost_entity_in_bc_fails_at_construction() -> None:
    """A typo in an entity name must not silently flow to the plugin. The
    IR itself must catch it, with the offending id in the message."""
    with pytest.raises(ValidationError) as exc_info:
        _minimal_valid_ir(ghost_bc=True)
    assert "GHOST_SOLUTE" in str(exc_info.value)


def test_distribution_with_wrong_params_fails_at_construction() -> None:
    """A reconciler emitting a malformed distribution must be caught at
    the ParameterBinding boundary, not at lowering time."""
    with pytest.raises(ValidationError, match="missing"):
        Distribution(shape="normal", params={"mean": 1.0})


def test_parameter_binding_with_no_value_fails_at_construction() -> None:
    """An extractor that forgot to set either point estimate or
    distribution must be caught loudly."""
    with pytest.raises(ValidationError, match="exactly one"):
        ParameterBinding(parameter_id="mu_max", canonical_unit="1/day")


# ---------------------------------------------------------------------------
# Story 8 — high-level adversarial: full pipeline with a conflicting corpus
# ---------------------------------------------------------------------------


def test_adversarial_full_pipeline_ends_with_structured_distribution_and_assumptions(
    tmp_path: Path,
) -> None:
    """End-to-end adversarial: the user gives us a corpus with a 10x
    disagreement and one failed-QC record. We expect:
      - reconciled binding is a distribution, not a point estimate
      - conflict flags are surfaced
      - QC failure is warned
      - IR validates, lowers, runs, and parses
      - assumption ledger captures the usual numerical/boundary choices
    """
    corpus = [
        make_record(
            parameter_id="mu_max", value=1.0, unit="1/day", canonical_value=1.0,
            doi="10.1000/a", species="AOB", grade=GradeRating.HIGH,
        ),
        make_record(
            parameter_id="mu_max", value=10.0, unit="1/day", canonical_value=10.0,
            doi="10.1000/b", species="AOB", grade=GradeRating.MODERATE,
        ),
        make_record(
            parameter_id="mu_max", value=2.5, unit="wrong_unit", canonical_value=None,
            doi="10.1000/c", species="AOB", dimensional_check_passed=False,
        ),
    ]
    result = reconcile_records(corpus)
    assert result.binding is not None
    assert result.binding.distribution is not None
    assert result.conflict_flags
    assert result.warnings

    ir = _minimal_valid_ir(mu_max_binding=result.binding)

    plugin = MockPlugin()
    skill = plugin.parse_docs(DocSources(sources=[]))
    report = plugin.validate_ir(ir, skill)
    assert report.ok

    artifact = plugin.lower(ir, skill)
    # Distribution flows through to the lowered artifact.
    ir_json = next(c for n, c in artifact.files if n == "ir.json")
    assert b'"distribution"' in ir_json

    # Approve + run to completion.
    for a in artifact.assumptions.assumptions:
        artifact.assumptions.approve(a.id, user_note="adversarial path")
    layout = RunLayout.under(tmp_path / "adversarial_run")
    handle = plugin.execute(artifact, layout)
    list(plugin.monitor(handle))
    bundle = plugin.parse_outputs(layout)
    assert "thickness" in bundle.scalar_time_series


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fixed(pid: str, unit: str, v: float) -> ParameterBinding:
    return ParameterBinding(
        parameter_id=pid, canonical_unit=unit, point_estimate=v,
    )


def _minimal_valid_ir(
    *,
    formalism: str = "agent_based",
    extra_processes: list | None = None,
    ghost_bc: bool = False,
    mu_max_binding: ParameterBinding | None = None,
) -> ScientificModel:
    mu = mu_max_binding or _fixed("mu_max", "1/day", 1.0)
    aob = AgentPopulation(id="AOB", name="AOB")
    nh4 = Solute(id="NH4", name="ammonium")
    o2 = Solute(id="O2", name="oxygen")
    dom = SpatialDomain(id="dom", dimensionality=1, extent_um=(64.0,))
    top = Surface(id="top", name="top", axis=0, position="high")
    bot = Surface(id="bottom", name="bottom", axis=0, position="low")
    growth = MonodGrowthProcess(
        id="g", growing_entity="AOB", consumed_solutes=["NH4", "O2"],
        parameters={
            "mu_max": mu,
            "K_s_NH4": _fixed("K_s", "mg/L", 1.0),
            "K_s_O2": _fixed("K_s", "mg/L", 0.5),
            "Y_XS_NH4": _fixed("Y_XS", "g_biomass/g_substrate", 0.12),
        },
    )
    decay = FirstOrderDecayProcess(
        id="d", decaying_entity="AOB",
        parameters={"b": _fixed("b", "1/day", 0.1)},
    )
    procs: list = [growth, decay]
    if extra_processes:
        procs.extend(extra_processes)
    bcs = [
        BoundaryCondition(
            id="NH4_top", target_entity="NH4", surface="top", kind="dirichlet",
            value=_fixed("S_bulk_initial", "mg/L", 30.0),
        ),
    ]
    if ghost_bc:
        bcs.append(
            BoundaryCondition(
                id="ghost", target_entity="GHOST_SOLUTE", surface="top",
                kind="no_flux",
            )
        )
    return ScientificModel(
        id="minimal", title="t", domain="d", formalism=formalism,  # type: ignore[arg-type]
        entities=[aob, nh4, o2, dom, top, bot],
        processes=procs,
        boundary_conditions=bcs,
        initial_conditions=[],
        observables=[],
        compute=ComputeBudget(time_horizon_s=1000.0),
    )
