"""Unit tests — Store save/load round-trip for meta-models, panels, suggestions."""

from __future__ import annotations

from pathlib import Path

from simtool.metamodel import (
    Evidence,
    EvidenceKind,
    SemVer,
    Suggestion,
    SuggestionLedger,
    SuggestionTargetKind,
)
from simtool.panels import Panel, UserConstraints
from simtool.persistence import Store

from tests._metamodel_fixtures import (
    make_nitrifying_biofilm_metamodel,
    make_small_panel_ir,
)


def test_metamodel_round_trip(tmp_path: Path):
    store = Store(tmp_path)
    mm = make_nitrifying_biofilm_metamodel()
    path = store.save_metamodel(mm)
    assert path.is_file()
    restored = store.load_metamodel(mm.id, mm.version)
    assert restored.id == mm.id
    assert len(restored.reconciled_parameters) == len(mm.reconciled_parameters)


def test_metamodel_list(tmp_path: Path):
    store = Store(tmp_path)
    mm = make_nitrifying_biofilm_metamodel()
    store.save_metamodel(mm)
    # bump and save a second version
    mm.version = SemVer.parse("1.3.0")
    store.save_metamodel(mm)
    listed = store.list_metamodels()
    assert (mm.id, "1.2.0") in listed
    assert (mm.id, "1.3.0") in listed


def test_latest_metamodel_version(tmp_path: Path):
    store = Store(tmp_path)
    mm = make_nitrifying_biofilm_metamodel()
    store.save_metamodel(mm)
    mm.version = SemVer.parse("2.0.0")
    store.save_metamodel(mm)
    latest = store.latest_metamodel_version(mm.id)
    assert latest == SemVer.parse("2.0.0")


def test_latest_metamodel_version_none_when_missing(tmp_path: Path):
    store = Store(tmp_path)
    assert store.latest_metamodel_version("nonexistent") is None


def test_panel_round_trip(tmp_path: Path):
    store = Store(tmp_path)
    panel = Panel(
        id="p1", title="t", user_id="alice",
        meta_model_id="mm", meta_model_version_pin="1.0.0",
        derived_ir=make_small_panel_ir(),
        constraints=UserConstraints(time_horizon_s=3600.0),
    )
    store.save_panel(panel)
    restored = store.load_panel("p1")
    assert restored.id == "p1"
    assert restored.title == "t"


def test_panel_list(tmp_path: Path):
    store = Store(tmp_path)
    for pid in ("a", "b", "c"):
        panel = Panel(
            id=pid, title=pid, user_id="u",
            meta_model_id="mm", meta_model_version_pin="1.0.0",
            derived_ir=make_small_panel_ir(),
            constraints=UserConstraints(time_horizon_s=100.0),
        )
        store.save_panel(panel)
    assert store.list_panels() == ["a", "b", "c"]


def test_suggestion_ledger_round_trip(tmp_path: Path):
    store = Store(tmp_path)
    s = Suggestion(
        id="s1", meta_model_id="mm", meta_model_version_seen="1.0.0",
        target_kind=SuggestionTargetKind.PARAMETER, target_id="mu_max",
        submitter_id="alice",
        summary="lower it", proposed_change="0.9",
        evidence=[Evidence(kind=EvidenceKind.DOI, reference="10.1/x")],
    )
    store.append_suggestion(s)
    restored = store.load_suggestion_ledger("mm")
    assert len(restored.suggestions) == 1
    assert restored.suggestions[0].id == "s1"


def test_load_suggestion_ledger_missing_returns_empty(tmp_path: Path):
    store = Store(tmp_path)
    led = store.load_suggestion_ledger("missing_mm")
    assert led.meta_model_id == "missing_mm"
    assert led.suggestions == []


def test_runs_root_created(tmp_path: Path):
    store = Store(tmp_path)
    root = store.runs_root()
    assert root.is_dir()
    # Twice — idempotent.
    root2 = store.runs_root()
    assert root2 == root
