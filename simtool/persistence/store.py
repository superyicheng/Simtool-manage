"""JSON-backed Store for Simtool artifacts.

Minimal, deterministic, single-user. Good enough to survive Ara VM
restarts; replaceable with a Supabase/Postgres backend later without
changing callers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from simtool.metamodel import (
    MetaModel,
    SemVer,
    Suggestion,
    SuggestionLedger,
)
from simtool.panels import Panel


class Store:
    """Filesystem-backed persistence for Simtool artifacts."""

    def __init__(self, root: Path):
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)

    # --- MetaModels --------------------------------------------------------

    def metamodel_path(self, meta_model_id: str, version: SemVer | str) -> Path:
        v = str(version)
        return self.root / "metamodels" / meta_model_id / v / "metamodel.json"

    def save_metamodel(self, mm: MetaModel) -> Path:
        path = self.metamodel_path(mm.id, mm.version)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(mm.model_dump_json(indent=2))
        return path

    def load_metamodel(
        self, meta_model_id: str, version: SemVer | str
    ) -> MetaModel:
        path = self.metamodel_path(meta_model_id, version)
        return MetaModel.model_validate_json(path.read_text())

    def list_metamodels(self) -> list[tuple[str, str]]:
        base = self.root / "metamodels"
        if not base.is_dir():
            return []
        out: list[tuple[str, str]] = []
        for id_dir in sorted(base.iterdir()):
            if not id_dir.is_dir():
                continue
            for ver_dir in sorted(id_dir.iterdir()):
                if (ver_dir / "metamodel.json").is_file():
                    out.append((id_dir.name, ver_dir.name))
        return out

    def latest_metamodel_version(
        self, meta_model_id: str
    ) -> Optional[SemVer]:
        base = self.root / "metamodels" / meta_model_id
        if not base.is_dir():
            return None
        versions = []
        for ver_dir in base.iterdir():
            if (ver_dir / "metamodel.json").is_file():
                try:
                    versions.append(SemVer.parse(ver_dir.name))
                except ValueError:
                    continue
        if not versions:
            return None
        return max(versions)

    # --- Panels ------------------------------------------------------------

    def panel_path(self, panel_id: str) -> Path:
        return self.root / "panels" / panel_id / "panel.json"

    def save_panel(self, panel: Panel) -> Path:
        path = self.panel_path(panel.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(panel.model_dump_json(indent=2))
        return path

    def load_panel(self, panel_id: str) -> Panel:
        return Panel.model_validate_json(self.panel_path(panel_id).read_text())

    def list_panels(self) -> list[str]:
        base = self.root / "panels"
        if not base.is_dir():
            return []
        return sorted(
            d.name for d in base.iterdir()
            if d.is_dir() and (d / "panel.json").is_file()
        )

    # --- Suggestions -------------------------------------------------------

    def suggestion_ledger_path(self, meta_model_id: str) -> Path:
        return self.root / "suggestions" / meta_model_id / "ledger.json"

    def save_suggestion_ledger(self, ledger: SuggestionLedger) -> Path:
        path = self.suggestion_ledger_path(ledger.meta_model_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(ledger.model_dump_json(indent=2))
        return path

    def load_suggestion_ledger(self, meta_model_id: str) -> SuggestionLedger:
        path = self.suggestion_ledger_path(meta_model_id)
        if not path.is_file():
            return SuggestionLedger(meta_model_id=meta_model_id)
        return SuggestionLedger.model_validate_json(path.read_text())

    def append_suggestion(self, suggestion: Suggestion) -> None:
        ledger = self.load_suggestion_ledger(suggestion.meta_model_id)
        ledger.submit(suggestion)
        self.save_suggestion_ledger(ledger)

    # --- Runs --------------------------------------------------------------

    def runs_root(self) -> Path:
        p = self.root / "runs"
        p.mkdir(parents=True, exist_ok=True)
        return p
