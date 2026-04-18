"""Ara session — high-level entrypoints for the chat-driven flow.

An Ara deployment looks like:

    from simtool.ara import AraSession
    s = AraSession(root="~/simtool")
    s.open_metamodel("nitrifying_biofilm")
    panel = s.create_panel("alice", title="my biofilm", required_phenomena=["growth"])
    reply = s.recommend()            # text
    reply, attachments = s.run()     # text + PNGs

Each method returns user-visible strings (markdown) and, where relevant,
image byte streams. Hides plumbing so the agent body stays readable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from simtool.ara.render_text import (
    render_adjustment_proposal,
    render_fit_result,
    render_metamodel_summary,
    render_output_bundle,
    render_panel_summary,
    render_progress_stream,
    render_recommendation,
    render_scope_status,
)
from simtool.connector.ir import ScientificModel
from simtool.frameworks.idynomics_2 import (
    IDynoMiCS2NotAvailable,
    IDynoMiCS2Plugin,
    resolve_jar_path,
)
from simtool.metamodel import MetaModel, evaluate_ir_against_scope
from simtool.panels import (
    AdjustmentRequest,
    ExperimentalDataset,
    FitCalibrationResult,
    MeasurementCapability,
    Panel,
    PublicationState,
    UserConstraints,
    adjust_model,
    fit_data,
    recommend_model,
)
from simtool.persistence import Store
from simtool.runner import RunExecutionSummary, run_panel


class AraSession:
    """Stateful session over a persistent Store. One per user chat."""

    def __init__(
        self,
        root: Path | str = "~/simtool",
        *,
        plugin=None,
    ):
        self.store = Store(Path(root))
        self.plugin = plugin or IDynoMiCS2Plugin()
        self._active_metamodel: Optional[MetaModel] = None
        self._active_panel: Optional[Panel] = None

    # --- meta-model --------------------------------------------------------

    def open_metamodel(
        self, meta_model_id: str, version: Optional[str] = None
    ) -> str:
        if version is None:
            latest = self.store.latest_metamodel_version(meta_model_id)
            if latest is None:
                raise FileNotFoundError(
                    f"no meta-model '{meta_model_id}' on disk under "
                    f"{self.store.root}"
                )
            version = str(latest)
        self._active_metamodel = self.store.load_metamodel(meta_model_id, version)
        return render_metamodel_summary(self._active_metamodel)

    def save_metamodel(self, mm: MetaModel) -> str:
        path = self.store.save_metamodel(mm)
        return f"saved meta-model to {path}"

    # --- panel -------------------------------------------------------------

    def open_panel(self, panel_id: str) -> str:
        self._active_panel = self.store.load_panel(panel_id)
        return render_panel_summary(self._active_panel)

    def create_panel(
        self,
        user_id: str,
        title: str,
        *,
        derived_ir: ScientificModel,
        required_phenomena: Optional[list[str]] = None,
        excluded_phenomena: Optional[list[str]] = None,
        time_horizon_s: float = 7200.0,
        observable_ids: Optional[list[str]] = None,
    ) -> str:
        self._require_meta_model()
        assert self._active_metamodel is not None
        constraints = UserConstraints(
            predictive_priorities=observable_ids or [],
            measurement_capabilities=[
                MeasurementCapability(observable_id=o)
                for o in (observable_ids or [])
            ],
            time_horizon_s=time_horizon_s,
            required_phenomena=required_phenomena or [],
            excluded_phenomena=excluded_phenomena or [],
        )
        panel = Panel(
            id=f"panel-{user_id}-{abs(hash(title)) & 0xFFFFFFFF:x}",
            title=title,
            user_id=user_id,
            meta_model_id=self._active_metamodel.id,
            meta_model_version_pin=str(self._active_metamodel.version),
            derived_ir=derived_ir,
            constraints=constraints,
        )
        self.store.save_panel(panel)
        self._active_panel = panel
        return render_panel_summary(panel)

    # --- workflows ---------------------------------------------------------

    def recommend(self) -> str:
        self._require_both()
        assert self._active_metamodel is not None and self._active_panel is not None
        rec = recommend_model(self._active_metamodel, self._active_panel.constraints)
        return render_recommendation(rec)

    def adjust(self, kind: str, target_id: str = "", **spec) -> str:
        self._require_both()
        assert self._active_metamodel is not None and self._active_panel is not None
        proposal = adjust_model(
            self._active_panel, self._active_metamodel,
            AdjustmentRequest(kind=kind, target_id=target_id, spec=spec),
        )
        return render_adjustment_proposal(proposal)

    def fit(
        self,
        datasets: list[ExperimentalDataset],
        fit_results: list[FitCalibrationResult],
    ) -> str:
        self._require_both()
        assert self._active_metamodel is not None and self._active_panel is not None
        result = fit_data(
            self._active_panel, self._active_metamodel,
            datasets, fit_results,
        )
        self.store.save_panel(self._active_panel)
        return render_fit_result(result)

    # --- scope / readiness -------------------------------------------------

    def scope_status(self) -> str:
        self._require_both()
        assert self._active_metamodel is not None and self._active_panel is not None
        report = evaluate_ir_against_scope(
            self._active_panel.derived_ir, self._active_metamodel,
        )
        return render_scope_status(report)

    # --- run ---------------------------------------------------------------

    def run(
        self,
        *,
        auto_approve_assumptions: bool = False,
        extra_jar_hint: Optional[str] = None,
    ) -> tuple[str, RunExecutionSummary]:
        """Execute the active panel through the plugin. Returns a
        markdown reply string and the structured summary (for the agent
        to attach charts, etc.)."""
        self._require_both()
        assert self._active_panel is not None
        if self._active_panel.publication_state == PublicationState.FROZEN:
            return (
                f"panel '{self._active_panel.id}' is FROZEN — fork before running.",
                None,  # type: ignore[return-value]
            )
        # Fail early with a helpful message if the iDynoMiCS jar isn't found.
        if isinstance(self.plugin, IDynoMiCS2Plugin):
            jar = resolve_jar_path()
            if jar is None:
                raise IDynoMiCS2NotAvailable(
                    "iDynoMiCS 2 jar not found. Set IDYNOMICS_2_JAR or place "
                    "the jar at ./vendor/iDynoMiCS-2.0.jar."
                    + (f"\nHint from caller: {extra_jar_hint}" if extra_jar_hint else "")
                )

        summary = run_panel(
            self._active_panel, self._active_metamodel,
            plugin=self.plugin,
            run_root=self.store.runs_root(),
            auto_approve_assumptions=auto_approve_assumptions,
        )
        # Persist the updated panel (run_history was appended).
        self.store.save_panel(self._active_panel)

        text_parts: list[str] = [
            f"# Run {summary.run_record.run_id} ({summary.run_record.status.value})",
        ]
        if summary.auto_approved_assumption_ids:
            text_parts.append(
                "auto-approved assumptions: "
                + ", ".join(summary.auto_approved_assumption_ids)
            )
        if summary.run_record.failure_reason:
            text_parts.append(f"failure: {summary.run_record.failure_reason}")
        if summary.progress_reports:
            text_parts.append("\n## Progress\n" +
                              render_progress_stream(summary.progress_reports[-10:]))
        text_parts.append("\n" + render_output_bundle(summary.output_bundle))
        if summary.protocol_doc_path:
            text_parts.append(f"\nODD protocol written: {summary.protocol_doc_path}")
        return "\n".join(text_parts), summary

    # --- Panel mutation helpers -------------------------------------------

    def freeze_active_panel(self) -> str:
        self._require_panel()
        assert self._active_panel is not None
        self._active_panel.freeze()
        self.store.save_panel(self._active_panel)
        return render_panel_summary(self._active_panel)

    def unfreeze_active_panel(self) -> str:
        self._require_panel()
        assert self._active_panel is not None
        self._active_panel.unfreeze()
        self.store.save_panel(self._active_panel)
        return render_panel_summary(self._active_panel)

    # --- Accessors ---------------------------------------------------------

    @property
    def active_metamodel(self) -> Optional[MetaModel]:
        return self._active_metamodel

    @property
    def active_panel(self) -> Optional[Panel]:
        return self._active_panel

    # --- Guards ------------------------------------------------------------

    def _require_meta_model(self) -> None:
        if self._active_metamodel is None:
            raise RuntimeError(
                "no active meta-model; call session.open_metamodel(id) first"
            )

    def _require_panel(self) -> None:
        if self._active_panel is None:
            raise RuntimeError(
                "no active panel; call session.open_panel(id) or create_panel(...)"
            )

    def _require_both(self) -> None:
        self._require_meta_model()
        self._require_panel()
