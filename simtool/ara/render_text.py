"""Text renderers for Ara deployment.

Ara is message-only: no frontend. These renderers produce markdown-
compatible text blocks that the agent sends back as chat messages.

Every renderer is pure: ``object in -> str out``, no I/O, no mutation.
The agent composes them.

Design rules:
  - Lead with the headline the user needs first (version, status, verdict).
  - Surface uncertainty: distributions render with ranges, not point
    estimates.
  - Provenance links are always present — a meta-model claim without a
    DOI is a bug.
  - No raw JSON dumps. Every renderer is for human reading.
"""

from __future__ import annotations

from typing import Iterable, Optional

from simtool.connector.assumptions import AssumptionLedger, AssumptionStatus
from simtool.connector.ir import Distribution, ParameterBinding, ScientificModel
from simtool.connector.runs import OutputBundle, ProgressReport
from simtool.metamodel import (
    MetaModel,
    ParameterStatus,
    ReconciledParameter,
    ScopeStatusReport,
    Suggestion,
    SuggestionLedger,
    SuggestionStatus,
    is_stale,
    staleness_warning,
)
from simtool.panels import (
    AdjustmentProposal,
    FitDataResult,
    ModelRecommendation,
    Panel,
    SupportLevel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_binding(b: ParameterBinding) -> str:
    if b.point_estimate is not None:
        return f"{b.point_estimate:g} {b.canonical_unit}"
    d = b.distribution
    if d is None:
        return f"(no value) {b.canonical_unit}"
    return f"{_fmt_distribution(d)} {b.canonical_unit}"


def _fmt_distribution(d: Distribution) -> str:
    if d.shape == "empirical" and d.samples:
        lo, hi = min(d.samples), max(d.samples)
        return f"empirical [{lo:g}..{hi:g}] from {len(d.samples)} samples"
    if d.shape == "uniform":
        return f"uniform[{d.params['low']:g}, {d.params['high']:g}]"
    if d.shape == "normal":
        return f"normal(mean={d.params['mean']:g}, stddev={d.params['stddev']:g})"
    if d.shape == "lognormal":
        return f"lognormal(mu={d.params['mu']:g}, sigma={d.params['sigma']:g})"
    if d.shape == "triangular":
        return (
            f"triangular[{d.params['low']:g}, "
            f"mode={d.params['mode']:g}, {d.params['high']:g}]"
        )
    return d.shape


def _fmt_ctx(ctx: dict[str, str]) -> str:
    if not ctx:
        return "(no context)"
    return ", ".join(f"{k}={v}" for k, v in sorted(ctx.items()))


# ---------------------------------------------------------------------------
# Meta-model
# ---------------------------------------------------------------------------


def render_metamodel_summary(mm: MetaModel) -> str:
    """Top-level overview the user sees first."""
    lines = [
        f"# Meta-model: {mm.title}",
        f"- **id**: {mm.id}",
        f"- **version**: {mm.version}",
        f"- **domain**: {mm.scientific_domain}",
    ]
    if mm.zenodo_doi:
        lines.append(f"- **citable**: {mm.zenodo_doi}")
    if mm.maintainers:
        lines.append(f"- **maintainers**: {', '.join(mm.maintainers)}")
    lines.append(f"- **reconciled parameters**: {len(mm.reconciled_parameters)}")
    lines.append(f"- **submodels**: {len(mm.submodels)}")
    lines.append(f"- **approximation operators**: {len(mm.approximation_operators)}")

    # Staleness warning — surfaced early, not buried.
    warn = staleness_warning(mm)
    if warn:
        lines.append("")
        lines.append(f"> **Staleness warning**: {warn}")
    elif mm.ingestion.last_ingestion_at is not None:
        lines.append(
            f"- **last ingested**: {mm.ingestion.last_ingestion_at.isoformat()}"
        )
    return "\n".join(lines)


def render_metamodel_parameter(rp: ReconciledParameter) -> str:
    lines = [
        f"### {rp.parameter_id}  ({_fmt_ctx(rp.context_keys)})",
        f"- value: {_fmt_binding(rp.binding)}",
        f"- supporting records: {len(rp.supporting_record_dois)} "
        f"({', '.join(rp.supporting_record_dois[:3])}"
        f"{'...' if len(rp.supporting_record_dois) > 3 else ''})",
        f"- quality: {rp.quality_rating.value}",
    ]
    if rp.conflict_flags:
        lines.append(f"- conflicts: {'; '.join(rp.conflict_flags)}")
    return "\n".join(lines)


def render_metamodel_parameters(mm: MetaModel, parameter_id: Optional[str] = None) -> str:
    """Full or filtered reconciled-parameter listing."""
    records = [
        rp for rp in mm.reconciled_parameters
        if parameter_id is None or rp.parameter_id == parameter_id
    ]
    if not records:
        return (
            f"no reconciled parameters matching parameter_id={parameter_id!r}"
        )
    header = (
        f"# Parameters in {mm.id} v{mm.version}"
        if parameter_id is None
        else f"# {parameter_id} in {mm.id} v{mm.version}"
    )
    return "\n\n".join([header] + [render_metamodel_parameter(r) for r in records])


def render_submodel_hierarchy(mm: MetaModel) -> str:
    """Indented listing of submodels ordered by complexity rank."""
    lines = [f"# Submodel hierarchy: {mm.id} v{mm.version}"]
    for s in sorted(mm.submodels, key=lambda x: x.complexity_rank):
        indent = "  " * s.complexity_rank
        lines.append(f"{indent}- **{s.id}** (rank {s.complexity_rank}): {s.name}")
        if s.covered_phenomena:
            lines.append(
                f"{indent}  covers: {', '.join(s.covered_phenomena)}"
            )
        if s.excluded_phenomena:
            lines.append(
                f"{indent}  excludes: {', '.join(s.excluded_phenomena)}"
            )
    if mm.approximation_operators:
        lines.append("")
        lines.append("## Approximation operators")
        for op in mm.approximation_operators:
            lines.append(
                f"- **{op.id}** ({op.kind.value}): "
                f"{op.from_submodel_id} → {op.to_submodel_id}"
            )
            if op.assumptions_introduced:
                lines.append(
                    f"  - assumes: {'; '.join(op.assumptions_introduced)}"
                )
    return "\n".join(lines)


def render_scope_status(report: ScopeStatusReport) -> str:
    """Ready-to-run verdict + per-parameter breakdown."""
    verdict = "READY" if report.is_ready_to_run() else "BLOCKED"
    lines = [
        f"# Scope status: {report.ir_id} vs {report.meta_model_id} "
        f"v{report.meta_model_version}",
        f"**Verdict**: {verdict}",
        "",
        f"- reconciled: {len(report.reconciled)}",
        f"- single-record: {len(report.single)}",
        f"- MISSING (blocking): {len(report.missing_blocking)}",
    ]
    if report.missing_blocking:
        lines.append("")
        lines.append("## Blocking — user must supply or accept speculation")
        for p in report.missing_blocking:
            lines.append(
                f"- **{p.parameter_id}** ({_fmt_ctx(p.context_keys)}) — {p.reason}"
            )
    if report.single:
        lines.append("")
        lines.append("## Single-record — lower confidence")
        for p in report.single:
            lines.append(
                f"- **{p.parameter_id}** ({_fmt_ctx(p.context_keys)}) — {p.reason}"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------


def render_panel_summary(panel: Panel) -> str:
    lines = [
        f"# Panel: {panel.title}",
        f"- **id**: {panel.id}",
        f"- **owner**: {panel.user_id}",
        f"- **state**: {panel.publication_state.value.upper()}",
        f"- **meta-model**: {panel.meta_model_id} @ {panel.meta_model_version_pin}",
        f"- **overrides**: {len(panel.parameter_overrides)}",
        f"- **runs**: {len(panel.run_history)}",
    ]
    if panel.frozen_at_meta_model_version:
        lines.append(
            f"- **frozen at**: {panel.frozen_at_meta_model_version} "
            f"({panel.frozen_at.isoformat() if panel.frozen_at else 'unknown'})"
        )
    if panel.collaborators:
        lines.append(f"- **collaborators**: {', '.join(panel.collaborators)}")
    if panel.tags:
        lines.append(f"- **tags**: {', '.join(panel.tags)}")
    return "\n".join(lines)


def render_panel_overrides(panel: Panel) -> str:
    if not panel.parameter_overrides:
        return f"panel '{panel.id}' has no overrides — all bindings inherit from the meta-model"
    lines = [f"# Overrides in panel '{panel.id}'"]
    for o in panel.parameter_overrides:
        lines.append(
            f"- **{o.parameter_id}** ({_fmt_ctx(o.context_keys)}): "
            f"{_fmt_binding(o.override_binding)}  "
            f"[source={o.source.value}]"
        )
        if o.original_binding is not None:
            lines.append(
                f"  - replaced meta-model value: "
                f"{_fmt_binding(o.original_binding)}"
            )
        lines.append(f"  - justification: {o.justification}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Workflows
# ---------------------------------------------------------------------------


def render_recommendation(rec: ModelRecommendation) -> str:
    if not rec.submodel_id:
        return (
            "# Recommendation: NO FIT\n"
            + "\n".join(f"- {c}" for c in rec.unmet_constraints)
        )
    lines = [
        f"# Recommendation: {rec.submodel_id}",
        f"- meta-model: {rec.meta_model_id} v{rec.meta_model_version}",
    ]
    if rec.derived_via_operator_id:
        lines.append(
            f"- **derived via**: {rec.derived_via_operator_id} "
            "(approximation operator)"
        )
    if rec.assumptions_introduced:
        lines.append("- assumptions introduced:")
        for a in rec.assumptions_introduced:
            lines.append(f"  - {a}")
    lines.append("")
    lines.append("## Reasoning")
    for step in rec.reasoning:
        if step.kind == "eligible":
            tag = "✓"
        elif step.kind == "rejected":
            tag = "✗"
        else:
            tag = "→"
        ident = step.submodel_id or step.operator_id or "?"
        lines.append(f"- {tag} {ident}: {step.note}")
    return "\n".join(lines)


def render_adjustment_proposal(proposal: AdjustmentProposal) -> str:
    support = proposal.support_level
    badge = {
        SupportLevel.SUPPORTED: "✅ SUPPORTED",
        SupportLevel.PARTIALLY_SUPPORTED: "⚠️ PARTIALLY SUPPORTED",
        SupportLevel.SPECULATIVE: "⚠️ SPECULATIVE — not grounded in meta-model",
    }[support]
    lines = [
        f"# Adjustment: {proposal.request.kind} ({proposal.request.target_id or 'scope'})",
        f"**{badge}**",
        "",
    ]
    if proposal.meta_model_has:
        lines.append(
            f"- meta-model has: {', '.join(proposal.meta_model_has)}"
        )
    if proposal.meta_model_missing:
        lines.append(
            f"- meta-model missing: {', '.join(proposal.meta_model_missing)}"
        )
    if proposal.depends_on_submodel_ids:
        lines.append(
            f"- candidate submodels: {', '.join(proposal.depends_on_submodel_ids)}"
        )
    if proposal.speculative_candidate_dois:
        lines.append(
            "- broader-literature candidates (speculative): "
            + ", ".join(proposal.speculative_candidate_dois)
        )
    if proposal.reasoning:
        lines.append(f"\n_reasoning_: {proposal.reasoning}")
    return "\n".join(lines)


def render_fit_result(result: FitDataResult) -> str:
    lines = [f"# Fit results for panel {result.panel_id}"]
    if result.within_consensus:
        lines.append(
            f"- ✅ within meta-model consensus: "
            f"{', '.join(result.within_consensus)}"
        )
    if result.outside_consensus:
        lines.append(
            f"- ⚠️ outside meta-model consensus: "
            f"{', '.join(result.outside_consensus)}"
        )
    if result.notes:
        lines.append("")
        lines.append("## Notes")
        for n in result.notes:
            lines.append(f"- {n}")
    if result.suggestions:
        lines.append("")
        lines.append("## Suggestions ready for community review")
        for s in result.suggestions:
            lines.append(
                f"- {s.id}: **{s.target_id}** — {s.summary}"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Suggestions
# ---------------------------------------------------------------------------


def render_suggestion(s: Suggestion) -> str:
    lines = [
        f"# Suggestion {s.id}",
        f"- target: {s.target_kind.value} `{s.target_id}` ({_fmt_ctx(s.target_context)})",
        f"- submitter: {s.submitter_id} (confidence {s.submitter_confidence:.2f})",
        f"- status: {s.status.value}",
        f"- meta-model version seen: {s.meta_model_version_seen}",
        "",
        f"**Summary**: {s.summary}",
        "",
        f"**Proposed change**: {s.proposed_change}",
    ]
    if s.evidence:
        lines.append("")
        lines.append("## Evidence")
        for e in s.evidence:
            lines.append(f"- [{e.kind.value}] {e.reference}  {e.note}")
    if s.resolver_explanation:
        lines.append("")
        lines.append(f"> {s.resolver_id} ({s.status.value}): {s.resolver_explanation}")
        if s.resulting_version:
            lines.append(f"> Resulting version: {s.resulting_version}")
    return "\n".join(lines)


def render_suggestion_ledger_summary(ledger: SuggestionLedger) -> str:
    by_status: dict[SuggestionStatus, int] = {}
    for s in ledger.suggestions:
        by_status[s.status] = by_status.get(s.status, 0) + 1
    lines = [f"# Suggestions for {ledger.meta_model_id}"]
    for status in SuggestionStatus:
        lines.append(f"- {status.value}: {by_status.get(status, 0)}")
    pending = ledger.pending()
    if pending:
        lines.append("")
        lines.append("## Pending")
        for s in pending[:10]:
            lines.append(
                f"- {s.id}: {s.target_kind.value}/{s.target_id} — "
                f"{s.summary} (from {s.submitter_id})"
            )
        if len(pending) > 10:
            lines.append(f"- ... +{len(pending) - 10} more")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Assumption ledger
# ---------------------------------------------------------------------------


def render_assumption_ledger(ledger: AssumptionLedger) -> str:
    lines = [
        f"# Assumption ledger for {ledger.ir_id} ({ledger.framework} v{ledger.framework_version})",
    ]
    ready = ledger.is_ready_to_run()
    lines.append(
        f"**Ready to run**: {'yes' if ready else 'NO'} "
        f"({len(ledger.assumptions)} total)"
    )
    if not ready:
        lines.append("")
        lines.append("## Blocking")
        for reason in ledger.blocking_reasons():
            lines.append(f"- {reason}")
    lines.append("")
    lines.append("## All assumptions")
    for a in ledger.assumptions:
        status_mark = {
            AssumptionStatus.PENDING: "⏳",
            AssumptionStatus.APPROVED: "✓",
            AssumptionStatus.REJECTED: "✗",
        }[a.status]
        lines.append(
            f"- {status_mark} **{a.id}** ({a.category.value}/{a.severity.value}): "
            f"{a.description}"
        )
        if a.alternatives:
            lines.append(
                f"  alternatives: {'; '.join(a.alternatives)}"
            )
        if a.user_note:
            lines.append(f"  user note: {a.user_note}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Runs — progress + outputs
# ---------------------------------------------------------------------------


def render_progress_line(pr: ProgressReport) -> str:
    """One-line status suitable for a chat stream."""
    parts = [f"[run {pr.run_id}]"]
    if pr.sim_time_s is not None and pr.sim_time_horizon_s:
        pct = 100.0 * pr.sim_time_s / pr.sim_time_horizon_s
        parts.append(f"t={pr.sim_time_s:.0f}s ({pct:.1f}%)")
    elif pr.sim_time_s is not None:
        parts.append(f"t={pr.sim_time_s:.0f}s")
    if pr.timestep_index is not None and pr.timestep_total is not None:
        parts.append(f"step {pr.timestep_index}/{pr.timestep_total}")
    for k, v in pr.observables.items():
        parts.append(f"{k}={v:g}")
    if pr.wall_time_estimated_remaining_s is not None:
        parts.append(f"~{pr.wall_time_estimated_remaining_s:.0f}s left")
    if pr.message:
        parts.append(f"— {pr.message}")
    return " ".join(parts)


def render_progress_stream(reports: Iterable[ProgressReport]) -> str:
    return "\n".join(render_progress_line(r) for r in reports)


def render_output_bundle(bundle: OutputBundle) -> str:
    lines = [f"# Outputs for run {bundle.run_id}"]
    if bundle.scalar_time_series:
        lines.append("")
        lines.append("## Scalar time series")
        for key, series in bundle.scalar_time_series.items():
            if series:
                final = series[-1]
                lines.append(
                    f"- **{key}**: {len(series)} points; final = "
                    f"{final[1]:g} at t={final[0]:g}s"
                )
            else:
                lines.append(f"- **{key}**: (no data)")
    if bundle.flux_time_series:
        lines.append("")
        lines.append("## Flux time series")
        for key, series in bundle.flux_time_series.items():
            if series:
                final = series[-1]
                lines.append(
                    f"- **{key}**: {len(series)} points; final = {final[1]:g}"
                )
    if bundle.distributions:
        lines.append("")
        lines.append("## Distributions")
        for key, samples in bundle.distributions.items():
            if samples:
                lines.append(
                    f"- **{key}**: {len(samples)} samples "
                    f"[{min(samples):g} .. {max(samples):g}]"
                )
    if bundle.spatial_field_paths:
        lines.append("")
        lines.append("## Spatial fields")
        for key, path in bundle.spatial_field_paths.items():
            lines.append(f"- **{key}**: {path}")
    if bundle.notes:
        lines.append("")
        lines.append(f"_notes_: {bundle.notes}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# IR (compact)
# ---------------------------------------------------------------------------


def render_ir_compact(ir: ScientificModel) -> str:
    lines = [
        f"# IR: {ir.title} ({ir.id})",
        f"- formalism: {ir.formalism}",
        f"- domain: {ir.domain}",
        f"- entities: {len(ir.entities)} — {', '.join(sorted(ir.entity_ids()))}",
        f"- processes: {len(ir.processes)} — {', '.join(p.kind for p in ir.processes)}",
        f"- BCs: {len(ir.boundary_conditions)}",
        f"- ICs: {len(ir.initial_conditions)}",
        f"- observables: {len(ir.observables)}",
        f"- time horizon: {ir.compute.time_horizon_s:g} s",
    ]
    if ir.assumption_hints:
        lines.append("- author-declared assumptions:")
        for h in ir.assumption_hints:
            lines.append(f"  - {h.id}: {h.description}")
    return "\n".join(lines)
