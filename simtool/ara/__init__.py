"""Ara deployment layer — message-shaped rendering + session entrypoints.

Ara is a message-only distribution channel: no frontend. The codebase is
deployed as a folder on a cloud computer; an agent running in Ara
invokes these renderers and attaches the resulting text + images to chat
messages.

Text renderers live in ``render_text``; image renderers live in
``render_image`` (lazy matplotlib — raises a clear error if matplotlib
is not installed).

Design rule: renderers are pure. Object in, string (or PNG bytes) out.
The agent composes them into a multi-part reply.
"""

from simtool.ara.render_text import (
    render_adjustment_proposal,
    render_assumption_ledger,
    render_fit_result,
    render_ir_compact,
    render_metamodel_parameter,
    render_metamodel_parameters,
    render_metamodel_summary,
    render_output_bundle,
    render_panel_overrides,
    render_panel_summary,
    render_progress_line,
    render_progress_stream,
    render_recommendation,
    render_scope_status,
    render_submodel_hierarchy,
    render_suggestion,
    render_suggestion_ledger_summary,
)

try:
    from simtool.ara.render_image import (
        ImageRenderingNotAvailable,
        render_distribution,
        render_output_bundle_overview,
        render_reconciled_parameter,
        render_scalar_timeseries,
    )
except ImportError:  # pragma: no cover
    # render_image itself is ImportError-free; this only fires if a
    # downstream import chain does something exotic. Keep a clean fallback.
    pass

from simtool.ara.session import AraSession

__all__ = [
    "AraSession",
    "render_adjustment_proposal",
    "render_assumption_ledger",
    "render_fit_result",
    "render_ir_compact",
    "render_metamodel_parameter",
    "render_metamodel_parameters",
    "render_metamodel_summary",
    "render_output_bundle",
    "render_panel_overrides",
    "render_panel_summary",
    "render_progress_line",
    "render_progress_stream",
    "render_recommendation",
    "render_scope_status",
    "render_submodel_hierarchy",
    "render_suggestion",
    "render_suggestion_ledger_summary",
    # image renderers — available if matplotlib is installed.
    "ImageRenderingNotAvailable",
    "render_distribution",
    "render_output_bundle_overview",
    "render_reconciled_parameter",
    "render_scalar_timeseries",
]
