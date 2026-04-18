"""Base extractor: a method-agnostic Anthropic-powered parameter extractor.

Design notes:

  - **Direct Anthropic SDK** — reproducibility, prompt caching, Batch API
    access, deterministic via temperature=0. Not Claude Code / Agent SDK.
  - **Tool-use forces structure** — we declare a `report_parameters` tool
    and pin `tool_choice={"type": "tool", "name": ...}`, so the model's
    only permitted response is JSON that conforms to our JSONSchema.
    Free-form hallucination risk is removed.
  - **Prompt caching** — the static system prompt (rules + vocab + tool
    instructions) carries `cache_control: ephemeral`. Across many PDFs
    this cache hit drops input-token cost by ~90%.
  - **Multimodal** — Claude supports PDF input natively via the
    `document` content block. We send the PDF bytes base64-encoded, and
    the model sees both text and figures.
  - **Extractor is method-agnostic at the class level**; the specific
    method it focuses on is passed in via `MethodProfile`. A subclass per
    measurement method exists only so each has a documented prose hint
    section specialized to that method's conventions.
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

from pydantic import ValidationError

from simtool.extractors.schemas import REPORT_PARAMETERS_TOOL, RawExtraction
from simtool.schema.idynomics_vocab import VOCAB
from simtool.schema.parameter_record import MeasurementMethod

logger = logging.getLogger(__name__)


# Default model. Sonnet 4.6 is the sweet spot: strong multimodal + structured
# output, much cheaper than Opus at extraction scale. Users can override.
DEFAULT_MODEL = "claude-sonnet-4-6"

# Conservative max_tokens — extractors never need long free-form output;
# all output goes through a structured tool call.
DEFAULT_MAX_TOKENS = 8192


@dataclass
class MethodProfile:
    """Describes how to coach the extractor for one measurement method.

    `method`: which MeasurementMethod this profile is for.
    `prose_hints`: bullet-list of what this method looks like in papers —
        shown in the system prompt so the model distinguishes it from
        other methods in the same paper.
    `expected_parameters`: parameter_ids this method typically yields.
        Used to bias the model toward the method's canonical outputs and
        to down-rank method/parameter combinations that don't make sense
        (e.g. cell_density from a chemostat paper).
    """

    method: MeasurementMethod
    prose_hints: list[str] = field(default_factory=list)
    expected_parameters: list[str] = field(default_factory=list)


class AnthropicClientLike(Protocol):
    """Minimal interface matching `anthropic.Anthropic().messages.create(...)`.

    Declared as a Protocol so tests can pass in mocks without depending on
    the real SDK.
    """

    def messages_create(self, **kwargs: Any) -> Any: ...


@dataclass
class ExtractorConfig:
    model: str = DEFAULT_MODEL
    temperature: float = 0.0
    max_tokens: int = DEFAULT_MAX_TOKENS
    extractor_version: str = "simtool-extractor/0.1"


@dataclass
class ExtractionResult:
    pdf_path: Path
    method: MeasurementMethod
    extractions: list[RawExtraction]
    raw_tool_input: Optional[dict[str, Any]] = None
    stop_reason: Optional[str] = None
    """For observability — usually 'tool_use' on success, 'end_turn' if the model refused to call the tool."""


class BaseExtractor:
    """Base class for method-centric extractors.

    Subclasses override `method_profile`. The class handles everything else:
    PDF loading, prompt assembly, Anthropic call, tool-call parsing,
    validation into `RawExtraction` objects.
    """

    method_profile: MethodProfile  # subclasses must set

    def __init__(
        self,
        client: AnthropicClientLike,
        *,
        config: Optional[ExtractorConfig] = None,
    ):
        self.client = client
        self.config = config or ExtractorConfig()

    # ----- public API -----

    def extract(
        self,
        pdf_path: Path,
        *,
        doi_hint: Optional[str] = None,
    ) -> ExtractionResult:
        """Run the extractor on one PDF, return the list of raw extractions.

        `doi_hint` is passed to the model for its internal bookkeeping;
        not used to construct any ParameterRecord (that happens downstream
        with the known DOI from the corpus manifest).
        """

        pdf_bytes = Path(pdf_path).read_bytes()
        pdf_b64 = base64.standard_b64encode(pdf_bytes).decode("ascii")

        system_blocks = self._build_system_blocks()
        user_message = self._build_user_message(pdf_b64, doi_hint)

        response = self.client.messages_create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            tools=[REPORT_PARAMETERS_TOOL],
            tool_choice={"type": "tool", "name": REPORT_PARAMETERS_TOOL["name"]},
            system=system_blocks,
            messages=[user_message],
        )

        raw_input = _extract_tool_input(response)
        if raw_input is None:
            logger.warning(
                "Extractor %s: no tool_use block in response for %s",
                self.__class__.__name__,
                pdf_path,
            )
            return ExtractionResult(
                pdf_path=Path(pdf_path),
                method=self.method_profile.method,
                extractions=[],
                stop_reason=_stop_reason(response),
            )

        extractions = _validate_raw_extractions(raw_input, self.method_profile.method)
        return ExtractionResult(
            pdf_path=Path(pdf_path),
            method=self.method_profile.method,
            extractions=extractions,
            raw_tool_input=raw_input,
            stop_reason=_stop_reason(response),
        )

    # ----- prompt assembly (subclasses may override) -----

    def _build_system_blocks(self) -> list[dict[str, Any]]:
        """Static-where-possible system prompt with a cache boundary.

        The boundary is placed AFTER the large static content (rules,
        vocab, method hints) but before any per-request content. Because
        a subclass's `method_profile` is part of the class, two different
        subclasses share separate caches; within a subclass the cache is
        reused across every PDF.
        """

        text = "\n".join(
            (
                self._rules_section(),
                "",
                self._vocab_section(),
                "",
                self._method_section(),
            )
        )
        return [
            {
                "type": "text",
                "text": text,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def _build_user_message(
        self,
        pdf_b64: str,
        doi_hint: Optional[str],
    ) -> dict[str, Any]:
        preamble_lines = [
            "Extract every parameter value from this paper that matches the vocabulary.",
            "Call the `report_parameters` tool exactly once with all findings.",
            "If the paper reports no relevant parameters, call the tool with an empty list.",
        ]
        if doi_hint:
            preamble_lines.append(f"(Source DOI: {doi_hint}.)")
        return {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": pdf_b64,
                    },
                },
                {"type": "text", "text": "\n".join(preamble_lines)},
            ],
        }

    def _rules_section(self) -> str:
        return (
            "You are a scientific parameter extractor. You read primary-literature "
            "PDFs and extract quantitative kinetic/biophysical parameters for "
            "microbial-biofilm agent-based modeling.\n"
            "\n"
            "RULES (hard, non-negotiable):\n"
            "  1. Only report values the paper itself reports as its own working "
            "values. Do NOT report values the paper cites from prior literature "
            "unless the paper adopts them as its own (e.g., 'we used Ks=... from [12]').\n"
            "  2. Every value MUST have a verbatim text_excerpt from the PDF. If you "
            "cannot produce a verbatim excerpt, omit the value.\n"
            "  3. Preserve the original reported unit string (do NOT convert units).\n"
            "  4. Attach study context (species, substrate, conditions) from the paper. "
            "If a field is not stated, leave it null — do NOT infer.\n"
            "  5. If a value is ambiguous or you are unsure, SET self_confidence LOW "
            "rather than omitting silently — provenance of uncertainty matters.\n"
            "  6. Only use parameter_ids from the supplied vocabulary. If no vocab id "
            "matches, do not report.\n"
        )

    def _vocab_section(self) -> str:
        lines = ["iDynoMiCS 2 parameter vocabulary (canonical ids):"]
        for entry in VOCAB.values():
            notes = f" — {entry.notes}" if entry.notes else ""
            lines.append(
                f"  - {entry.id}: {entry.description} Canonical unit: {entry.canonical_unit}.{notes}"
            )
        return "\n".join(lines)

    def _method_section(self) -> str:
        p = self.method_profile
        lines = [
            f"FOCUS METHOD: {p.method.value}",
            "",
            "Look for values reported via this method. The paper may also contain "
            "values from other methods — those belong to a different extractor run; "
            "prefer to omit them here rather than mislabel the method.",
        ]
        if p.prose_hints:
            lines.append("")
            lines.append("How this method appears in methods sections:")
            for hint in p.prose_hints:
                lines.append(f"  - {hint}")
        if p.expected_parameters:
            lines.append("")
            lines.append(
                "Parameters this method typically yields (not a strict whitelist): "
                + ", ".join(p.expected_parameters)
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _extract_tool_input(response: Any) -> Optional[dict[str, Any]]:
    """Pull the first tool_use content block's `input` from an Anthropic response.

    Works for both the real SDK's response objects and plain dicts (used in
    tests). We accept either attribute-style or dict-style access.
    """

    content = _get(response, "content") or []
    for block in content:
        if _get(block, "type") == "tool_use":
            raw = _get(block, "input")
            if isinstance(raw, dict):
                return raw
            if isinstance(raw, str):
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    return None
    return None


def _stop_reason(response: Any) -> Optional[str]:
    val = _get(response, "stop_reason")
    if isinstance(val, str):
        return val
    return None


def _get(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _validate_raw_extractions(
    raw_input: dict[str, Any], method: MeasurementMethod
) -> list[RawExtraction]:
    items = raw_input.get("extractions")
    if not isinstance(items, list):
        logger.warning("report_parameters.input.extractions is not a list: %r", items)
        return []

    out: list[RawExtraction] = []
    for idx, item in enumerate(items):
        # Coerce reported method if the model omitted it — we know the profile's method.
        if isinstance(item, dict) and not item.get("method"):
            item = {**item, "method": method.value}
        try:
            out.append(RawExtraction.model_validate(item))
        except ValidationError as exc:
            logger.warning(
                "Dropping extraction %d: validation failed: %s", idx, exc.errors()[:2]
            )
    return out
