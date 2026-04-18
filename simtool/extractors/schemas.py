"""Schema definitions for extractor output.

The extractor emits `RawExtraction` records — shaped to match the evidence it
found in the source paper, without post-hoc quality signals. A later pipeline
stage (QC + reconciliation) turns these into full `ParameterRecord` objects.

Two separate shapes live here:

  - `RawExtraction` (Pydantic) — the final Python object our pipeline
    consumes.
  - `REPORT_PARAMETERS_TOOL` (dict) — the JSONSchema tool definition we
    feed to the Anthropic SDK via the `tools=[...]` parameter. The model
    is forced to call this tool (`tool_choice`), which guarantees the
    output is valid JSON conforming to this schema and removes the risk
    of "free-form" hallucinated values.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from simtool.schema.parameter_record import ExtractionModality, MeasurementMethod


class RawSpanAnchor(BaseModel):
    """Where in the PDF the extractor found this value.

    Verbatim text excerpts are mandatory for prose extractions so a
    downstream QC step can verify the anchor against rendered PDF text
    and reject unanchored values. Page numbers are 1-indexed.
    """

    page: int = Field(ge=1)
    text_excerpt: Optional[str] = Field(
        default=None,
        description="Verbatim text span for prose/table; caption for figure.",
    )
    modality: ExtractionModality


class RawStudyContext(BaseModel):
    """Experimental context the extractor attached to this value.

    All fields optional — only what the paper actually reports.
    """

    species: Optional[str] = None
    strain: Optional[str] = None
    substrate: Optional[str] = None
    temperature_c: Optional[float] = None
    ph: Optional[float] = None
    dissolved_oxygen_mgL: Optional[float] = None
    redox_regime: Optional[str] = None
    culture_mode: Optional[str] = None
    notes: Optional[str] = None


class RawExtraction(BaseModel):
    """One (parameter, value, context, span) extracted from one PDF by one agent.

    Intentionally lightweight — no GRADE, no multi-agent agreement, no
    harmonized value. Those are computed downstream.
    """

    parameter_id: str = Field(description="Must be a known id in simtool.schema.idynomics_vocab.")
    value: float
    unit: str = Field(description="Original reported unit, verbatim.")
    method: MeasurementMethod
    context: RawStudyContext
    span: RawSpanAnchor
    self_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Extractor's own confidence that the value is correct and the context is faithful. Used only as a signal; final confidence is composite.",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Free-text caveats the extractor wants to flag (e.g. 'implied value, not directly stated'; 'fitted parameter from curve digitization').",
    )


# ---------------------------------------------------------------------------
# Anthropic tool schema — the model is forced to emit this shape.
# ---------------------------------------------------------------------------

_MODALITY_VALUES = [m.value for m in ExtractionModality]
_METHOD_VALUES = [m.value for m in MeasurementMethod]


REPORT_PARAMETERS_TOOL: dict[str, Any] = {
    "name": "report_parameters",
    "description": (
        "Report every quantitative parameter value you found in the supplied paper "
        "that corresponds to an entry in the provided iDynoMiCS 2 vocabulary. "
        "Only include values that are explicitly reported (numeric) in the paper — "
        "NOT values cited from prior literature unless the paper adopts them as its "
        "own working values. Every value MUST include a verbatim text excerpt from "
        "the PDF so the claim can be verified. If unsure, omit."
    ),
    "input_schema": {
        "type": "object",
        "required": ["extractions"],
        "properties": {
            "extractions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": [
                        "parameter_id",
                        "value",
                        "unit",
                        "method",
                        "context",
                        "span",
                        "self_confidence",
                    ],
                    "properties": {
                        "parameter_id": {
                            "type": "string",
                            "description": "Canonical id from the supplied vocabulary.",
                        },
                        "value": {"type": "number"},
                        "unit": {
                            "type": "string",
                            "description": "Verbatim reported unit (e.g. 'd-1', 'mg N/L', 'um^2/s').",
                        },
                        "method": {
                            "type": "string",
                            "enum": _METHOD_VALUES,
                            "description": (
                                "How the value was measured. Use 'unspecified' only if "
                                "truly not stated; prefer the specific method when possible."
                            ),
                        },
                        "context": {
                            "type": "object",
                            "properties": {
                                "species": {"type": ["string", "null"]},
                                "strain": {"type": ["string", "null"]},
                                "substrate": {"type": ["string", "null"]},
                                "temperature_c": {"type": ["number", "null"]},
                                "ph": {"type": ["number", "null"]},
                                "dissolved_oxygen_mgL": {"type": ["number", "null"]},
                                "redox_regime": {"type": ["string", "null"]},
                                "culture_mode": {"type": ["string", "null"]},
                                "notes": {"type": ["string", "null"]},
                            },
                        },
                        "span": {
                            "type": "object",
                            "required": ["page", "modality"],
                            "properties": {
                                "page": {"type": "integer", "minimum": 1},
                                "text_excerpt": {
                                    "type": ["string", "null"],
                                    "description": "Verbatim text from the PDF anchoring the value. Required for prose/table.",
                                },
                                "modality": {
                                    "type": "string",
                                    "enum": _MODALITY_VALUES,
                                },
                            },
                        },
                        "self_confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                        "notes": {"type": ["string", "null"]},
                    },
                },
            }
        },
    },
}
