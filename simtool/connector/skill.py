"""Per-framework skill file.

A skill file is a persistent, versioned artifact that captures a simulation
framework's input *language* — the schema, the idioms, and the worked
examples the connector needs to lower an IR into that framework's native
code. One skill file per (framework, framework_version).

It is built once (by the five-stage pipeline below), cached, and refreshed
when the framework version moves. The connector reads skill files; it does
not build them per-run.

Three layers — deliberately separated so each layer can be refreshed on
its own cadence:
  1. Grammar:     the structured, authoritative schema — element names,
                  parameter types, constraints. Comes from the most
                  authoritative source available (formal schema > parser
                  source > host-language API > prose).
  2. Annotations: prose-derived enrichment — intent of each element,
                  typical values, known idioms, deprecations.
  3. Examples:    curated, validated input files exercising the grammar.
                  Round-tripped through Stage 4 validation; discrepancies
                  flagged.

Five-stage learning pipeline (run once per framework version):
  Stage 1 — classify_sources:   enumerate authoritative sources that exist.
  Stage 2 — extract_grammar:    ingest the most authoritative source, emit
                                 a typed grammar.
  Stage 3 — annotate:           enrich grammar with prose docs.
  Stage 4 — validate_corpus:    round-trip known-good input files against
                                 the grammar; flag every discrepancy.
  Stage 5 — targeted_probe:     ONLY to close specific gaps identified by
                                 Stage 4. Never a primary learning strategy.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Source classification (Stage 1)
# ---------------------------------------------------------------------------


class SourceKind(str, Enum):
    """Authoritativeness ranking, high -> low."""

    FORMAL_SCHEMA = "formal_schema"          # XSD, JSON Schema, protobuf, etc.
    PARSER_SOURCE = "parser_source"          # reading the actual parser code
    HOST_LANGUAGE_API = "host_language_api"  # typed API surface (Java/Python classes)
    REFERENCE_PROSE = "reference_prose"      # official prose docs, manual
    COMMUNITY_PROSE = "community_prose"      # wikis, tutorials, forum threads


class SourceRef(BaseModel):
    """Pointer to a single source ingested at any stage."""

    kind: SourceKind
    uri: str = Field(description="Local path, URL, or versioned identifier.")
    version: Optional[str] = None
    notes: str = ""


# ---------------------------------------------------------------------------
# Grammar (Stage 2)
# ---------------------------------------------------------------------------


class GrammarType(str, Enum):
    STRING = "string"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    ENUM = "enum"
    LIST = "list"
    OBJECT = "object"
    REFERENCE = "reference"


class GrammarField(BaseModel):
    """A single field/attribute in the framework's input language."""

    name: str
    type: GrammarType
    required: bool = False
    description: str = ""
    enum_values: list[str] = Field(default_factory=list)
    default: Optional[str] = None
    unit: Optional[str] = Field(
        default=None,
        description="Framework-native unit (free text). Unit conversion into "
        "canonical IR units is handled by the plugin's lower() method.",
    )
    constraints: dict[str, Any] = Field(
        default_factory=dict,
        description="Numeric bounds, regex patterns, referential integrity, "
        "etc., as supplied by the authoritative source.",
    )
    item_type: Optional[str] = Field(
        default=None,
        description="For LIST: name of the element type.",
    )
    object_name: Optional[str] = Field(
        default=None,
        description="For OBJECT/REFERENCE: name of the referenced GrammarElement.",
    )


class GrammarElement(BaseModel):
    """A named element (XML tag, JSON object type, record) in the language."""

    name: str
    fields: list[GrammarField] = Field(default_factory=list)
    children: list[str] = Field(
        default_factory=list,
        description="Names of GrammarElements that may nest inside this one.",
    )
    description: str = ""


class Grammar(BaseModel):
    """The structured schema of a framework's input language."""

    elements: list[GrammarElement] = Field(default_factory=list)
    root_element: Optional[str] = None
    source: Optional[SourceRef] = None


# ---------------------------------------------------------------------------
# Annotations (Stage 3)
# ---------------------------------------------------------------------------


class Annotation(BaseModel):
    """Prose-derived enrichment keyed to a grammar location.

    ``target`` is ``element_name`` or ``element_name.field_name`` or a
    dotted path for nested access.
    """

    target: str
    intent: str = ""
    typical_values: str = ""
    idiom: str = Field(
        default="",
        description="Common usage pattern — how experienced users combine "
        "this element with others.",
    )
    deprecation: Optional[str] = None
    sources: list[SourceRef] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Examples (Stage 4 + 5)
# ---------------------------------------------------------------------------


class Example(BaseModel):
    """A validated worked example of a framework input.

    ``validated_against`` records whether the example round-trips against
    the grammar. Any discrepancy is listed in ``validation_notes`` and
    becomes a gap Stage 5 must resolve.
    """

    name: str
    description: str = ""
    input_text: str = Field(description="Verbatim framework input code.")
    input_format: str = Field(description="e.g. 'xml', 'lua', 'json'.")
    validated_against_grammar: bool = False
    validation_notes: list[str] = Field(default_factory=list)
    covers_elements: list[str] = Field(
        default_factory=list,
        description="Grammar element names this example exercises.",
    )


# ---------------------------------------------------------------------------
# Stage pipeline and reports
# ---------------------------------------------------------------------------


class PipelineStage(str, Enum):
    CLASSIFY_SOURCES = "classify_sources"
    EXTRACT_GRAMMAR = "extract_grammar"
    ANNOTATE = "annotate"
    VALIDATE_CORPUS = "validate_corpus"
    TARGETED_PROBE = "targeted_probe"


class StageReport(BaseModel):
    stage: PipelineStage
    ok: bool
    summary: str
    gaps: list[str] = Field(
        default_factory=list,
        description="Specific items the next stage must address. Stage 5 "
        "consumes Stage 4's gaps as its entire scope.",
    )
    ran_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------


SKILL_SCHEMA_VERSION = "0.1.0"


class SkillFile(BaseModel):
    """Three-layer skill file for a single (framework, framework_version).

    Built by the five-stage pipeline once per framework version; persisted
    and cached. The connector reads skill files during every lowering; it
    does not rebuild them.
    """

    framework: str
    framework_version: str

    sources: list[SourceRef] = Field(
        default_factory=list,
        description="Output of Stage 1 — every source classified during "
        "building, ordered by authoritativeness.",
    )
    grammar: Grammar = Field(
        default_factory=Grammar,
        description="Output of Stage 2 — structured schema.",
    )
    annotations: list[Annotation] = Field(
        default_factory=list,
        description="Output of Stage 3 — prose-derived enrichment.",
    )
    examples: list[Example] = Field(
        default_factory=list,
        description="Output of Stage 4 — validated worked examples.",
    )
    stage_reports: list[StageReport] = Field(default_factory=list)

    built_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    skill_schema_version: str = SKILL_SCHEMA_VERSION

    def get_element(self, name: str) -> Optional[GrammarElement]:
        for el in self.grammar.elements:
            if el.name == name:
                return el
        return None

    def get_annotations_for(self, target: str) -> list[Annotation]:
        return [a for a in self.annotations if a.target == target]
