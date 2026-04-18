"""Skill-file module tests.

Covers: construction, element/annotation lookup, stage-report attachment,
JSON round-trip, enum validity.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from simtool.connector.skill import (
    SKILL_SCHEMA_VERSION,
    Annotation,
    Example,
    Grammar,
    GrammarElement,
    GrammarField,
    GrammarType,
    PipelineStage,
    SkillFile,
    SourceKind,
    SourceRef,
    StageReport,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def grammar_fragment() -> Grammar:
    compartment = GrammarElement(
        name="compartment",
        description="Reactor compartment holding species and solutes.",
        fields=[
            GrammarField(
                name="name",
                type=GrammarType.STRING,
                required=True,
                description="Compartment identifier.",
            ),
            GrammarField(
                name="shape",
                type=GrammarType.ENUM,
                required=True,
                enum_values=["dimensionless", "rectangle", "cuboid"],
            ),
        ],
        children=["solute", "species"],
    )
    solute = GrammarElement(
        name="solute",
        fields=[
            GrammarField(name="name", type=GrammarType.STRING, required=True),
            GrammarField(
                name="defaultDiffusivity",
                type=GrammarType.FLOAT,
                required=True,
                unit="um^2/s",
                constraints={"minimum": 0.0},
            ),
        ],
    )
    return Grammar(
        elements=[compartment, solute],
        root_element="compartment",
        source=SourceRef(
            kind=SourceKind.FORMAL_SCHEMA, uri="file:///idynomics/protocol.xsd"
        ),
    )


@pytest.fixture
def skill_file(grammar_fragment: Grammar) -> SkillFile:
    return SkillFile(
        framework="idynomics_2",
        framework_version="2.0.0",
        sources=[
            SourceRef(kind=SourceKind.FORMAL_SCHEMA, uri="protocol.xsd"),
            SourceRef(
                kind=SourceKind.REFERENCE_PROSE,
                uri="template_PARAMETERS.md",
                version="2025-07",
            ),
        ],
        grammar=grammar_fragment,
        annotations=[
            Annotation(
                target="solute.defaultDiffusivity",
                intent="Diffusion coefficient of the solute in bulk liquid.",
                typical_values="O2: ~2000 um^2/s; NH4: ~1800 um^2/s.",
                idiom="Usually paired with biofilmDiffusivity ~60-80% of bulk.",
            ),
            Annotation(
                target="solute",
                intent="Continuum chemical field.",
            ),
        ],
        examples=[
            Example(
                name="chemostat_minimal",
                description="Minimal chemostat with one solute.",
                input_text="<protocol><compartment name='c'/></protocol>",
                input_format="xml",
                validated_against_grammar=True,
                covers_elements=["compartment"],
            ),
        ],
        stage_reports=[
            StageReport(
                stage=PipelineStage.CLASSIFY_SOURCES,
                ok=True,
                summary="Found XSD; prose docs supplemental.",
            ),
            StageReport(
                stage=PipelineStage.EXTRACT_GRAMMAR,
                ok=True,
                summary="Extracted 2 elements from XSD.",
            ),
            StageReport(
                stage=PipelineStage.ANNOTATE,
                ok=True,
                summary="Annotated 2 targets from prose.",
            ),
            StageReport(
                stage=PipelineStage.VALIDATE_CORPUS,
                ok=False,
                summary="1 of 5 examples failed round-trip.",
                gaps=["solute.biofilmDiffusivity field missing from grammar"],
            ),
            StageReport(
                stage=PipelineStage.TARGETED_PROBE,
                ok=True,
                summary="Closed all gaps from Stage 4.",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Construction + metadata
# ---------------------------------------------------------------------------


def test_skill_file_construction(skill_file: SkillFile) -> None:
    assert skill_file.framework == "idynomics_2"
    assert skill_file.framework_version == "2.0.0"
    assert skill_file.skill_schema_version == SKILL_SCHEMA_VERSION
    assert len(skill_file.grammar.elements) == 2
    assert skill_file.grammar.root_element == "compartment"


def test_all_five_stage_reports_present(skill_file: SkillFile) -> None:
    stages = {r.stage for r in skill_file.stage_reports}
    assert stages == {
        PipelineStage.CLASSIFY_SOURCES,
        PipelineStage.EXTRACT_GRAMMAR,
        PipelineStage.ANNOTATE,
        PipelineStage.VALIDATE_CORPUS,
        PipelineStage.TARGETED_PROBE,
    }


# ---------------------------------------------------------------------------
# Grammar lookup
# ---------------------------------------------------------------------------


def test_get_element_hit(skill_file: SkillFile) -> None:
    el = skill_file.get_element("solute")
    assert el is not None
    assert any(f.name == "defaultDiffusivity" for f in el.fields)


def test_get_element_miss_returns_none(skill_file: SkillFile) -> None:
    assert skill_file.get_element("nonexistent") is None


def test_get_annotations_for_target(skill_file: SkillFile) -> None:
    anns = skill_file.get_annotations_for("solute.defaultDiffusivity")
    assert len(anns) == 1
    assert "Diffusion" in anns[0].intent


def test_get_annotations_for_target_miss(skill_file: SkillFile) -> None:
    assert skill_file.get_annotations_for("ghost") == []


def test_get_annotations_multiple_same_target() -> None:
    sf = SkillFile(
        framework="x",
        framework_version="0.1",
        annotations=[
            Annotation(target="foo", intent="first"),
            Annotation(target="foo", intent="second"),
            Annotation(target="bar", intent="elsewhere"),
        ],
    )
    foo_anns = sf.get_annotations_for("foo")
    assert len(foo_anns) == 2
    assert {a.intent for a in foo_anns} == {"first", "second"}


# ---------------------------------------------------------------------------
# Enum coverage
# ---------------------------------------------------------------------------


def test_all_source_kinds_cover_authority_spectrum() -> None:
    assert {k.value for k in SourceKind} == {
        "formal_schema",
        "parser_source",
        "host_language_api",
        "reference_prose",
        "community_prose",
    }


def test_all_grammar_types_available() -> None:
    assert {t.value for t in GrammarType} == {
        "string", "int", "float", "bool", "enum", "list", "object", "reference",
    }


def test_pipeline_stage_order_matches_spec() -> None:
    expected = [
        "classify_sources",
        "extract_grammar",
        "annotate",
        "validate_corpus",
        "targeted_probe",
    ]
    assert [s.value for s in PipelineStage] == expected


def test_unknown_source_kind_rejected() -> None:
    with pytest.raises(ValidationError):
        SourceRef(kind="random_tweet", uri="x")


def test_unknown_grammar_type_rejected() -> None:
    with pytest.raises(ValidationError):
        GrammarField(name="x", type="tensor")


def test_unknown_pipeline_stage_rejected() -> None:
    with pytest.raises(ValidationError):
        StageReport(stage="hallucinate", ok=False, summary="x")


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_skill_file_json_round_trip(skill_file: SkillFile) -> None:
    payload = skill_file.model_dump_json()
    restored = SkillFile.model_validate_json(payload)
    assert restored.framework == skill_file.framework
    assert len(restored.grammar.elements) == len(skill_file.grammar.elements)
    assert restored.grammar.elements[0].name == "compartment"
    assert restored.stage_reports[3].ok is False
    assert restored.stage_reports[3].gaps == [
        "solute.biofilmDiffusivity field missing from grammar"
    ]


def test_grammar_with_no_source_ok() -> None:
    """A skill file under construction (Stage 1 not finished) may have a
    grammar without a source yet."""
    g = Grammar(elements=[], source=None)
    assert g.root_element is None


def test_example_validation_state_preserved() -> None:
    ex = Example(
        name="e",
        input_text="<x/>",
        input_format="xml",
        validated_against_grammar=False,
        validation_notes=["missing required 'shape' attribute on compartment"],
    )
    assert ex.validated_against_grammar is False
    assert len(ex.validation_notes) == 1


# ---------------------------------------------------------------------------
# Gaps feed Stage 5 — document the intended flow
# ---------------------------------------------------------------------------


def test_stage4_gaps_are_the_only_input_to_stage5_per_spec(
    skill_file: SkillFile,
) -> None:
    """Per the design spec, Stage 5 targeted probing is the ONLY stage whose
    scope is defined by earlier gaps. We validate this by asserting Stage 4
    reports carry gaps when ok=False, and Stage 5 follows in the sequence."""
    s4 = next(
        r for r in skill_file.stage_reports
        if r.stage == PipelineStage.VALIDATE_CORPUS
    )
    s5 = next(
        r for r in skill_file.stage_reports
        if r.stage == PipelineStage.TARGETED_PROBE
    )
    if not s4.ok:
        assert s4.gaps, "failed Stage 4 must declare gaps for Stage 5 to consume"
    assert s5.ran_at >= s4.ran_at
