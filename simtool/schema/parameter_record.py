from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class GradeRating(str, Enum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    VERY_LOW = "very_low"


class ExtractionModality(str, Enum):
    PROSE = "prose"
    TABLE = "table"
    FIGURE = "figure"


class MeasurementMethod(str, Enum):
    CHEMOSTAT = "chemostat"
    BATCH_FIT = "batch_fit"
    MICROFLUIDIC = "microfluidic"
    RESPIROMETRY = "respirometry"
    INITIAL_RATE = "initial_rate"
    CITED_FROM_OTHER = "cited_from_other"
    UNSPECIFIED = "unspecified"


class SpanAnchor(BaseModel):
    """Pointer back to the source PDF region a value was extracted from.

    Every ParameterRecord must carry one of these — no span, no record.
    """

    doi: str
    page: int
    bbox: Optional[tuple[float, float, float, float]] = Field(
        default=None,
        description="Bounding box on the page (x0,y0,x1,y1) in PDF points. Required for table/figure modality.",
    )
    text_excerpt: Optional[str] = Field(
        default=None,
        description="Verbatim text the value was drawn from, for prose modality.",
    )
    modality: ExtractionModality


class StudyContext(BaseModel):
    """Experimental context under which the parameter was measured.

    Used by the reconciliation layer to decide whether two records are
    comparable (translate) or belong on different branches (stratify).
    """

    species: Optional[str] = None
    strain: Optional[str] = None
    substrate: Optional[str] = None
    temperature_c: Optional[float] = None
    ph: Optional[float] = None
    dissolved_oxygen_mgL: Optional[float] = None
    redox_regime: Optional[str] = Field(
        default=None, description="aerobic / anoxic / anaerobic / microaerobic"
    )
    culture_mode: Optional[str] = Field(
        default=None, description="planktonic / biofilm / flocs / granules"
    )
    notes: Optional[str] = None


class ExtractorAgreement(BaseModel):
    """Multi-agent QC: how many independent extractors produced this value,
    and whether they agreed within tolerance."""

    n_extractors: int
    n_agreed: int
    disagreement_delta: Optional[float] = Field(
        default=None,
        description="Max relative disagreement across extractors, if multiple values were produced.",
    )


class ParameterRecord(BaseModel):
    """A single parameter value extracted from a single study.

    The meta-model is keyed on (parameter_id, StudyContext); this record
    is the atomic evidence unit.
    """

    # Identity
    parameter_id: str = Field(
        description="Canonical parameter key from idynomics_vocab (e.g. 'mu_max', 'K_s', 'Y_XS').",
    )
    value: float
    unit: str = Field(description="Original reported unit string; harmonized separately.")
    canonical_value: Optional[float] = Field(
        default=None,
        description="Value after unit harmonization to this parameter's canonical unit. None if not yet harmonized.",
    )

    # Provenance
    citation: str = Field(description="Full citation (author year title journal).")
    doi: str
    span: SpanAnchor
    method: MeasurementMethod

    # Context
    context: StudyContext

    # Quality signals
    extractor_agreement: ExtractorAgreement
    dimensional_check_passed: bool
    range_check_passed: bool
    extractor_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Composite: extractor_agreement x dimensional_check x range_check x source_quality.",
    )
    grade: GradeRating
    grade_reasons: list[str] = Field(
        default_factory=list,
        description="Human-readable reasons that drove the GRADE rating (e.g. 'risk of bias: no replication').",
    )

    # Reconciliation
    branch_id: Optional[str] = Field(
        default=None,
        description="Which sub-model branch this record belongs to, if the context is on a stratified axis.",
    )
    conflict_flags: list[str] = Field(
        default_factory=list,
        description="Free-text flags surfaced by conflict detection (e.g. 'conflicts with record X: 3x higher Ks').",
    )

    # Bookkeeping
    extracted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    extractor_version: Optional[str] = None
