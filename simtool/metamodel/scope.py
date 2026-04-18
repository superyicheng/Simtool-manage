"""Scope contract: what a meta-model covers and what it does not.

Behavior-based, not count-based — the contract is:

  - For any (parameter_id, context), if the corpus has >= 2 comparable
    measurements, the meta-model produces a reconciliation (RECONCILED).
  - If only 1 measurement, a single value with a quality rating (SINGLE).
  - Otherwise the parameter is MISSING.

MISSING parameters block simulation runs — the user must either supply
a value explicitly or accept an AI-suggested default with a speculation
flag in the assumption ledger. Silent fallback defaults are not allowed.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from simtool.connector.ir import ParameterBinding, ScientificModel
from simtool.metamodel.library import MetaModel


class ParameterStatus(str, Enum):
    RECONCILED = "reconciled"             # >= 2 comparable records
    SINGLE = "single_with_rating"         # exactly 1 record
    MISSING = "missing"                   # 0 records


class ScopeContract(BaseModel):
    """What a meta-model declares it covers.

    The list of ``in_scope_parameter_ids`` is the AUTHORITATIVE set —
    adding or removing from it is a MAJOR version change, by convention.
    ``out_of_scope_notes`` captures explicit non-goals (e.g. 'ignores
    pH effects below 5.5'); these are user-facing caveats.
    """

    in_scope_parameter_ids: list[str] = Field(
        default_factory=list,
        description="Parameters the meta-model commits to reporting on.",
    )
    in_scope_phenomena: list[str] = Field(default_factory=list)
    out_of_scope_notes: list[str] = Field(default_factory=list)
    min_records_for_reconciliation: int = 2


class ParameterStatusReport(BaseModel):
    """Per-parameter status computed against a meta-model + IR."""

    parameter_id: str
    context_keys: dict[str, str] = Field(default_factory=dict)
    status: ParameterStatus
    supporting_record_count: int = 0
    reason: str = ""


class ScopeStatusReport(BaseModel):
    """Full status of an IR against a meta-model's scope contract.

    ``missing_blocking`` are the parameters the IR requires that the
    meta-model cannot provide. Execution is blocked until each is
    resolved (user-supplied value OR speculative default with flag).
    """

    ir_id: str
    meta_model_id: str
    meta_model_version: str
    per_parameter: list[ParameterStatusReport] = Field(default_factory=list)

    @property
    def reconciled(self) -> list[ParameterStatusReport]:
        return [p for p in self.per_parameter if p.status == ParameterStatus.RECONCILED]

    @property
    def single(self) -> list[ParameterStatusReport]:
        return [p for p in self.per_parameter if p.status == ParameterStatus.SINGLE]

    @property
    def missing_blocking(self) -> list[ParameterStatusReport]:
        return [p for p in self.per_parameter if p.status == ParameterStatus.MISSING]

    def is_ready_to_run(self) -> bool:
        """True only when every required parameter has reconciled or single
        coverage. MISSING parameters block until resolved."""
        return not self.missing_blocking


# ---------------------------------------------------------------------------
# Status computation
# ---------------------------------------------------------------------------


def parameter_status(
    meta_model: MetaModel,
    parameter_id: str,
    context_keys: Optional[dict[str, str]] = None,
    scope: Optional[ScopeContract] = None,
) -> ParameterStatusReport:
    """Compute the status of one parameter against a meta-model.

    The threshold ``min_records_for_reconciliation`` comes from the scope
    contract (default 2). A reconciled parameter with fewer supporting
    records than the threshold falls back to SINGLE.
    """
    threshold = scope.min_records_for_reconciliation if scope else 2
    rec = meta_model.find_parameter(parameter_id, context_keys)
    if rec is None:
        return ParameterStatusReport(
            parameter_id=parameter_id,
            context_keys=context_keys or {},
            status=ParameterStatus.MISSING,
            supporting_record_count=0,
            reason="no reconciled parameter for (parameter_id, context)",
        )
    n = len(rec.supporting_record_dois)
    if n >= threshold:
        return ParameterStatusReport(
            parameter_id=parameter_id,
            context_keys=context_keys or {},
            status=ParameterStatus.RECONCILED,
            supporting_record_count=n,
            reason=f"{n} supporting records (>= {threshold})",
        )
    return ParameterStatusReport(
        parameter_id=parameter_id,
        context_keys=context_keys or {},
        status=ParameterStatus.SINGLE,
        supporting_record_count=n,
        reason=f"only {n} supporting record(s); quality={rec.quality_rating.value}",
    )


def evaluate_ir_against_scope(
    ir: ScientificModel,
    meta_model: MetaModel,
    scope: Optional[ScopeContract] = None,
) -> ScopeStatusReport:
    """Compute per-parameter status for every parameter the IR references.

    We walk every ParameterBinding in the IR (entities, processes, BCs,
    ICs) and look each up in the meta-model.
    """
    reports: list[ParameterStatusReport] = []
    seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()

    for binding in _iter_bindings(ir):
        key = (binding.parameter_id, tuple(sorted(binding.context_keys.items())))
        if key in seen:
            continue
        seen.add(key)
        reports.append(
            parameter_status(
                meta_model,
                binding.parameter_id,
                binding.context_keys,
                scope=scope,
            )
        )
    return ScopeStatusReport(
        ir_id=ir.id,
        meta_model_id=meta_model.id,
        meta_model_version=str(meta_model.version),
        per_parameter=reports,
    )


def _iter_bindings(ir: ScientificModel):
    """Walk every ParameterBinding the IR declares."""
    for e in ir.entities:
        params = getattr(e, "parameters", None) or {}
        for b in params.values():
            if isinstance(b, ParameterBinding):
                yield b
    for p in ir.processes:
        params = getattr(p, "parameters", None) or {}
        for b in params.values():
            if isinstance(b, ParameterBinding):
                yield b
    for bc in ir.boundary_conditions:
        if bc.value is not None:
            yield bc.value
        if bc.robin_bulk_value is not None:
            yield bc.robin_bulk_value
    for ic in ir.initial_conditions:
        for b in (ic.parameters or {}).values():
            yield b
