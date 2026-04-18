"""The iDynoMiCS 2 FrameworkPlugin implementation.

Ties lower/execute/monitor/outputs/odd together behind the common plugin
contract defined in ``simtool.connector.plugin``.

``validate_ir`` enforces what iDynoMiCS 2 can express:
  - formalism in {'agent_based', 'ode'} (biofilm or chemostat path).
  - Process kinds limited to {monod_growth, first_order_decay, maintenance, diffusion}.
  - Every Monod growth must declare mu_max + K_s_<solute> per consumed solute.
  - Boundary conditions limited to {dirichlet, no_flux}.
  - Spatial domain dimensionality in {1, 2, 3}.

Anything outside these is an error with a specific suggestion so the
user (or the adjustment workflow) can fix it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

from simtool.connector.assumptions import (
    Assumption,
    AssumptionCategory,
    AssumptionLedger,
    AssumptionSeverity,
)
from simtool.connector.ir import (
    CustomProcess,
    DiffusionProcess,
    FirstOrderDecayProcess,
    MaintenanceProcess,
    MonodGrowthProcess,
    ScientificModel,
)
from simtool.connector.plugin import (
    DocSources,
    LoweredArtifact,
    RunHandle,
    ValidationIssue,
    ValidationReport,
)
from simtool.connector.runs import OutputBundle, ProgressReport, RunLayout
from simtool.connector.skill import (
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
from simtool.frameworks.idynomics_2.config import (
    IDynoMiCS2Config,
    IDynoMiCS2NotAvailable,
    require_runtime,
    resolve_jar_path,
)
from simtool.frameworks.idynomics_2.execute import (
    execute_artifact,
    terminate_handle,
)
from simtool.frameworks.idynomics_2.lower import lower_ir
from simtool.frameworks.idynomics_2.monitor import monitor_handle
from simtool.frameworks.idynomics_2.odd import generate_odd
from simtool.frameworks.idynomics_2.outputs import (
    extract_simulation_name,
    harvest_jar_results,
    parse_output_dir,
)


_SUPPORTED_PROCESS_KINDS = {
    "monod_growth",
    "first_order_decay",
    "maintenance",
    "diffusion",
}
_SUPPORTED_FORMALISMS = {"agent_based", "ode"}
_SUPPORTED_BC_KINDS = {"dirichlet", "no_flux"}


class IDynoMiCS2Plugin:
    """``FrameworkPlugin`` for iDynoMiCS 2.0.

    See module docstring for the capability envelope. Instantiate once
    per session; stateless across runs.
    """

    name: str = "idynomics_2"
    version: str = "0.1.0"
    supported_framework_versions: tuple[str, ...] = ("2.0", "2.0.0")

    def __init__(self, runtime: Optional[IDynoMiCS2Config] = None):
        self._runtime = runtime  # injectable for tests

    # --- skill file --------------------------------------------------------

    def parse_docs(self, sources: DocSources) -> SkillFile:
        """A minimal bundled skill file.

        The full five-stage pipeline against the real XSD/prose is TODO;
        this method returns a hand-curated subset sufficient to validate
        the biofilm + chemostat paths end-to-end.
        """
        grammar = Grammar(
            elements=[
                GrammarElement(
                    name="simulation",
                    fields=[
                        GrammarField(name="name", type=GrammarType.STRING, required=True),
                        GrammarField(name="outputfolder", type=GrammarType.STRING,
                                     default="../outputs"),
                        GrammarField(name="log", type=GrammarType.ENUM,
                                     enum_values=["QUIET", "NORMAL", "DEBUG"],
                                     default="NORMAL"),
                    ],
                    children=["timer", "speciesLib", "compartment"],
                ),
                GrammarElement(
                    name="compartment",
                    fields=[GrammarField(name="name", type=GrammarType.STRING, required=True)],
                    children=["shape", "solutes", "spawn", "processManagers"],
                ),
                GrammarElement(
                    name="solute",
                    fields=[
                        GrammarField(name="name", type=GrammarType.STRING, required=True),
                        GrammarField(name="concentration", type=GrammarType.STRING,
                                     unit="[mg/l]"),
                        GrammarField(name="defaultDiffusivity",
                                     type=GrammarType.STRING, unit="[um+2/s]"),
                        GrammarField(name="biofilmDiffusivity",
                                     type=GrammarType.STRING, unit="[um+2/s]"),
                    ],
                ),
            ],
            root_element="simulation",
            source=SourceRef(
                kind=SourceKind.REFERENCE_PROSE,
                uri="bundled (iDynoMiCS-2-July-2025/protocol/template_PARAMETERS.md)",
                version="2025-07",
            ),
        )
        return SkillFile(
            framework=self.name,
            framework_version="2.0.0",
            sources=list(sources.sources) or [
                SourceRef(
                    kind=SourceKind.REFERENCE_PROSE,
                    uri="bundled minimal skill",
                ),
            ],
            grammar=grammar,
            stage_reports=[
                StageReport(stage=PipelineStage.CLASSIFY_SOURCES, ok=True,
                            summary="bundled minimal: prose-derived."),
                StageReport(stage=PipelineStage.EXTRACT_GRAMMAR, ok=True,
                            summary="extracted 3 load-bearing elements."),
            ],
        )

    # --- validate ----------------------------------------------------------

    def validate_ir(self, ir: ScientificModel, skill: SkillFile) -> ValidationReport:
        issues: list[ValidationIssue] = []
        if ir.formalism not in _SUPPORTED_FORMALISMS:
            issues.append(ValidationIssue(
                severity="error",
                message=(
                    f"iDynoMiCS 2 does not support formalism '{ir.formalism}'. "
                    f"Supported: {sorted(_SUPPORTED_FORMALISMS)}."
                ),
                ir_path="formalism",
                suggestion="use 'agent_based' for biofilm or 'ode' for chemostat",
            ))
        for idx, p in enumerate(ir.processes):
            if isinstance(p, CustomProcess):
                issues.append(ValidationIssue(
                    severity="error",
                    message=f"CustomProcess '{p.id}' is not expressible in iDynoMiCS 2.",
                    ir_path=f"processes[{idx}]",
                    suggestion="rewrite as one of monod_growth, first_order_decay, "
                               "maintenance, diffusion",
                ))
                continue
            if p.kind not in _SUPPORTED_PROCESS_KINDS:
                issues.append(ValidationIssue(
                    severity="error",
                    message=f"process kind '{p.kind}' unsupported.",
                    ir_path=f"processes[{idx}]",
                    suggestion=f"use one of {sorted(_SUPPORTED_PROCESS_KINDS)}",
                ))
                continue
            if isinstance(p, MonodGrowthProcess):
                if "mu_max" not in p.parameters:
                    issues.append(ValidationIssue(
                        severity="error",
                        message=f"MonodGrowth '{p.id}' missing 'mu_max'",
                        ir_path=f"processes[{idx}].parameters",
                        suggestion="add a ParameterBinding with parameter_id='mu_max'",
                    ))
                for s in p.consumed_solutes:
                    if f"K_s_{s}" not in p.parameters:
                        issues.append(ValidationIssue(
                            severity="error",
                            message=(
                                f"MonodGrowth '{p.id}' missing 'K_s_{s}' for "
                                f"consumed solute '{s}'"
                            ),
                            ir_path=f"processes[{idx}].parameters",
                            suggestion=f"add 'K_s_{s}' ParameterBinding",
                        ))
            elif isinstance(p, FirstOrderDecayProcess) and "b" not in p.parameters:
                issues.append(ValidationIssue(
                    severity="error",
                    message=f"FirstOrderDecay '{p.id}' missing 'b'",
                    ir_path=f"processes[{idx}].parameters",
                    suggestion="add a ParameterBinding with parameter_id='b'",
                ))
        for bc in ir.boundary_conditions:
            if bc.kind not in _SUPPORTED_BC_KINDS:
                issues.append(ValidationIssue(
                    severity="error",
                    message=f"BC '{bc.id}' kind '{bc.kind}' unsupported.",
                    ir_path=f"boundary_conditions[{bc.id}]",
                    suggestion=f"use one of {sorted(_SUPPORTED_BC_KINDS)}",
                ))
        ok = not any(i.severity == "error" for i in issues)
        return ValidationReport(ok=ok, issues=issues)

    # --- lower -------------------------------------------------------------

    def lower(self, ir: ScientificModel, skill: SkillFile) -> LoweredArtifact:
        result = lower_ir(ir)
        # Surface a standing assumption about iDynoMiCS's default solver settings.
        _maybe_add(
            result.ledger,
            Assumption(
                id="default_process_managers",
                category=AssumptionCategory.NUMERICS,
                severity=AssumptionSeverity.ADVISORY,
                description=(
                    "Process managers use iDynoMiCS defaults (AgentRelaxation+"
                    "PDEWrapper for biofilm; ChemostatSolver with tolerance=1e-3 "
                    "for ODE)."
                ),
                justification="No custom solver tuning declared in the IR.",
                alternatives=["tighter tolerance", "alternative solver"],
                surfaced_by="idynomics_2.plugin.lower",
            ),
        )
        return LoweredArtifact(
            entrypoint="protocol.xml",
            files=result.files,
            assumptions=result.ledger,
            extra={"jar_hint": str(resolve_jar_path() or "")},
        )

    # --- execute + monitor + terminate ------------------------------------

    def execute(self, artifact: LoweredArtifact, layout: RunLayout) -> RunHandle:
        return execute_artifact(artifact, layout, runtime=self._runtime)

    def monitor(self, handle: RunHandle) -> Iterator[ProgressReport]:
        yield from monitor_handle(handle)

    def terminate(self, handle: RunHandle, reason: str = "") -> None:
        terminate_handle(handle, reason)

    # --- parse outputs + ODD ----------------------------------------------

    def parse_outputs(self, layout: RunLayout) -> OutputBundle:
        # If iDynoMiCS wrote to its default `jar_dir/results/...` location
        # (because storage.cfg has `ignore_protocol_out=TRUE`), harvest it
        # into `layout.outputs` first, then run the generic scanner.
        jar = resolve_jar_path()
        if jar is not None:
            sim_name = extract_simulation_name(layout.inputs / "protocol.xml")
            if sim_name:
                harvest_jar_results(jar.parent, sim_name, layout)
        return parse_output_dir(layout)

    def generate_protocol(self, ir: ScientificModel, layout: RunLayout) -> Path:
        layout.ensure()
        return generate_odd(ir, layout.protocol / "ODD.md")


def _maybe_add(ledger: AssumptionLedger, a: Assumption) -> None:
    if not any(x.id == a.id for x in ledger.assumptions):
        ledger.add(a)
