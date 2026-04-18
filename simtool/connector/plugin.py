"""FrameworkPlugin Protocol — the common interface every framework
implementation exposes to the connector core.

The core orchestrator knows ONLY this interface. It does not import
iDynoMiCS-, LAMMPS-, or NetLogo-specific modules. Adding a new framework
is a contained engineering task: implement this Protocol, ship a skill
file, add a corpus of validated examples.

The seven methods map onto the lifecycle of a single simulation request:

    1. parse_docs(sources)         -> SkillFile           # one-off per framework version
    2. validate_ir(ir)             -> ValidationReport    # can this framework express it?
    3. lower(ir, skill)            -> LoweredArtifact     # produce framework-native inputs
                                  + AssumptionLedger   # with every implicit choice surfaced
    4. execute(artifact, layout)   -> RunHandle           # kick off the run under the harness
    5. monitor(handle)             -> Iterator[ProgressReport]
    6. parse_outputs(layout)       -> OutputBundle
    7. generate_protocol(ir, layout) -> Path (to ODD/equivalent doc)

Design rules for implementers:
  - validate_ir MUST NOT mutate the IR. Return a report.
  - lower MUST surface every implicit choice into the ledger. Silent
    defaults are a bug.
  - execute MUST NOT block the caller beyond launch; monitoring is the
    streaming channel.
  - parse_outputs MUST tolerate incomplete runs (status != SUCCEEDED) and
    return whatever is available.
  - generate_protocol is NOT optional. Every run produces a reproducibility
    protocol as its primary scientific artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Protocol, runtime_checkable

from simtool.connector.assumptions import AssumptionLedger
from simtool.connector.ir import ScientificModel
from simtool.connector.runs import OutputBundle, ProgressReport, RunLayout
from simtool.connector.skill import SkillFile, SourceRef


# ---------------------------------------------------------------------------
# Plugin-facing data types
# ---------------------------------------------------------------------------


@dataclass
class DocSources:
    """Inputs to ``parse_docs`` — the authoritative sources the plugin
    should ingest. Ordering matters (most authoritative first)."""

    sources: list[SourceRef]


@dataclass
class ValidationIssue:
    """A single validation finding produced by ``validate_ir``."""

    severity: str                    # "error" | "warning" | "info"
    message: str
    ir_path: str = ""                # dotted path into the IR (e.g. "processes[3]")
    suggestion: str = ""


@dataclass
class ValidationReport:
    """Can this framework express this IR?

    ``ok`` is True iff there are no ``error``-severity issues. Warnings
    should be surfaced to the user but do not block.
    """

    ok: bool
    issues: list[ValidationIssue] = field(default_factory=list)

    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]


@dataclass
class LoweredArtifact:
    """The framework-native input package produced by ``lower``.

    ``files`` holds the content to write into ``RunLayout.inputs`` — each
    entry is (relative_path, bytes). ``entrypoint`` names the file the
    framework launches from. ``assumptions`` is the ledger that must be
    user-approved before ``execute`` is allowed.
    """

    entrypoint: str
    files: list[tuple[str, bytes]]
    assumptions: AssumptionLedger
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunHandle:
    """Opaque plugin-side handle for a running simulation.

    The core orchestrator passes this back to ``monitor`` / ``terminate``
    without inspecting it. Plugins typically carry a subprocess handle, a
    container id, or a job id here.
    """

    run_id: str
    layout: RunLayout
    backend: Any
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# The plugin Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class FrameworkPlugin(Protocol):
    """Common interface every framework plugin implements.

    Implementations are typically a concrete class named after the framework
    (e.g. ``IDynoMiCS2Plugin``) living under a dedicated package
    ``simtool.frameworks.<name>``.
    """

    name: str
    """Stable identifier — e.g. 'idynomics_2'. Used to locate skill files."""

    version: str
    """Plugin version (semver). Independent of the framework version it
    targets; both are recorded on every RunRecord."""

    supported_framework_versions: tuple[str, ...]
    """Framework versions this plugin is known to support."""

    # --- Stage-1..5 learning -------------------------------------------------

    def parse_docs(self, sources: DocSources) -> SkillFile:
        """Run the five-stage pipeline end-to-end and emit a SkillFile.

        Typically invoked once per framework version — the result is cached
        and reused across runs. Implementations should internally emit one
        ``StageReport`` per pipeline stage and attach them to the
        ``SkillFile``.
        """
        ...

    # --- Per-request pipeline -----------------------------------------------

    def validate_ir(self, ir: ScientificModel, skill: SkillFile) -> ValidationReport:
        """Report whether ``ir`` is expressible under this framework.

        MUST NOT mutate ``ir``. An unknown process kind, unsupported
        formalism, or IR feature outside the grammar becomes an error.
        """
        ...

    def lower(self, ir: ScientificModel, skill: SkillFile) -> LoweredArtifact:
        """Translate IR -> framework-native inputs + assumption ledger.

        Every implicit choice must be surfaced into the returned
        ``AssumptionLedger``. The artifact is NOT yet written to disk —
        the harness will place its ``files`` under ``RunLayout.inputs``
        only after the ledger is approved.
        """
        ...

    def execute(self, artifact: LoweredArtifact, layout: RunLayout) -> RunHandle:
        """Launch the run. MUST be non-blocking beyond the launch itself.

        Preconditions: ``layout`` directories exist; ``artifact.files``
        have been materialized under ``layout.inputs``; the assumption
        ledger on ``artifact`` is approved.
        """
        ...

    def monitor(self, handle: RunHandle) -> Iterator[ProgressReport]:
        """Yield structured progress reports until the run ends.

        Implementations stream one report per timestep or per periodic
        tick. On run termination (success, failure, or external stop)
        the iterator stops cleanly.
        """
        ...

    def terminate(self, handle: RunHandle, reason: str = "") -> None:
        """Request a clean termination. Safe to call on an already-stopped run."""
        ...

    def parse_outputs(self, layout: RunLayout) -> OutputBundle:
        """Project framework-native outputs under ``layout.outputs`` into a
        framework-agnostic ``OutputBundle`` the comparator can consume.

        Must tolerate incomplete runs — missing observables become empty
        series rather than raising.
        """
        ...

    def generate_protocol(self, ir: ScientificModel, layout: RunLayout) -> Path:
        """Emit the run's reproducibility protocol under ``layout.protocol``.

        For agent-based models this is an ODD document. For other
        formalisms, the community-standard equivalent applies (e.g. SBML
        annotations for ODEs, a PDB-anchored manifest for MD). Returns the
        absolute path to the primary protocol file.
        """
        ...
