"""Simulation Framework Connector.

Infrastructure layer that lowers a framework-agnostic Intermediate
Representation (IR) into specific simulation frameworks (iDynoMiCS 2,
LAMMPS, NetLogo, ...), executes runs under a supervised harness, and
produces reproducibility artifacts.

Architectural principle: ONE IR shared with the meta-model. The meta-model
builds up IR-shaped knowledge from literature; the user specifies IR-shaped
requirements; the connector lowers IR to framework code. No second schema.

Module map:
    ir.py           — the IR schema (framework-agnostic scientific model)
    plugin.py       — common FrameworkPlugin Protocol; one implementation per
                      supported framework
    skill.py        — per-framework skill file (grammar + annotations + examples),
                      versioned against the framework version
    assumptions.py  — assumption ledger; every implicit choice made during
                      lowering is recorded here; user approves before run
    runs.py         — run harness artifact types: RunRecord, ProgressReport,
                      OutputBundle. Binds all artifacts to a run ID.

Integration note: the IR in ``ir.py`` is currently defined connector-side to
avoid collision with the concurrently-evolving ``simtool.schema`` package.
Once the meta-model's internal contract stabilizes, ``ir.py`` and
``simtool.schema`` will merge — the IR IS the meta-model's canonical shape,
not a sibling.
"""

from simtool.connector.ir import (
    AgentPopulation,
    AssumptionHint,
    BoundaryCondition,
    ComputeBudget,
    CustomProcess,
    DiffusionProcess,
    Distribution,
    EPSProductionProcess,
    FirstOrderDecayProcess,
    InitialCondition,
    MaintenanceProcess,
    MonodGrowthProcess,
    Observable,
    ParameterBinding,
    SamplingSpec,
    ScientificModel,
    Solute,
    SpatialDomain,
    Surface,
)

__all__ = [
    "AgentPopulation",
    "AssumptionHint",
    "BoundaryCondition",
    "ComputeBudget",
    "CustomProcess",
    "DiffusionProcess",
    "Distribution",
    "EPSProductionProcess",
    "FirstOrderDecayProcess",
    "InitialCondition",
    "MaintenanceProcess",
    "MonodGrowthProcess",
    "Observable",
    "ParameterBinding",
    "SamplingSpec",
    "ScientificModel",
    "Solute",
    "SpatialDomain",
    "Surface",
]
