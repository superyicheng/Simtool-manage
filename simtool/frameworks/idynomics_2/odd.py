"""Generate a compact ODD (Overview, Design concepts, Details) protocol
document for an iDynoMiCS 2 run.

The ODD protocol is the primary scientific artifact alongside the
numerical outputs. This implementation emits a minimal but
standards-conformant markdown document; richer templating (e.g. the
Grimm 2020 update) can slot in later without breaking the plugin
contract.
"""

from __future__ import annotations

from pathlib import Path

from simtool.connector.ir import (
    AgentPopulation,
    DiffusionProcess,
    FirstOrderDecayProcess,
    MaintenanceProcess,
    MonodGrowthProcess,
    ScientificModel,
    Solute,
)


def generate_odd(ir: ScientificModel, out_path: Path) -> Path:
    """Write the ODD protocol document. Returns the written path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sections: list[str] = [f"# ODD Protocol — {ir.title}", ""]
    sections.append(f"**IR id**: {ir.id}")
    sections.append(f"**formalism**: {ir.formalism}")
    sections.append(f"**domain**: {ir.domain}")
    sections.append(
        f"**simulated time horizon**: {ir.compute.time_horizon_s:g} s"
    )
    if ir.metadata:
        sections.append("")
        sections.append("## Metadata")
        for k, v in sorted(ir.metadata.items()):
            sections.append(f"- {k}: {v}")

    sections += ["", "## 1. Purpose"]
    sections.append(
        f"Simulate {ir.title} under the IR's declared conditions; reproduce "
        "literature-reported behavior of the system as captured by the "
        "associated meta-model."
    )

    sections += ["", "## 2. Entities, state variables, scales"]
    agents = [e for e in ir.entities if isinstance(e, AgentPopulation)]
    solutes = [e for e in ir.entities if isinstance(e, Solute)]
    if agents:
        sections.append("**Agent populations**:")
        for a in agents:
            sections.append(f"- `{a.id}` ({a.name}; morphology {a.morphology})")
    if solutes:
        sections.append("")
        sections.append("**Solutes**:")
        for s in solutes:
            sections.append(
                f"- `{s.id}` ({s.name}{', ' + s.chemical_formula if s.chemical_formula else ''})"
            )

    sections += ["", "## 3. Process overview and scheduling"]
    for p in ir.processes:
        sections.append(f"- `{p.kind}:{p.id}` — {_describe(p)}")

    sections += ["", "## 4. Design concepts"]
    sections.append(
        "Emergence: biomass distribution and solute gradients arise from "
        "local Monod kinetics and diffusion. Stochasticity: agent "
        "placement and division order are random. Observation: "
        + ", ".join(o.id for o in ir.observables) + "."
    )

    if ir.assumption_hints:
        sections += ["", "## 5. Initialization and assumptions"]
        for h in ir.assumption_hints:
            sections.append(f"- **{h.id}**: {h.description}")
            if h.justification:
                sections.append(f"  _why_: {h.justification}")

    sections += ["", "## 6. Input data"]
    sections.append("Parameter values are inherited from the meta-model's "
                    "reconciled distributions; user overrides (if any) "
                    "appear in the panel's override list.")

    out_path.write_text("\n".join(sections) + "\n", encoding="utf-8")
    return out_path


def _describe(p) -> str:
    if isinstance(p, MonodGrowthProcess):
        return (
            f"Monod growth of {p.growing_entity} on "
            + ", ".join(p.consumed_solutes)
            + (f" producing {', '.join(p.produced_solutes)}"
               if p.produced_solutes else "")
        )
    if isinstance(p, FirstOrderDecayProcess):
        return f"first-order decay of {p.decaying_entity}"
    if isinstance(p, MaintenanceProcess):
        return f"maintenance of {p.entity} on {', '.join(p.consumed_solutes)}"
    if isinstance(p, DiffusionProcess):
        return f"diffusion of {p.solute} in " + " / ".join(p.regions)
    return getattr(p, "description", p.kind)
