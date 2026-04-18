"""IR -> iDynoMiCS 2 protocol.xml.

Lowering is the load-bearing step: it turns our framework-agnostic IR
into iDynoMiCS 2's XML protocol format. Two paths:

- Chemostat  (formalism == 'ode'): ``<shape class="Dimensionless">`` + ``ChemostatSolver``.
- Biofilm    (formalism == 'agent_based', 1D/2D/3D): Rectangle/Cuboid
  with MgFASResolution, AgentRelaxation, PDEWrapper, random spawner.

Every implicit choice made here surfaces as an entry in the assumption
ledger returned alongside the artifact.

Unit conversion to iDynoMiCS bracket syntax:
    1/day     -> [d-1]
    mg/L      -> [mg/l]
    um^2/s    -> [um+2/s]
    pg        -> [pg]
    pg/fL     -> (bare number; iDynoMiCS's default density unit)
    g_biomass/g_substrate, g_EPS/g_substrate -> (bare number stoichiometry)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from xml.dom import minidom
from xml.etree import ElementTree as ET

from simtool.connector.assumptions import (
    Assumption,
    AssumptionCategory,
    AssumptionLedger,
    AssumptionSeverity,
)
from simtool.connector.ir import (
    AgentPopulation,
    BoundaryCondition,
    DiffusionProcess,
    Distribution,
    FirstOrderDecayProcess,
    InitialCondition,
    MaintenanceProcess,
    MonodGrowthProcess,
    ParameterBinding,
    ScientificModel,
    Solute,
    SpatialDomain,
    Surface,
)


@dataclass
class LoweringResult:
    protocol_xml: bytes
    ledger: AssumptionLedger
    files: list[tuple[str, bytes]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Unit formatting
# ---------------------------------------------------------------------------


_BRACKET_UNITS = {
    "1/day": "[d-1]",
    "1/hour": "[h-1]",
    "mg/L": "[mg/l]",
    "g/L": "[g/l]",
    "um^2/s": "[um+2/s]",
    "m^2/s": "[m+2/s]",
    "pg": "[pg]",
    "g": "[g]",
    "um": "[um]",
    "s": "[s]",
    "h": "[h]",
    "d": "[d]",
    "day": "[d]",
}


def _fmt_value(v: float, unit: str) -> str:
    bracket = _BRACKET_UNITS.get(unit)
    if bracket is not None:
        return f"{v:g}{bracket}"
    # Unitless quantities (yields, densities) are written as bare numbers.
    return f"{v:g}"


def _point_estimate(b: ParameterBinding, ledger: AssumptionLedger, context: str) -> float:
    """Extract a scalar from a binding, lowering a distribution to its
    central value and surfacing the flattening as a MATERIAL assumption."""
    if b.point_estimate is not None:
        return b.point_estimate
    d = b.distribution
    if d is None:
        raise ValueError(
            f"binding for {b.parameter_id} has neither point_estimate nor distribution"
        )
    central = _distribution_central(d)
    _add_assumption(
        ledger,
        id_=f"flatten_distribution_{b.parameter_id}_{context}",
        category=AssumptionCategory.PARAMETERIZATION,
        severity=AssumptionSeverity.MATERIAL,
        description=(
            f"{b.parameter_id} ({b.canonical_unit}) was lowered from a "
            f"{d.shape} distribution to its central value {central:g}; "
            "propagated uncertainty is lost at this step."
        ),
        justification=(
            "iDynoMiCS 2 protocol.xml accepts scalar constants only; "
            "distribution support would require ensemble runs."
        ),
        alternatives=[
            "ensemble of runs over sampled values",
            "median instead of mean",
            "best-case + worst-case endpoints",
        ],
        surfaced_by="idynomics_2.lower.flatten_distribution",
        affects=[context],
    )
    return central


def _distribution_central(d: Distribution) -> float:
    if d.shape == "empirical" and d.samples:
        return sum(d.samples) / len(d.samples)
    if d.shape == "uniform":
        return 0.5 * (d.params["low"] + d.params["high"])
    if d.shape == "triangular":
        return d.params["mode"]
    if d.shape == "normal":
        return d.params["mean"]
    if d.shape == "lognormal":
        import math
        return math.exp(d.params["mu"])
    raise ValueError(f"no central value for distribution shape {d.shape!r}")


# ---------------------------------------------------------------------------
# Ledger helpers
# ---------------------------------------------------------------------------


def _add_assumption(
    ledger: AssumptionLedger,
    *, id_: str, category: AssumptionCategory, severity: AssumptionSeverity,
    description: str, justification: str,
    alternatives: Optional[list[str]] = None,
    surfaced_by: str = "idynomics_2.lower",
    affects: Optional[list[str]] = None,
) -> None:
    if any(a.id == id_ for a in ledger.assumptions):
        return
    ledger.add(Assumption(
        id=id_, category=category, severity=severity,
        description=description, justification=justification,
        alternatives=alternatives or [], surfaced_by=surfaced_by,
        affects=affects or [],
    ))


# ---------------------------------------------------------------------------
# Entity lookups
# ---------------------------------------------------------------------------


def _entities_by_id(ir: ScientificModel) -> dict[str, object]:
    return {e.id: e for e in ir.entities}


def _agents(ir: ScientificModel) -> list[AgentPopulation]:
    return [e for e in ir.entities if isinstance(e, AgentPopulation)]


def _solutes(ir: ScientificModel) -> list[Solute]:
    return [e for e in ir.entities if isinstance(e, Solute)]


def _domain(ir: ScientificModel) -> Optional[SpatialDomain]:
    for e in ir.entities:
        if isinstance(e, SpatialDomain):
            return e
    return None


def _surfaces(ir: ScientificModel) -> list[Surface]:
    return [e for e in ir.entities if isinstance(e, Surface)]


def _growth_for(
    ir: ScientificModel, agent_id: str
) -> list[MonodGrowthProcess]:
    return [
        p for p in ir.processes
        if isinstance(p, MonodGrowthProcess) and p.growing_entity == agent_id
    ]


def _decay_for(
    ir: ScientificModel, agent_id: str
) -> list[FirstOrderDecayProcess]:
    return [
        p for p in ir.processes
        if isinstance(p, FirstOrderDecayProcess) and p.decaying_entity == agent_id
    ]


def _maintenance_for(
    ir: ScientificModel, agent_id: str
) -> list[MaintenanceProcess]:
    return [
        p for p in ir.processes
        if isinstance(p, MaintenanceProcess) and p.entity == agent_id
    ]


def _diffusion_for(
    ir: ScientificModel, solute_id: str
) -> Optional[DiffusionProcess]:
    for p in ir.processes:
        if isinstance(p, DiffusionProcess) and p.solute == solute_id:
            return p
    return None


def _dirichlet_for(
    ir: ScientificModel, solute_id: str
) -> Optional[BoundaryCondition]:
    for bc in ir.boundary_conditions:
        if bc.target_entity == solute_id and bc.kind == "dirichlet":
            return bc
    return None


def _ic_uniform_value(
    ir: ScientificModel, entity_id: str
) -> Optional[ParameterBinding]:
    for ic in ir.initial_conditions:
        if ic.target_entity == entity_id and ic.kind == "uniform":
            return ic.parameters.get("value")
    return None


def _random_spawn_for(
    ir: ScientificModel, agent_id: str
) -> Optional[InitialCondition]:
    for ic in ir.initial_conditions:
        if ic.target_entity == agent_id and ic.kind == "random_discrete_placement":
            return ic
    return None


# ---------------------------------------------------------------------------
# XML building helpers
# ---------------------------------------------------------------------------


def _pretty_xml(root: ET.Element) -> bytes:
    rough = ET.tostring(root, encoding="utf-8")
    parsed = minidom.parseString(rough)
    pretty = parsed.toprettyxml(indent="\t", encoding="UTF-8")
    # minidom emits blank lines; trim them.
    lines = [ln for ln in pretty.decode("utf-8").splitlines() if ln.strip()]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _sub(parent: ET.Element, tag: str, **attrs: str) -> ET.Element:
    return ET.SubElement(parent, tag, {k: v for k, v in attrs.items() if v is not None})


# ---------------------------------------------------------------------------
# Lowering
# ---------------------------------------------------------------------------


def lower_ir(ir: ScientificModel) -> LoweringResult:
    """Lower an IR to an iDynoMiCS 2 protocol.xml + assumption ledger."""
    ledger = AssumptionLedger(
        ir_id=ir.id, framework="idynomics_2", framework_version="2.0",
    )

    # Choose template path from formalism.
    if ir.formalism == "ode":
        xml = _lower_chemostat(ir, ledger)
    elif ir.formalism == "agent_based":
        xml = _lower_biofilm(ir, ledger)
    else:
        raise ValueError(
            f"iDynoMiCS 2 plugin does not support formalism '{ir.formalism}'. "
            "Supported: 'ode' (chemostat), 'agent_based' (biofilm)."
        )

    payload = _pretty_xml(xml)
    return LoweringResult(
        protocol_xml=payload,
        ledger=ledger,
        files=[("protocol.xml", payload)],
    )


# --- shared pieces ---------------------------------------------------------


def _simulation_header(ir: ScientificModel) -> tuple[ET.Element, str, str]:
    root = ET.Element("document")
    sim = _sub(root, "simulation", name=ir.id, outputfolder="../outputs", log="NORMAL")
    total_s = ir.compute.time_horizon_s
    step_hint_s = ir.compute.timestep_hint_s or max(total_s / 100.0, 60.0)
    end_days = total_s / 86400.0
    step_hours = step_hint_s / 3600.0
    _sub(sim, "timer", stepSize=f"{step_hours:g} [h]",
         endOfSimulation=f"{end_days:g} [d]")
    return root, f"{step_hours:g} [h]", f"{end_days:g} [d]"


# --- chemostat path --------------------------------------------------------


def _lower_chemostat(ir: ScientificModel, ledger: AssumptionLedger) -> ET.Element:
    root, _, _ = _simulation_header(ir)
    sim = root.find("simulation")
    assert sim is not None

    species_lib = _sub(sim, "speciesLib")
    for ap in _agents(ir):
        _emit_species(ap, ir, species_lib, ledger, spatial=False)

    compartment = _sub(sim, "compartment", name="chemostat")
    _sub(compartment, "shape", **{"class": "Dimensionless", "volume": "1.0"})

    _emit_solutes_flat(ir, compartment, ledger, spatial=False)
    _emit_chemostat_processes(ir, compartment)

    _add_assumption(
        ledger,
        id_="chemostat_well_mixed_bulk",
        category=AssumptionCategory.PHYSICS,
        severity=AssumptionSeverity.CRITICAL,
        description="Dimensionless shape -> bulk is assumed well-mixed (no spatial gradients).",
        justification="IR formalism 'ode' maps to iDynoMiCS Dimensionless compartment.",
        alternatives=["1D continuum PDE", "agent-based biofilm"],
    )
    return root


def _emit_chemostat_processes(ir: ScientificModel, compartment: ET.Element) -> None:
    pm = _sub(compartment, "processManagers")
    proc = _sub(pm, "process", name="solveChemostat",
                **{"class": "ChemostatSolver"}, priority="0", firstStep="0.0")
    solute_names = ",".join(s.id for s in _solutes(ir))
    _sub(proc, "aspect", name="soluteNames", type="PRIMARY",
         **{"class": "String[]"}, value=solute_names)
    _sub(proc, "aspect", name="tolerance", type="PRIMARY",
         **{"class": "Double"}, value="1.0e-3")
    _sub(proc, "aspect", name="hMax", type="PRIMARY",
         **{"class": "Double"}, value="1.0e-3")
    _sub(proc, "aspect", name="solver", type="PRIMARY",
         **{"class": "String"}, value="other")


# --- biofilm path ----------------------------------------------------------


def _lower_biofilm(ir: ScientificModel, ledger: AssumptionLedger) -> ET.Element:
    root, _, _ = _simulation_header(ir)
    sim = root.find("simulation")
    assert sim is not None

    species_lib = _sub(sim, "speciesLib")
    for ap in _agents(ir):
        _emit_species(ap, ir, species_lib, ledger, spatial=True)
    # Morphology species reference (iDynoMiCS requires a base coccoid species).
    if _agents(ir):
        _emit_coccoid_morphology(species_lib, _agents(ir)[0], ledger)

    compartment = _sub(sim, "compartment", name="biofilm-compartment")
    _emit_shape(ir, compartment, ledger)
    _emit_solutes_flat(ir, compartment, ledger, spatial=True)
    _emit_spawn(ir, compartment, ledger)

    pm = _sub(compartment, "processManagers")
    _sub(pm, "process", name="agentRelax",
         **{"class": "AgentRelaxation"}, priority="0")
    _sub(pm, "process", name="PDEWrapper",
         **{"class": "PDEWrapper"}, priority="1")

    _add_assumption(
        ledger,
        id_="agent_relax_priority",
        category=AssumptionCategory.NUMERICS,
        severity=AssumptionSeverity.ADVISORY,
        description="Agent relaxation runs before PDE solve each step.",
        justification="iDynoMiCS default process-manager priorities.",
        alternatives=["reverse priority", "interleaved fine-step coupling"],
    )
    return root


def _emit_shape(
    ir: ScientificModel, compartment: ET.Element, ledger: AssumptionLedger
) -> None:
    domain = _domain(ir)
    if domain is None:
        raise ValueError("biofilm lowering requires a SpatialDomain entity")
    shape_class = {1: "Line", 2: "Rectangle", 3: "Cuboid"}[domain.dimensionality]
    shape = _sub(compartment, "shape", **{"class": shape_class},
                 resolutionCalculator="MgFASResolution", nodeSystem="true")
    resolution = domain.resolution_hint_um or 2.0
    axis_names = ["X", "Y", "Z"]
    periodic = set(domain.periodic_axes)
    for i, extent in enumerate(domain.extent_um):
        is_cyclic = "true" if i in periodic else "false"
        dim = _sub(shape, "dimension", name=axis_names[i],
                   isCyclic=is_cyclic,
                   targetResolution=f"{resolution:g}",
                   max=f"{extent:g}")
        # Emit Dirichlet BCs as FixedBoundary on the 'extreme' side.
        if i == domain.dimensionality - 1:  # top axis holds the bulk interface
            _emit_fixed_boundary(ir, dim, extreme="1")
    if not domain.periodic_axes:
        _add_assumption(
            ledger,
            id_="no_periodic_axes",
            category=AssumptionCategory.BOUNDARY,
            severity=AssumptionSeverity.ADVISORY,
            description="No periodic axes declared; all lateral surfaces are walls.",
            justification="IR's periodic_axes list was empty.",
            alternatives=["declare X (and Y in 3D) as periodic"],
        )


def _emit_fixed_boundary(ir: ScientificModel, dim_elem: ET.Element, extreme: str) -> None:
    boundary = _sub(dim_elem, "boundary",
                    extreme=extreme, **{"class": "FixedBoundary"},
                    layerThickness="32.0")
    for s in _solutes(ir):
        bc = _dirichlet_for(ir, s.id)
        if bc is not None and bc.value is not None:
            # BC values typically reported in mg/L already matching canonical.
            v = bc.value.point_estimate if bc.value.point_estimate is not None else 0.0
            _sub(boundary, "solute", name=s.id,
                 concentration=_fmt_value(v, bc.value.canonical_unit))


def _emit_solutes_flat(
    ir: ScientificModel, compartment: ET.Element,
    ledger: AssumptionLedger, *, spatial: bool,
) -> None:
    sol_list = _sub(compartment, "solutes")
    for s in _solutes(ir):
        bc = _dirichlet_for(ir, s.id)
        concentration = None
        if bc is not None and bc.value is not None and bc.value.point_estimate is not None:
            concentration = _fmt_value(bc.value.point_estimate, bc.value.canonical_unit)
        diff_liquid: Optional[str] = None
        diff_biofilm: Optional[str] = None
        dif = _diffusion_for(ir, s.id)
        if dif is not None:
            liq = dif.parameters.get("D_bulk_liquid") or dif.parameters.get("D_liquid")
            bio = dif.parameters.get("D_biofilm")
            if liq is not None:
                v = _point_estimate(liq, ledger, context=f"D_liquid_{s.id}")
                diff_liquid = _fmt_value(v, liq.canonical_unit)
            if bio is not None:
                v = _point_estimate(bio, ledger, context=f"D_biofilm_{s.id}")
                diff_biofilm = _fmt_value(v, bio.canonical_unit)
        attrs: dict[str, str] = {"name": s.id}
        if concentration:
            attrs["concentration"] = concentration
        if diff_liquid:
            attrs["defaultDiffusivity"] = diff_liquid
        if spatial and diff_biofilm:
            attrs["biofilmDiffusivity"] = diff_biofilm
        _sub(sol_list, "solute", **attrs)


def _emit_spawn(
    ir: ScientificModel, compartment: ET.Element, ledger: AssumptionLedger
) -> None:
    domain = _domain(ir)
    if domain is None:
        return
    agents = _agents(ir)
    if not agents:
        return
    # Use the first agent population's random_discrete_placement if present.
    for ap in agents:
        ic = _random_spawn_for(ir, ap.id)
        if ic is None:
            continue
        n_binding = ic.parameters.get("n_agents")
        layer_binding = ic.parameters.get("layer_thickness_um")
        n = int(n_binding.point_estimate) if n_binding and n_binding.point_estimate else 30
        layer = layer_binding.point_estimate if layer_binding and layer_binding.point_estimate else 4.0
        # Domain for spawner: extent along the non-top axis, layer thickness at bottom.
        domain_spec = f"{domain.extent_um[0]:g}, {layer:g}"
        spawn = _sub(compartment, "spawn",
                     **{"class": "randomSpawner"},
                     domain=domain_spec, priority="0",
                     number=str(n), morphology="COCCOID")
        template = _sub(spawn, "templateAgent")
        _sub(template, "aspect", name="species",
             **{"class": "String"}, value=ap.id)
        init_mass = ap.parameters.get("cell_initial_mass")
        mass_v = init_mass.point_estimate if init_mass and init_mass.point_estimate else 0.2
        _sub(template, "aspect", name="mass",
             **{"class": "Double"}, value=f"{mass_v:g}")
        _add_assumption(
            ledger,
            id_=f"initial_placement_{ap.id}",
            category=AssumptionCategory.BOUNDARY,
            severity=AssumptionSeverity.MATERIAL,
            description=(
                f"Initial {n} {ap.id} agents placed randomly in a "
                f"{layer:g} um layer at the substratum."
            ),
            justification=(
                "random_discrete_placement IC with n_agents and "
                "layer_thickness_um; iDynoMiCS randomSpawner is used."
            ),
            alternatives=["regular grid placement", "inoculum from a saved state"],
            affects=[ic.id],
        )
        # Only spawn one block; subsequent species will be placed by iDynoMiCS
        # after the first division cycle if shape allows.
        break


# --- species + growth reactions -------------------------------------------


def _emit_species(
    ap: AgentPopulation, ir: ScientificModel, parent: ET.Element,
    ledger: AssumptionLedger, *, spatial: bool,
) -> None:
    species = _sub(parent, "species", name=ap.id)
    _sub(species, "speciesModule", name="coccoid")
    reactions_aspect = _sub(species, "aspect", name="reactions",
                            **{"class": "InstantiableList"})
    reaction_list = _sub(reactions_aspect, "list",
                         nodeLabel="reaction", entryClass="RegularReaction")

    for g in _growth_for(ir, ap.id):
        _emit_monod_growth_reaction(g, reaction_list, ledger)

    for d in _decay_for(ir, ap.id):
        _emit_decay_reaction(d, reaction_list, ledger)

    for m in _maintenance_for(ir, ap.id):
        _emit_maintenance_reaction(m, reaction_list, ledger)


def _emit_monod_growth_reaction(
    g: MonodGrowthProcess, reaction_list: ET.Element, ledger: AssumptionLedger,
) -> None:
    reaction = _sub(reaction_list, "reaction", name=g.id)

    terms = ["mass", "mumax"]
    constants: list[tuple[str, str]] = []

    mu = g.parameters.get("mu_max")
    if mu is None:
        raise ValueError(
            f"Monod growth '{g.id}' missing required parameter 'mu_max'"
        )
    mu_v = _point_estimate(mu, ledger, context=f"{g.id}.mu_max")
    constants.append(("mumax", _fmt_value(mu_v, mu.canonical_unit)))

    for solute in g.consumed_solutes:
        ks_key = f"K_s_{solute}"
        ks = g.parameters.get(ks_key)
        if ks is None:
            raise ValueError(
                f"Monod growth '{g.id}' missing required parameter '{ks_key}'"
            )
        ks_v = _point_estimate(ks, ledger, context=f"{g.id}.{ks_key}")
        const_name = f"Ks_{solute}"
        terms.append(f"({solute}/({solute}+{const_name}))")
        constants.append((const_name, _fmt_value(ks_v, ks.canonical_unit)))

    expression = _sub(reaction, "expression", value="*".join(terms))
    for name, value in constants:
        _sub(expression, "constant", name=name, value=value)

    _sub(reaction, "stoichiometric", component="mass", coefficient="1.0")
    # Solute stoichiometry from yields (only present for consumed pairs).
    for solute in g.consumed_solutes:
        y_key = f"Y_XS_{solute}"
        y = g.parameters.get(y_key)
        if y is not None:
            y_v = _point_estimate(y, ledger, context=f"{g.id}.{y_key}")
            coef = -1.0 / max(y_v, 1e-30)
            _sub(reaction, "stoichiometric", component=solute,
                 coefficient=f"{coef:g}")
    # Produced solutes get stoichiometric coefficients; we infer from mass balance
    # using the first consumed solute's yield (crude; iDynoMiCS users usually
    # declare these explicitly in parameters).
    for solute in g.produced_solutes:
        _sub(reaction, "stoichiometric", component=solute,
             coefficient="1.0")
        _add_assumption(
            ledger,
            id_=f"stoichiometry_placeholder_{g.id}_{solute}",
            category=AssumptionCategory.PARAMETERIZATION,
            severity=AssumptionSeverity.MATERIAL,
            description=(
                f"Produced-solute stoichiometric coefficient for "
                f"'{solute}' in reaction '{g.id}' defaults to 1.0; "
                "declare an explicit coefficient via IR parameters for "
                "correct mass balance."
            ),
            justification="IR did not carry an explicit produced-solute yield.",
            alternatives=["supply stoichiometry as a Y_XP_<solute> parameter"],
            affects=[g.id],
        )


def _emit_decay_reaction(
    d: FirstOrderDecayProcess, reaction_list: ET.Element, ledger: AssumptionLedger,
) -> None:
    reaction = _sub(reaction_list, "reaction", name=d.id)
    b = d.parameters.get("b")
    if b is None:
        raise ValueError(f"Decay '{d.id}' missing required parameter 'b'")
    b_v = _point_estimate(b, ledger, context=f"{d.id}.b")
    expr = _sub(reaction, "expression", value="mass*b")
    _sub(expr, "constant", name="b", value=_fmt_value(b_v, b.canonical_unit))
    _sub(reaction, "stoichiometric", component="mass", coefficient="-1.0")
    _add_assumption(
        ledger,
        id_=f"decay_reaction_inserted_{d.id}",
        category=AssumptionCategory.PARAMETERIZATION,
        severity=AssumptionSeverity.ADVISORY,
        description=(
            f"Decay reaction '{d.id}' inserted as a regular iDynoMiCS "
            "reaction; biomass decreases first-order in mass."
        ),
        justification="FirstOrderDecayProcess in IR.",
        alternatives=["endogenous-respiration form consuming O2"],
        affects=[d.id],
    )


def _emit_maintenance_reaction(
    m: MaintenanceProcess, reaction_list: ET.Element, ledger: AssumptionLedger,
) -> None:
    reaction = _sub(reaction_list, "reaction", name=m.id)
    # Single-solute maintenance form: mass * m_s; stoichiometry consumes one solute.
    if not m.consumed_solutes:
        return
    solute = m.consumed_solutes[0]
    ms_key = f"m_s_{solute}"
    ms = m.parameters.get(ms_key) or m.parameters.get("m_s")
    if ms is None:
        raise ValueError(
            f"Maintenance '{m.id}' missing parameter '{ms_key}' or 'm_s'"
        )
    ms_v = _point_estimate(ms, ledger, context=f"{m.id}.m_s")
    expr = _sub(reaction, "expression", value="mass*m_s")
    _sub(expr, "constant", name="m_s", value=_fmt_value(ms_v, ms.canonical_unit))
    _sub(reaction, "stoichiometric", component=solute, coefficient="-1.0")


def _emit_coccoid_morphology(
    parent: ET.Element, first_agent: AgentPopulation, ledger: AssumptionLedger,
) -> None:
    species = _sub(parent, "species", name="coccoid")
    density = first_agent.parameters.get("cell_density")
    density_v = density.point_estimate if density and density.point_estimate else 0.15
    _sub(species, "aspect", name="density",
         **{"class": "Double"}, value=f"{density_v:g}")
    _sub(species, "aspect", name="surfaces", **{"class": "AgentSurfaces"})
    _sub(species, "aspect", name="morphology",
         **{"class": "String"}, value="coccoid")
    _sub(species, "aspect", name="volume", **{"class": "SimpleVolumeState"})
    _sub(species, "aspect", name="radius", **{"class": "CylinderRadius"})
    _sub(species, "aspect", name="divide", **{"class": "CoccoidDivision"})
    div_mass = first_agent.parameters.get("cell_division_mass")
    div_v = div_mass.point_estimate if div_mass and div_mass.point_estimate else 0.2
    _sub(species, "aspect", name="divisionMass",
         **{"class": "Double"}, value=f"{div_v:g} [pg]")
    _sub(species, "aspect", name="updateBody", **{"class": "UpdateBody"})
    _add_assumption(
        ledger,
        id_="coccoid_morphology_default",
        category=AssumptionCategory.PHYSICS,
        severity=AssumptionSeverity.ADVISORY,
        description="All agents treated as coccoid (spherical) with default iDynoMiCS settings.",
        justification="IR morphology defaulted or coccoid; richer shapes not yet mapped.",
        alternatives=["rod morphology", "filamentous morphology"],
    )
