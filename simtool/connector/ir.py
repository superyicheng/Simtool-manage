"""The Intermediate Representation (IR) — framework-agnostic scientific model.

The IR is the shared contract between (a) the meta-model, which populates
it from reconciled literature evidence, and (b) the connector, which lowers
it to framework-specific code via a FrameworkPlugin. It is intentionally
typed and closed-world for the kinds it understands, with a ``CustomProcess``
escape hatch — plugins MAY reject custom processes during ``validate_ir``.

Design priorities (in order):
  1. **Framework-agnostic**: no iDynoMiCS, LAMMPS, or NetLogo concepts leak
     into this schema. If a concept only makes sense in one framework, it
     belongs in that framework's plugin, not here.
  2. **Typed kinds over free expressions**: a ``MonodGrowthProcess`` is a
     first-class type, not a string expression the plugin has to parse. The
     plugin lowers the typed process into the framework's native idiom.
  3. **Parameters carry provenance**: every ``ParameterBinding`` can point
     back to supporting DOIs. The IR is serializable on its own; full
     ``ParameterRecord`` evidence lives in the meta-model and is resolved
     lazily at lowering time if needed.
  4. **Units are canonical**: every ``ParameterBinding`` declares its
     canonical unit (matching ``simtool.schema.idynomics_vocab``). Lowering
     applies framework-native unit conversion — this module does not.

First-plugin caveat: the user has explicitly budgeted time to refactor this
IR after the first plugin (iDynoMiCS 2) ships and before the second
framework is added. Expect kinds and field names to change.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Parameter bindings
# ---------------------------------------------------------------------------


class Distribution(BaseModel):
    """Distribution over a parameter value.

    When the meta-model has reconciled multiple literature values, the result
    is a distribution (often with an informative ``shape``). When only a
    single consensus value is available, a point estimate on the enclosing
    ``ParameterBinding`` is more appropriate than ``shape="point"`` here.
    """

    shape: Literal["normal", "lognormal", "uniform", "triangular", "empirical"]
    params: dict[str, float] = Field(
        default_factory=dict,
        description="Shape-specific params. normal: mean, stddev; "
        "lognormal: mu, sigma (of log); uniform: low, high; "
        "triangular: low, mode, high.",
    )
    samples: Optional[list[float]] = Field(
        default=None,
        description="For empirical distributions: the supporting values "
        "(already in canonical units).",
    )

    @model_validator(mode="after")
    def _check_shape_params(self) -> "Distribution":
        required_by_shape = {
            "normal": {"mean", "stddev"},
            "lognormal": {"mu", "sigma"},
            "uniform": {"low", "high"},
            "triangular": {"low", "mode", "high"},
        }
        if self.shape == "empirical":
            if not self.samples:
                raise ValueError(
                    "empirical distribution requires non-empty samples"
                )
            return self
        required = required_by_shape.get(self.shape, set())
        missing = required - set(self.params)
        if missing:
            raise ValueError(
                f"distribution shape '{self.shape}' requires params "
                f"{sorted(required)}; missing {sorted(missing)}"
            )
        # uniform/triangular sanity: low < high (and mode in between)
        if self.shape in {"uniform", "triangular"}:
            low, high = self.params["low"], self.params["high"]
            if not low < high:
                raise ValueError(
                    f"{self.shape} distribution requires low < high "
                    f"(got low={low}, high={high})"
                )
            if self.shape == "triangular":
                mode = self.params["mode"]
                if not low <= mode <= high:
                    raise ValueError(
                        f"triangular distribution requires low <= mode <= high "
                        f"(got low={low}, mode={mode}, high={high})"
                    )
        if self.shape in {"normal", "lognormal"}:
            sigma_key = "stddev" if self.shape == "normal" else "sigma"
            if self.params[sigma_key] <= 0.0:
                raise ValueError(
                    f"{self.shape} distribution requires {sigma_key} > 0 "
                    f"(got {self.params[sigma_key]})"
                )
        return self


class ParameterBinding(BaseModel):
    """Binds a canonical parameter key to a concrete value or distribution.

    Exactly one of ``point_estimate`` and ``distribution`` should be set.
    ``parameter_id`` matches a key from ``simtool.schema.idynomics_vocab.VOCAB``
    (shared with the meta-model). ``context_keys`` disambiguates per-context
    bindings — e.g. ``{"species": "AOB", "substrate": "ammonium"}`` lets the
    meta-model route to the AOB-on-ammonium K_s record, not NOB-on-nitrite.
    """

    parameter_id: str = Field(
        description="Canonical parameter key (e.g. 'mu_max', 'K_s'). Matches "
        "simtool.schema.idynomics_vocab.VOCAB."
    )
    canonical_unit: str = Field(
        description="Canonical unit matching the vocab entry for parameter_id. "
        "Held on the binding so the lowering path never has to look it up."
    )
    context_keys: dict[str, str] = Field(
        default_factory=dict,
        description="Context disambiguators consumed by the meta-model "
        "(species, substrate, redox_regime, ...).",
    )
    point_estimate: Optional[float] = None
    distribution: Optional[Distribution] = None
    provenance_dois: list[str] = Field(
        default_factory=list,
        description="DOIs of supporting ParameterRecord(s). The full "
        "evidence graph stays in the meta-model; this is enough to resolve.",
    )
    source_note: str = Field(
        default="",
        description="Human-readable source note (e.g. "
        "'reconciled from 4 chemostat studies, 20-25C AOB').",
    )

    @model_validator(mode="after")
    def _exactly_one_value_spec(self) -> "ParameterBinding":
        has_point = self.point_estimate is not None
        has_dist = self.distribution is not None
        if has_point and has_dist:
            raise ValueError(
                f"ParameterBinding('{self.parameter_id}'): cannot have both "
                "point_estimate and distribution — choose one"
            )
        if not has_point and not has_dist:
            raise ValueError(
                f"ParameterBinding('{self.parameter_id}'): requires exactly "
                "one of point_estimate or distribution"
            )
        if not self.canonical_unit.strip():
            raise ValueError(
                f"ParameterBinding('{self.parameter_id}'): canonical_unit "
                "must be non-empty"
            )
        return self


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------


class AgentPopulation(BaseModel):
    """A discrete-agent population (cells, particles, organisms).

    Used whenever the formalism represents individuals rather than a
    continuum field. Morphology is a hint to the plugin — frameworks that
    don't distinguish morphologies can ignore it.
    """

    kind: Literal["agent_population"] = "agent_population"
    id: str
    name: str
    morphology: Literal["coccus", "rod", "filament", "point", "custom"] = "coccus"
    parameters: dict[str, ParameterBinding] = Field(
        default_factory=dict,
        description="Agent-intrinsic parameters (cell_density, "
        "cell_division_mass, cell_initial_mass, ...).",
    )
    notes: str = ""


class Solute(BaseModel):
    """A continuum chemical field (dissolved species).

    Transport and reaction of the solute are described by ``Process``
    entries, not on the ``Solute`` itself — this keeps dynamics separate
    from identity.
    """

    kind: Literal["solute"] = "solute"
    id: str
    name: str
    chemical_formula: Optional[str] = None
    notes: str = ""


class SpatialDomain(BaseModel):
    """The spatial extent over which entities exist.

    ``extent_um`` is always length-per-axis in micrometres (canonical),
    regardless of ``dimensionality``. Resolution is a HINT — the plugin
    may adjust it based on numerical stability for the chosen formalism.
    """

    kind: Literal["spatial_domain"] = "spatial_domain"
    id: str
    dimensionality: Literal[1, 2, 3]
    extent_um: tuple[float, ...] = Field(
        description="Length per axis in micrometres. Length must equal "
        "dimensionality."
    )
    resolution_hint_um: Optional[float] = None
    periodic_axes: list[int] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def _check_extent_and_periodic(self) -> "SpatialDomain":
        if len(self.extent_um) != self.dimensionality:
            raise ValueError(
                f"SpatialDomain('{self.id}'): extent_um length "
                f"({len(self.extent_um)}) must equal dimensionality "
                f"({self.dimensionality})"
            )
        for ext in self.extent_um:
            if ext <= 0:
                raise ValueError(
                    f"SpatialDomain('{self.id}'): extent_um entries must be "
                    f"positive (got {ext})"
                )
        for ax in self.periodic_axes:
            if not 0 <= ax < self.dimensionality:
                raise ValueError(
                    f"SpatialDomain('{self.id}'): periodic_axes entry {ax} "
                    f"out of range for dimensionality {self.dimensionality}"
                )
        if len(set(self.periodic_axes)) != len(self.periodic_axes):
            raise ValueError(
                f"SpatialDomain('{self.id}'): periodic_axes has duplicates"
            )
        if self.resolution_hint_um is not None and self.resolution_hint_um <= 0:
            raise ValueError(
                f"SpatialDomain('{self.id}'): resolution_hint_um must be "
                f"positive (got {self.resolution_hint_um})"
            )
        return self


class Surface(BaseModel):
    """A boundary surface of the spatial domain.

    Referenced by BoundaryCondition.surface. Position is ``low`` (axis
    minimum) or ``high`` (axis maximum).
    """

    kind: Literal["surface"] = "surface"
    id: str
    name: str
    axis: int
    position: Literal["low", "high"]
    notes: str = ""


Entity = Annotated[
    Union[AgentPopulation, Solute, SpatialDomain, Surface],
    Field(discriminator="kind"),
]


# ---------------------------------------------------------------------------
# Processes (dynamics)
# ---------------------------------------------------------------------------


class MonodGrowthProcess(BaseModel):
    """Growth of an agent population via Monod kinetics on one or more
    limiting substrates.

    The multiplicative Monod form is assumed:
        mu = mu_max * prod_i ( S_i / (K_s_i + S_i) )
    Stoichiometry is expressed by ``consumed_solutes`` and ``produced_solutes``
    with yield parameters; inhibition terms go through ``inhibitors``.
    """

    kind: Literal["monod_growth"] = "monod_growth"
    id: str
    growing_entity: str
    consumed_solutes: list[str] = Field(
        description="Solute IDs consumed. Each should have a K_s_<solute_id> "
        "and Y_XS_<solute_id> binding in ``parameters``."
    )
    produced_solutes: list[str] = Field(
        default_factory=list,
        description="Solute IDs produced. Stoichiometry derived from yields.",
    )
    inhibitors: list[str] = Field(
        default_factory=list,
        description="Solute IDs acting as non-competitive inhibitors. Each "
        "should have a K_i_<solute_id> binding in ``parameters``.",
    )
    parameters: dict[str, ParameterBinding] = Field(
        default_factory=dict,
        description="Expected keys: 'mu_max', 'K_s_<solute>', 'Y_XS_<solute>', "
        "optionally 'K_i_<solute>'.",
    )
    notes: str = ""


class FirstOrderDecayProcess(BaseModel):
    """First-order biomass decay / endogenous respiration of an agent
    population."""

    kind: Literal["first_order_decay"] = "first_order_decay"
    id: str
    decaying_entity: str
    parameters: dict[str, ParameterBinding] = Field(
        default_factory=dict,
        description="Expected keys: 'b' (decay rate, 1/day).",
    )
    notes: str = ""


class MaintenanceProcess(BaseModel):
    """Substrate consumption by an agent population for maintenance
    (non-growth-associated)."""

    kind: Literal["maintenance"] = "maintenance"
    id: str
    entity: str
    consumed_solutes: list[str]
    parameters: dict[str, ParameterBinding] = Field(
        default_factory=dict,
        description="Expected keys: 'm_s_<solute>' "
        "(maintenance coefficient, g_substrate / (g_biomass * day)).",
    )
    notes: str = ""


class DiffusionProcess(BaseModel):
    """Fickian diffusion of a solute, optionally with region-dependent
    diffusivity (e.g. bulk liquid vs biofilm matrix)."""

    kind: Literal["diffusion"] = "diffusion"
    id: str
    solute: str
    regions: list[str] = Field(
        default_factory=lambda: ["bulk_liquid"],
        description="Named regions with potentially distinct diffusivities. "
        "Typical: ['bulk_liquid', 'biofilm']. Plugin interprets region names.",
    )
    parameters: dict[str, ParameterBinding] = Field(
        default_factory=dict,
        description="Expected keys: 'D_<region>' (e.g. 'D_bulk_liquid', "
        "'D_biofilm').",
    )
    notes: str = ""


class EPSProductionProcess(BaseModel):
    """EPS (extracellular polymeric substance) production by an agent
    population, coupled to growth or substrate consumption."""

    kind: Literal["eps_production"] = "eps_production"
    id: str
    producing_entity: str
    coupled_to: Literal["growth", "substrate_consumption"] = "growth"
    parameters: dict[str, ParameterBinding] = Field(
        default_factory=dict,
        description="Expected keys: 'Y_EPS' (yield).",
    )
    notes: str = ""


class CustomProcess(BaseModel):
    """Escape hatch for processes not yet canonicalized.

    Plugins MAY reject a ``CustomProcess`` during ``validate_ir`` if they
    cannot express it. The ``expression`` field is a free-text mathematical
    expression whose interpretation is plugin-dependent.
    """

    kind: Literal["custom"] = "custom"
    id: str
    description: str
    actors: list[str] = Field(
        description="Entity IDs the process acts on (consumes, produces, "
        "modifies)."
    )
    parameters: dict[str, ParameterBinding] = Field(default_factory=dict)
    expression: Optional[str] = None


Process = Annotated[
    Union[
        MonodGrowthProcess,
        FirstOrderDecayProcess,
        MaintenanceProcess,
        DiffusionProcess,
        EPSProductionProcess,
        CustomProcess,
    ],
    Field(discriminator="kind"),
]


# ---------------------------------------------------------------------------
# Boundary / initial conditions
# ---------------------------------------------------------------------------


class BoundaryCondition(BaseModel):
    """A boundary condition applied to an entity (typically a solute) on a
    given surface.

    Value interpretation:
      - dirichlet:    ``value`` is the prescribed concentration/state.
      - neumann_flux: ``value`` is the prescribed flux.
      - robin:        ``value`` is a mass-transfer coefficient; the bulk
                      value is carried in ``robin_bulk_value``.
      - no_flux:      ``value`` is ignored.
      - periodic:     paired with ``paired_surface``; ``value`` ignored.
    """

    id: str
    target_entity: str
    surface: str
    kind: Literal["dirichlet", "neumann_flux", "robin", "no_flux", "periodic"]
    value: Optional[ParameterBinding] = None
    robin_bulk_value: Optional[ParameterBinding] = None
    paired_surface: Optional[str] = None
    notes: str = ""


class InitialCondition(BaseModel):
    """Initial state for an entity.

    ``kind`` is framework-agnostic; parameters are kind-specific:
      - uniform:                     ``value`` required
      - zero:                        no parameters
      - gaussian_blob:               ``center_um``, ``sigma_um``, ``peak_value``
      - random_discrete_placement:   ``n_agents``, ``surface_id``, ``layer_thickness_um``
      - equilibrium:                 no parameters; plugin solves for a
                                     self-consistent equilibrium if supported
    """

    id: str
    target_entity: str
    kind: Literal[
        "uniform",
        "zero",
        "gaussian_blob",
        "random_discrete_placement",
        "equilibrium",
    ]
    parameters: dict[str, ParameterBinding] = Field(default_factory=dict)
    notes: str = ""


# ---------------------------------------------------------------------------
# Observables
# ---------------------------------------------------------------------------


class SamplingSpec(BaseModel):
    """When to record an observable. Exactly one of ``interval_s``,
    ``at_times_s``, or ``final_only`` should be set."""

    interval_s: Optional[float] = None
    at_times_s: Optional[list[float]] = None
    final_only: bool = False


class Observable(BaseModel):
    """A quantity of interest the user wants recorded.

    ``target`` is free-text — either an entity id ("AOB"), a composite
    ("biofilm_thickness"), or a derived quantity ("N_removal_efficiency").
    Composite targets are plugin-resolved.
    """

    id: str
    name: str
    kind: Literal[
        "scalar_time_series",
        "spatial_field_snapshot",
        "distribution",
        "flux_through_surface",
    ]
    target: str
    sampling: Optional[SamplingSpec] = None
    notes: str = ""


# ---------------------------------------------------------------------------
# Compute budget
# ---------------------------------------------------------------------------


class ComputeBudget(BaseModel):
    """User-declared computational constraints.

    ``time_horizon_s`` is the simulated time to reach (not wall time);
    ``wall_time_budget_s`` is the real-time budget for the run. Plugins
    should downscale resolution / parallelism to stay within the budget
    and surface that choice in the assumption ledger.
    """

    time_horizon_s: float
    wall_time_budget_s: Optional[float] = None
    memory_budget_gb: Optional[float] = None
    timestep_hint_s: Optional[float] = None
    parallel_hint: Optional[int] = None
    precision: Literal["single", "double"] = "double"

    @model_validator(mode="after")
    def _check_positive(self) -> "ComputeBudget":
        if self.time_horizon_s <= 0:
            raise ValueError(
                f"ComputeBudget.time_horizon_s must be positive "
                f"(got {self.time_horizon_s})"
            )
        for fname in ("wall_time_budget_s", "memory_budget_gb", "timestep_hint_s"):
            v = getattr(self, fname)
            if v is not None and v <= 0:
                raise ValueError(
                    f"ComputeBudget.{fname} must be positive if set (got {v})"
                )
        if self.parallel_hint is not None and self.parallel_hint < 1:
            raise ValueError(
                f"ComputeBudget.parallel_hint must be >= 1 if set "
                f"(got {self.parallel_hint})"
            )
        return self


# ---------------------------------------------------------------------------
# Assumption hints
# ---------------------------------------------------------------------------


class AssumptionHint(BaseModel):
    """Author-declared assumption intrinsic to this IR instance.

    These travel with the IR; they are NOT the runtime assumption ledger.
    The ledger (see ``assumptions.py``) is produced at lowering time and
    captures additional implicit assumptions the plugin makes. Hints here
    are things the user or meta-model stated up front (e.g. "temperature
    held at 20 C", "well-mixed bulk liquid").
    """

    id: str
    description: str
    justification: str = ""
    alternatives: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------


IR_SCHEMA_VERSION = "0.1.0"


class ScientificModel(BaseModel):
    """The IR root — a canonical, typed, framework-agnostic scientific model.

    This object is the single shared artifact between the meta-model and the
    connector. It is serializable (JSON/YAML) and round-trippable.
    """

    id: str
    title: str
    domain: str = Field(
        description="Free-text scientific domain "
        "(e.g. 'microbial_biofilm', 'granular_physics')."
    )
    formalism: Literal[
        "agent_based",
        "continuum_pde",
        "ode",
        "molecular_dynamics",
        "discrete_event",
        "lattice_boltzmann",
        "network_ode",
        "hybrid",
    ]
    entities: list[Entity]
    processes: list[Process]
    boundary_conditions: list[BoundaryCondition] = Field(default_factory=list)
    initial_conditions: list[InitialCondition] = Field(default_factory=list)
    observables: list[Observable] = Field(default_factory=list)
    compute: ComputeBudget
    assumption_hints: list[AssumptionHint] = Field(default_factory=list)
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Tool/version/snapshot IDs: ir_schema_version, "
        "meta_model_snapshot_id, simtool_version, ...",
    )
    ir_schema_version: str = IR_SCHEMA_VERSION

    def entity_ids(self) -> set[str]:
        return {e.id for e in self.entities}

    def surface_ids(self) -> set[str]:
        return {e.id for e in self.entities if isinstance(e, Surface)}

    @model_validator(mode="after")
    def _check_referential_integrity(self) -> "ScientificModel":
        _check_unique_ids(self.entities, "entity")
        _check_unique_ids(self.processes, "process")
        _check_unique_ids(self.boundary_conditions, "boundary condition")
        _check_unique_ids(self.initial_conditions, "initial condition")
        _check_unique_ids(self.observables, "observable")
        _check_unique_ids(self.assumption_hints, "assumption hint")

        ids = self.entity_ids()
        surfaces = self.surface_ids()

        # Surface axes must fit within some declared spatial domain.
        domains = [e for e in self.entities if isinstance(e, SpatialDomain)]
        if domains:
            max_dim = max(d.dimensionality for d in domains)
            for e in self.entities:
                if isinstance(e, Surface) and not 0 <= e.axis < max_dim:
                    raise ValueError(
                        f"Surface('{e.id}'): axis {e.axis} out of range for "
                        f"max declared domain dimensionality {max_dim}"
                    )

        for bc in self.boundary_conditions:
            if bc.target_entity not in ids:
                raise ValueError(
                    f"BoundaryCondition('{bc.id}'): target_entity "
                    f"'{bc.target_entity}' not found"
                )
            if bc.surface not in surfaces:
                raise ValueError(
                    f"BoundaryCondition('{bc.id}'): surface "
                    f"'{bc.surface}' is not a Surface entity"
                )
            if bc.paired_surface is not None and bc.paired_surface not in surfaces:
                raise ValueError(
                    f"BoundaryCondition('{bc.id}'): paired_surface "
                    f"'{bc.paired_surface}' is not a Surface entity"
                )
            if bc.kind == "periodic" and bc.paired_surface is None:
                raise ValueError(
                    f"BoundaryCondition('{bc.id}'): periodic kind requires "
                    "paired_surface"
                )
            if bc.kind == "robin" and bc.robin_bulk_value is None:
                raise ValueError(
                    f"BoundaryCondition('{bc.id}'): robin kind requires "
                    "robin_bulk_value"
                )

        for ic in self.initial_conditions:
            if ic.target_entity not in ids:
                raise ValueError(
                    f"InitialCondition('{ic.id}'): target_entity "
                    f"'{ic.target_entity}' not found"
                )

        for p in self.processes:
            _check_process_refs(p, ids)

        return self


# ---------------------------------------------------------------------------
# Helpers (module private)
# ---------------------------------------------------------------------------


def _check_unique_ids(items: list, kind: str) -> None:
    seen_ids = [getattr(i, "id") for i in items]
    if len(set(seen_ids)) != len(seen_ids):
        dupes = sorted({x for x in seen_ids if seen_ids.count(x) > 1})
        raise ValueError(f"duplicate {kind} ids: {dupes}")


def _check_process_refs(p, known_ids: set[str]) -> None:
    def _check(attr_value: str, label: str) -> None:
        if attr_value not in known_ids:
            raise ValueError(
                f"Process('{p.id}'): {label} '{attr_value}' not found"
            )

    if isinstance(p, MonodGrowthProcess):
        _check(p.growing_entity, "growing_entity")
        for s in p.consumed_solutes:
            _check(s, "consumed_solute")
        for s in p.produced_solutes:
            _check(s, "produced_solute")
        for s in p.inhibitors:
            _check(s, "inhibitor")
    elif isinstance(p, FirstOrderDecayProcess):
        _check(p.decaying_entity, "decaying_entity")
    elif isinstance(p, MaintenanceProcess):
        _check(p.entity, "entity")
        for s in p.consumed_solutes:
            _check(s, "consumed_solute")
    elif isinstance(p, DiffusionProcess):
        _check(p.solute, "solute")
    elif isinstance(p, EPSProductionProcess):
        _check(p.producing_entity, "producing_entity")
    elif isinstance(p, CustomProcess):
        for a in p.actors:
            _check(a, "actor")
