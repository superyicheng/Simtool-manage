"""iDynoMiCS 2 parameter vocabulary.

Canonical units here match iDynoMiCS 2's native XML unit tags
(see protocol/template.xml and protocol/template_PARAMETERS.md in the
iDynoMiCS-2 distribution). Extraction reports values in whatever units
the source paper used; the units harmonization layer converts into the
canonical unit on this record before the meta-model stores it.

`idynomics_xml_path` is an XPath-like locator pointing at the attribute
that holds the value in a fully-populated iDynoMiCS 2 protocol.xml.
`{substrate}` is a placeholder filled at emit time from
ParameterRecord.context.substrate. `None` means the parameter cannot
be written by replacing a single attribute (e.g. decay/maintenance
require inserting a whole new <reaction> block — handled by the
emitter, not by a constant substitution).
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class VocabEntry(BaseModel):
    id: str
    description: str
    canonical_unit: str
    sanity_range: tuple[float, float]
    idynomics_xml_path: Optional[str] = None
    substrate_keyed: bool = False
    requires_reaction_insert: bool = False
    notes: Optional[str] = None


_CORE: list[VocabEntry] = [
    # --- Microbial growth kinetics (Monod) ---
    VocabEntry(
        id="mu_max",
        description="Maximum specific growth rate in the primary growth reaction.",
        canonical_unit="1/day",
        sanity_range=(1e-3, 50.0),
        idynomics_xml_path=(
            "species[@name='{species}']/aspect[@name='reactions']"
            "//reaction[@name='growth']/expression/constant[@name='mumax']/@value"
        ),
        notes="iDynoMiCS 2 uses [d-1]. AOB ~0.7-2.0 /d, NOB ~0.5-1.2 /d at 20-25C; "
        "heterotrophs 2-20 /d. Chemostat-derived values are typically lower than "
        "initial-rate values under the same conditions.",
    ),
    VocabEntry(
        id="K_s",
        description="Half-saturation (Monod) constant for the substrate denoted in StudyContext.substrate.",
        canonical_unit="mg/L",
        sanity_range=(1e-4, 1e3),
        idynomics_xml_path=(
            "species[@name='{species}']/aspect[@name='reactions']"
            "//reaction[@name='growth']/expression/constant[@name='Ks_{substrate}']/@value"
        ),
        substrate_keyed=True,
        notes="iDynoMiCS 2 uses [g/m+3] which is numerically mg/L. Interpretation "
        "depends on method: chemostat Ks is typically biased high vs intrinsic Ks. "
        "For O2, record StudyContext.substrate='oxygen'.",
    ),
    VocabEntry(
        id="Y_XS",
        description="Biomass yield on substrate (substrate denoted in StudyContext.substrate).",
        canonical_unit="g_biomass/g_substrate",
        sanity_range=(1e-3, 1.0),
        idynomics_xml_path=None,
        substrate_keyed=True,
        notes="iDynoMiCS 2 encodes this as a stoichiometric coefficient (-1/Y) on "
        "the substrate in the growth reaction; the emitter derives the coefficient. "
        "AOB on NH4-N ~0.1-0.15; NOB on NO2-N ~0.02-0.08.",
    ),
    VocabEntry(
        id="m_s",
        description="Maintenance coefficient on primary substrate.",
        canonical_unit="g_substrate/(g_biomass*day)",
        sanity_range=(0.0, 20.0),
        idynomics_xml_path=None,
        substrate_keyed=True,
        requires_reaction_insert=True,
        notes="Not present in the template kinetic expression. Emitter must insert "
        "a <reaction name='maintenance'> block or fold into growth stoichiometry.",
    ),
    VocabEntry(
        id="b",
        description="First-order biomass decay / endogenous-respiration rate.",
        canonical_unit="1/day",
        sanity_range=(0.0, 2.0),
        idynomics_xml_path=None,
        requires_reaction_insert=True,
        notes="Not present in the template kinetic expression. Emitter must insert "
        "a <reaction name='decay'> block. AOB/NOB b typically 0.05-0.2 /d.",
    ),
    # --- Solute transport ---
    VocabEntry(
        id="D_liquid",
        description="Diffusion coefficient of the solute in bulk liquid (for StudyContext.substrate).",
        canonical_unit="um^2/s",
        sanity_range=(10.0, 5000.0),
        idynomics_xml_path=(
            "compartment//solute[@name='{substrate}']/@defaultDiffusivity"
        ),
        substrate_keyed=True,
        notes="iDynoMiCS 2 uses [um+2/s]. Small molecules (O2, NH3): 1800-2500; "
        "glucose: 600-800. Biofilm value is typically 60-80% of liquid.",
    ),
    VocabEntry(
        id="D_biofilm",
        description="Effective diffusion coefficient of the solute inside biofilm matrix.",
        canonical_unit="um^2/s",
        sanity_range=(1.0, 5000.0),
        idynomics_xml_path=(
            "compartment//solute[@name='{substrate}']/@biofilmDiffusivity"
        ),
        substrate_keyed=True,
        notes="Often reported as ratio D_biofilm/D_liquid — extractor should "
        "capture both as records, or the ratio with note.",
    ),
    # --- Cell / agent properties ---
    VocabEntry(
        id="cell_density",
        description="Dry-biomass density of a cell / biofilm matrix element.",
        canonical_unit="pg/fL",
        sanity_range=(0.05, 0.5),
        idynomics_xml_path=(
            "species[@name='{morphology}']/aspect[@name='density']/@value"
        ),
        notes="iDynoMiCS 2 uses pg/fL (equivalent to g/L numerically: 0.15 pg/fL "
        "= 150 g-biomass/L wet basis). Watch 'dry' vs 'wet' basis — different "
        "schools report numerically different values for the same physical quantity.",
    ),
    VocabEntry(
        id="cell_division_mass",
        description="Dry mass at which an agent divides.",
        canonical_unit="pg",
        sanity_range=(1e-3, 1e3),
        idynomics_xml_path=(
            "species[@name='{morphology}']/aspect[@name='divisionMass']/@value"
        ),
    ),
    VocabEntry(
        id="cell_initial_mass",
        description="Initial dry mass per agent at simulation start.",
        canonical_unit="pg",
        sanity_range=(1e-4, 1e3),
        idynomics_xml_path=(
            "compartment/spawn/templateAgent/aspect[@name='biomass']"
            "//item[@key='mass']/@value"
        ),
    ),
    # --- EPS / biofilm matrix ---
    VocabEntry(
        id="Y_EPS",
        description="EPS yield (g_EPS per g_substrate consumed, or per g_biomass — record basis in notes).",
        canonical_unit="g_EPS/g_substrate",
        sanity_range=(0.0, 1.0),
        idynomics_xml_path=None,
        requires_reaction_insert=True,
        notes="Requires a separate EPS-production reaction in iDynoMiCS 2. Basis "
        "(per substrate vs per biomass) varies across schools; capture in context.notes.",
    ),
    # --- Initial solute concentrations (boundary + bulk) ---
    VocabEntry(
        id="S_bulk_initial",
        description="Initial bulk concentration of a solute (boundary / reactor feed).",
        canonical_unit="mg/L",
        sanity_range=(0.0, 1e4),
        idynomics_xml_path=(
            "compartment//solute[@name='{substrate}']/@concentration"
        ),
        substrate_keyed=True,
        notes="Not a kinetic parameter but often reported alongside and needed "
        "to reproduce a sim setup. iDynoMiCS 2 uses [mg/l].",
    ),
]


VOCAB: dict[str, VocabEntry] = {e.id: e for e in _CORE}


def get(parameter_id: str) -> VocabEntry:
    try:
        return VOCAB[parameter_id]
    except KeyError as exc:
        raise KeyError(f"Unknown parameter_id '{parameter_id}'. Known: {sorted(VOCAB)}") from exc


def all_ids() -> list[str]:
    return sorted(VOCAB)
