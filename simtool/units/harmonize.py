"""Units harmonization against the iDynoMiCS 2 canonical units declared in
`simtool.schema.idynomics_vocab`.

Responsibilities:
  - Parse an arbitrary reported unit string via `pint`.
  - Check that its dimensions are compatible with the parameter's canonical unit.
  - Convert to the canonical numeric value.
  - Sanity-range check.

Dimensionless yields (g/g) are handled specially because pint reduces them to
dimensionless `1` and loses the biomass/substrate label. We normalize within the
mass-ratio dimension but keep the label externally on the ParameterRecord.

Molar-to-mass and mass-to-molar conversions (e.g. mmol/L -> mg/L) require a
molar mass for the specific compound; this module exposes a `MolarMassLookup`
protocol so a PubChem-backed implementation can be plugged in later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Protocol

import pint

from simtool.schema.idynomics_vocab import VOCAB, VocabEntry


_UREG = pint.UnitRegistry()
# iDynoMiCS 2 writes rate units as [d-1] / [h-1] / [s-1] and area-per-time as
# [um+2/s]. pint accepts `1/day`, `um**2/s`, etc. directly — no custom defs
# needed. For readability we normalize `[d-1]` style strings before parsing.


_CHEMICAL_SUFFIX_RE = re.compile(
    r"\s+(?:as\s+)?(?:[A-Z][a-z]?[0-9]*[+-]?|N|TAN|COD|MLVSS|VSS|TSS|DW|total\s+ammonium|total\s+ammonia|"
    r"NH4\+\s*\+\s*NH3|NH3\s*\+\s*NH4\+?)(?:\s+(?:basis|equivalent))?$",
    re.IGNORECASE,
)
# Common per-substrate annotations that should be dropped before pint sees the string
# (e.g. "mg N/L" -> "mg/L", "nmol NH3/L" -> "nmol/L")
_INFIX_CHEMICAL_RE = re.compile(
    r"\s+(?:N|NH3|NH4\+?|NO2-?|NO3-?|O2|CO2|TAN|COD)\s*(?=/)",
    re.IGNORECASE,
)
# Hyphen-chemical convention common in wastewater engineering:
# "mg-N/L" ≡ "mg-of-nitrogen per L" → we treat as "mg/L" for v1.
_HYPHEN_CHEMICAL_RE = re.compile(
    r"-(?:N|NH3|NH4\+?|NO2-?|NO3-?|O2|CO2|TAN|COD|TN|VSS|MLVSS)(?=[\s/]|$)",
    re.IGNORECASE,
)


def _pintify(unit_str: str) -> str:
    """Normalize reported unit strings into pint-parseable syntax.

    Handles a bunch of field conventions that pint can't parse directly:
      - iDynoMiCS bracket notation: [d-1], [um+2/s], [g/m+3], [mg/l]
      - trailing negative exponents: 'd-1', 'h-1', 's-1', 'min-1'
      - m2, um2, cm3 -> m**2, um**2, cm**3
      - chemical suffixes: "mg N/L" -> "mg/L" (elemental N basis preserved
        only notionally — we treat as equivalent mass/volume; user of the
        meta-model should be aware)
      - micro-sign µ/μ -> u
      - non-standard casing: mg/l -> mg/L
    """

    import re as _re  # local to avoid polluting module top if unused

    s = unit_str.strip()

    # bracket notation from iDynoMiCS
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]

    # Drop chemical annotations BEFORE whitespace stripping so the regex can
    # rely on word boundaries (e.g. "mg N/L" needs the space, "mg-N/L" the hyphen).
    # Hyphen-infix: "mg-N/L" / "mg-O2/L" -> "mg/L"
    s = _HYPHEN_CHEMICAL_RE.sub("", s)
    # Whitespace-infix: "mg N/L" -> "mg/L"
    s = _INFIX_CHEMICAL_RE.sub("", s)
    # Trailing: "nM NH3" / "uM NO2-" / "mg/L as NH4+" -> "nM" / "uM" / "mg/L"
    prev = None
    while prev != s:
        prev = s
        s = _CHEMICAL_SUFFIX_RE.sub("", s)

    s = s.replace(" ", "")

    # exponent markers (iDynoMiCS: um+2 -> um**2)
    s = s.replace("+", "**")

    # m2, m3, cm2, cm3, um2, um3 -> m**2 etc. Only apply when the base is a
    # known length unit so we don't mangle chemical tokens.
    s = _re.sub(r"\b(um|mm|cm|dm|m|km|nm|fm|pm|Å)(\d+)\b", r"\1**\2", s)

    # Handle trailing negative exponents like 'd-1', 'h-1', 's-1', 'day-1', 'hr-1'
    for base, canonical in (
        ("day", "day"),
        ("d", "day"),
        ("hr", "hour"),
        ("h", "hour"),
        ("s", "second"),
        ("min", "minute"),
    ):
        token = f"{base}-1"
        if s == token:
            s = f"1/{canonical}"
            break
        if s.endswith(token):
            s = s[: -len(token)] + f"/{canonical}"
            break

    # 'd' alone is ambiguous in pint (day vs deci-); disambiguate common cases.
    if s.endswith("/d"):
        s = s[:-1] + "day"

    # Litre casing
    s = s.replace("mg/l", "mg/L").replace("g/l", "g/L").replace("ug/l", "ug/L").replace("nmol/l", "nmol/L").replace("umol/l", "umol/L")

    # micro-sign -> u
    s = s.replace("μ", "u").replace("µ", "u")

    return s


class MolarMassLookup(Protocol):
    def molar_mass_g_per_mol(self, compound: str) -> Optional[float]: ...


# Built-in MWs for the nitrogen cycle + oxygen. Covers all common substrate
# labels that show up in nitrifier kinetic papers. A full PubChem lookup is
# pluggable via the `MolarMassLookup` protocol if we need more compounds.
_BUILTIN_MW: dict[str, float] = {
    "nh3": 17.031,
    "nh4": 18.039,
    "nh4+": 18.039,
    "ammonium": 18.039,
    "ammonia": 17.031,
    "total ammonium": 18.039,
    "total ammonia": 17.031,
    "tan": 14.007,  # TAN (total ammonia nitrogen) is usually expressed as N mass
    "no2": 46.005,
    "no2-": 46.005,
    "nitrite": 46.005,
    "no3": 62.004,
    "no3-": 62.004,
    "nitrate": 62.004,
    "o2": 31.998,
    "oxygen": 31.998,
    "n": 14.007,  # elemental N basis
    "co2": 44.009,
    "glucose": 180.156,
    "glucose-c": 12.011,  # C basis of glucose
}


class _BuiltinMolarMassLookup:
    def molar_mass_g_per_mol(self, compound: str) -> Optional[float]:
        if not compound:
            return None
        return _BUILTIN_MW.get(compound.strip().lower())


_DEFAULT_MW_LOOKUP = _BuiltinMolarMassLookup()


@dataclass
class HarmonizationResult:
    canonical_value: float
    canonical_unit: str
    dimensional_check_passed: bool
    range_check_passed: bool
    detail: str
    """Human-readable note (e.g. 'converted hr->day', 'used MW(NH3)=17.03 g/mol')."""


def harmonize(
    value: float,
    unit_str: str,
    parameter_id: str,
    *,
    compound: Optional[str] = None,
    molar_mass_lookup: Optional[MolarMassLookup] = None,
) -> HarmonizationResult:
    """Convert `(value, unit_str)` to the canonical unit for `parameter_id`.

    If the reported unit is molar (e.g. mmol/L) but the canonical unit is mass
    (mg/L), a `compound` and `molar_mass_lookup` are required.
    """

    entry: VocabEntry = VOCAB[parameter_id]

    # --- dimensionless yields bypass pint entirely (pint can't parse
    #     'g_biomass/g_substrate' — those labels are ours, not unit tokens).
    if "_biomass/" in entry.canonical_unit or "_substrate" in entry.canonical_unit or "_EPS/" in entry.canonical_unit:
        # Accept raw ratios in common forms: g/g, kg/kg, mg/mg, dimensionless.
        try:
            source_q = _UREG.Quantity(value, _pintify(unit_str.split("_")[0] if "_" in unit_str else unit_str))
            ratio_value = source_q.to("dimensionless").magnitude if source_q.dimensionless else value
        except (pint.errors.UndefinedUnitError, pint.errors.DimensionalityError):
            ratio_value = value
        range_ok = entry.sanity_range[0] <= ratio_value <= entry.sanity_range[1]
        return HarmonizationResult(
            canonical_value=ratio_value,
            canonical_unit=entry.canonical_unit,
            dimensional_check_passed=True,
            range_check_passed=range_ok,
            detail="ratio (g/g) — passed through",
        )

    # --- normal pint-based path ---
    canonical_str = _pintify(entry.canonical_unit)
    canonical_q = _UREG.Quantity(1.0, canonical_str)
    source_str = _pintify(unit_str)
    try:
        source_q = _UREG.Quantity(value, source_str)
    except (pint.errors.UndefinedUnitError, pint.errors.PintError, AssertionError, ValueError, TypeError) as exc:
        return HarmonizationResult(
            canonical_value=float("nan"),
            canonical_unit=entry.canonical_unit,
            dimensional_check_passed=False,
            range_check_passed=False,
            detail=f"unparseable unit '{unit_str}' -> pint '{source_str}': {type(exc).__name__}: {exc}",
        )

    # Compatible dimensions?
    if source_q.dimensionality == canonical_q.dimensionality:
        conv = source_q.to(canonical_str).magnitude
        range_ok = entry.sanity_range[0] <= conv <= entry.sanity_range[1]
        return HarmonizationResult(
            canonical_value=conv,
            canonical_unit=entry.canonical_unit,
            dimensional_check_passed=True,
            range_check_passed=range_ok,
            detail=f"{source_q.units} -> {canonical_q.units}",
        )

    # Mismatched dimensions — try molar<->mass concentration conversion.
    if _is_molar_concentration(source_q) and _is_mass_concentration(canonical_q):
        if compound is None or molar_mass_lookup is None:
            return HarmonizationResult(
                canonical_value=float("nan"),
                canonical_unit=entry.canonical_unit,
                dimensional_check_passed=False,
                range_check_passed=False,
                detail="molar->mass conversion needs compound + molar_mass_lookup",
            )
        mw = molar_mass_lookup.molar_mass_g_per_mol(compound)
        if mw is None:
            return HarmonizationResult(
                canonical_value=float("nan"),
                canonical_unit=entry.canonical_unit,
                dimensional_check_passed=False,
                range_check_passed=False,
                detail=f"no molar mass available for '{compound}'",
            )
        # Convert molar concentration to mol/L, multiply by g/mol, convert to canonical.
        molar_per_L = source_q.to("mol/L").magnitude
        mass_per_L_g = molar_per_L * mw
        mass_q = _UREG.Quantity(mass_per_L_g, "g/L")
        conv = mass_q.to(canonical_str).magnitude
        range_ok = entry.sanity_range[0] <= conv <= entry.sanity_range[1]
        return HarmonizationResult(
            canonical_value=conv,
            canonical_unit=entry.canonical_unit,
            dimensional_check_passed=True,
            range_check_passed=range_ok,
            detail=f"molar->mass via MW({compound})={mw} g/mol",
        )

    return HarmonizationResult(
        canonical_value=float("nan"),
        canonical_unit=entry.canonical_unit,
        dimensional_check_passed=False,
        range_check_passed=False,
        detail=f"incompatible dimensions: {source_q.dimensionality} vs {canonical_q.dimensionality}",
    )


def _is_molar_concentration(q: pint.Quantity) -> bool:
    return q.dimensionality == _UREG.Quantity(1.0, "mol/L").dimensionality


def _is_mass_concentration(q: pint.Quantity) -> bool:
    return q.dimensionality == _UREG.Quantity(1.0, "g/L").dimensionality
