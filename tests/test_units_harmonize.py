"""Smoke tests for unit harmonization round-trips against canonical vocab units."""

from __future__ import annotations

import math

import pytest

from simtool.units.harmonize import harmonize


def _close(a: float, b: float, rel: float = 1e-6) -> bool:
    return math.isclose(a, b, rel_tol=rel, abs_tol=0.0)


def test_mu_max_per_hour_to_per_day():
    r = harmonize(value=0.05, unit_str="1/hour", parameter_id="mu_max")
    assert r.dimensional_check_passed
    assert _close(r.canonical_value, 1.2)  # 0.05 /h * 24 = 1.2 /d
    assert r.range_check_passed


def test_mu_max_idynomics_bracket_notation():
    r = harmonize(value=1.2, unit_str="[d-1]", parameter_id="mu_max")
    assert r.dimensional_check_passed
    assert _close(r.canonical_value, 1.2)


def test_ks_g_per_m3_equals_mg_per_L():
    r = harmonize(value=2.4, unit_str="g/m**3", parameter_id="K_s")
    assert r.dimensional_check_passed
    assert _close(r.canonical_value, 2.4)
    assert r.range_check_passed


def test_ks_micromolar_requires_compound_mw():
    r = harmonize(value=50.0, unit_str="umol/L", parameter_id="K_s")
    # No MW lookup provided, so conversion must fail cleanly.
    assert not r.dimensional_check_passed
    assert "compound" in r.detail or "molar" in r.detail


def test_ks_micromolar_with_nh4_mw():
    class _NH4Lookup:
        def molar_mass_g_per_mol(self, compound: str):
            return 18.04 if compound.lower() in ("nh4", "nh4+", "ammonium") else None

    r = harmonize(
        value=50.0,
        unit_str="umol/L",
        parameter_id="K_s",
        compound="NH4",
        molar_mass_lookup=_NH4Lookup(),
    )
    assert r.dimensional_check_passed
    # 50 umol/L * 18.04 g/mol = 902 ug/L = 0.902 mg/L
    assert _close(r.canonical_value, 0.902, rel=1e-3)
    assert r.range_check_passed


def test_diffusivity_m2_per_s_to_um2_per_s():
    r = harmonize(value=2e-9, unit_str="m**2/s", parameter_id="D_liquid")
    assert r.dimensional_check_passed
    assert _close(r.canonical_value, 2000.0)  # 2e-9 m2/s = 2000 um2/s
    assert r.range_check_passed


def test_range_check_flags_absurd_value():
    # mu_max of 200 /d is nonsense — outside sanity range
    r = harmonize(value=200.0, unit_str="1/day", parameter_id="mu_max")
    assert r.dimensional_check_passed
    assert not r.range_check_passed


def test_incompatible_unit_rejected():
    # Try to harmonize a mass as if it were a rate
    r = harmonize(value=1.0, unit_str="gram", parameter_id="mu_max")
    assert not r.dimensional_check_passed
    assert "incompatible" in r.detail


def test_unknown_parameter_id_raises():
    with pytest.raises(KeyError):
        harmonize(value=1.0, unit_str="1/day", parameter_id="not_a_real_param")
