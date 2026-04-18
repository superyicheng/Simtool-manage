"""Chemostat extractor — continuous-culture steady-state kinetic parameters."""

from __future__ import annotations

from simtool.extractors.base import BaseExtractor, MethodProfile
from simtool.schema.parameter_record import MeasurementMethod


class ChemostatExtractor(BaseExtractor):
    """Extract parameters reported from chemostat / continuous-culture experiments.

    Chemostat-derived Ks values are typically biased high vs intrinsic Ks due
    to diffusion limitation in the culture — we do NOT adjust the value here
    (that's reconciliation's job), but we capture the method faithfully so
    the reconciler can apply the right bias correction at its layer.
    """

    method_profile = MethodProfile(
        method=MeasurementMethod.CHEMOSTAT,
        prose_hints=[
            "Continuous culture at fixed dilution rate D (D = F/V, often h^-1 or d^-1).",
            "Steady-state: mu = D. mu_max is reported as the washout dilution rate, "
            "or fitted from Monod plots of mu vs S.",
            "Ks from steady-state substrate concentration at different D, fitted to "
            "S = D*Ks/(mu_max - D).",
            "Yield Y_XS from steady-state biomass X vs (S0 - S) at fixed D.",
            "Often reports substrate-limitation regime (carbon-limited, N-limited, etc.).",
        ],
        expected_parameters=[
            "mu_max",
            "K_s",
            "Y_XS",
            "m_s",
            "b",
        ],
    )
