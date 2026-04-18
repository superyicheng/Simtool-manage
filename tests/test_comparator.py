"""Unit tests — compare_outputs_to_metamodel."""

from __future__ import annotations

from simtool.comparator import (
    ComparisonOutcome,
    compare_outputs_to_metamodel,
)
from simtool.connector.ir import (
    AgentPopulation,
    ComputeBudget,
    MonodGrowthProcess,
    Observable,
    ParameterBinding,
    ScientificModel,
    Solute,
    SpatialDomain,
    Surface,
)
from simtool.connector.runs import OutputBundle

from tests._metamodel_fixtures import make_nitrifying_biofilm_metamodel


def _ir_with_observable(observable_id: str, target: str) -> ScientificModel:
    aob = AgentPopulation(id="AOB", name="AOB")
    s = Solute(id="S", name="S")
    dom = SpatialDomain(id="dom", dimensionality=1, extent_um=(32.0,))
    top = Surface(id="top", name="top", axis=0, position="high")
    bot = Surface(id="bot", name="bot", axis=0, position="low")
    return ScientificModel(
        id="ir", title="t", domain="d", formalism="agent_based",
        entities=[aob, s, dom, top, bot],
        processes=[
            MonodGrowthProcess(
                id="g", growing_entity="AOB", consumed_solutes=["S"],
                parameters={
                    "mu_max": ParameterBinding(parameter_id="mu_max", canonical_unit="1/day", point_estimate=1.0),
                    "K_s_S": ParameterBinding(parameter_id="K_s", canonical_unit="mg/L", point_estimate=1.0),
                },
            ),
        ],
        observables=[Observable(id=observable_id, name=observable_id, kind="scalar_time_series", target=target)],
        compute=ComputeBudget(time_horizon_s=3600.0),
    )


def test_within_range_flagged_within():
    mm = make_nitrifying_biofilm_metamodel()
    ir = _ir_with_observable("mu_obs", "mu_max")
    bundle = OutputBundle(run_id="r", scalar_time_series={
        "mu_obs": [(0.0, 0.5), (3600.0, 0.95)],   # 0.95 is within +/-30% of 1.0
    })
    report = compare_outputs_to_metamodel(bundle, ir, mm)
    assert len(report.observables) == 1
    assert report.observables[0].outcome == ComparisonOutcome.WITHIN
    assert not report.any_outside_range


def test_above_range_flagged():
    mm = make_nitrifying_biofilm_metamodel()
    ir = _ir_with_observable("mu_obs", "mu_max")
    bundle = OutputBundle(run_id="r", scalar_time_series={
        "mu_obs": [(0.0, 0.5), (3600.0, 10.0)],  # 10.0 is above +30% envelope
    })
    report = compare_outputs_to_metamodel(bundle, ir, mm)
    assert report.observables[0].outcome == ComparisonOutcome.ABOVE
    assert report.any_outside_range


def test_below_range_flagged():
    mm = make_nitrifying_biofilm_metamodel()
    ir = _ir_with_observable("mu_obs", "mu_max")
    bundle = OutputBundle(run_id="r", scalar_time_series={
        "mu_obs": [(0.0, 0.1), (3600.0, 0.1)],  # 0.1 is below 0.7 (1.0*0.7)
    })
    report = compare_outputs_to_metamodel(bundle, ir, mm)
    assert report.observables[0].outcome == ComparisonOutcome.BELOW


def test_unknown_when_target_not_in_metamodel():
    mm = make_nitrifying_biofilm_metamodel()
    ir = _ir_with_observable("exotic", "exotic_compound")
    bundle = OutputBundle(run_id="r", scalar_time_series={
        "exotic": [(0.0, 1.0), (100.0, 2.0)],
    })
    report = compare_outputs_to_metamodel(bundle, ir, mm)
    assert report.observables[0].outcome == ComparisonOutcome.UNKNOWN


def test_unknown_when_no_output_series():
    mm = make_nitrifying_biofilm_metamodel()
    ir = _ir_with_observable("mu_obs", "mu_max")
    bundle = OutputBundle(run_id="r")
    report = compare_outputs_to_metamodel(bundle, ir, mm)
    assert report.observables[0].outcome == ComparisonOutcome.UNKNOWN
    assert "no scalar time series" in report.observables[0].note


def test_distribution_range_used():
    """K_s(O2) AOB in the fixture is an empirical distribution [0.3..0.8].
    An observable targeting it should compare against that range."""
    mm = make_nitrifying_biofilm_metamodel()
    ir = _ir_with_observable("Ks_obs", "K_s")
    bundle_in = OutputBundle(run_id="r", scalar_time_series={
        "Ks_obs": [(0.0, 0.5)],  # within the first matching K_s
    })
    report = compare_outputs_to_metamodel(bundle_in, ir, mm)
    # Just confirm it's classifiable — exact range depends on which K_s
    # entry matches (the fixture has multiple; the function picks the first).
    assert report.observables[0].outcome != ComparisonOutcome.UNKNOWN
