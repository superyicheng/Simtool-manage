# Simtool-manage

Meta-analytic simulation agent — literature to reconciled meta-model to runnable simulation.

## What it is

Simtool turns scientific literature into a **meta-model**: a community-maintained, versioned, citable reconciliation of the field's knowledge about a specific system (e.g., nitrifying biofilm, lithium-ion cathode degradation). Users layer **panels** on top — personal workspaces that derive a tailored simulation model from the meta-model, run it against a chosen framework, and compare outputs to literature-reported ranges.

The scientific novelty lives in the meta-model and the assumption-surfacing translation. The connector, panels, and frontends are engineering infrastructure built to make those contributions testable and usable.

## Layout

```
simtool/
  connector/      # shared Intermediate Representation + framework plugin contract
                  # + skill files + assumption ledger + run harness
  metamodel/      # MetaModel library + scope contract + ingestion tiers +
                  # community suggestions + SemVer propagation policy
  panels/         # user workspaces + three workflows (recommend / adjust / fit)
  ara/            # text + image renderers for Ara chat deployment
  schema/         # atomic ParameterRecord (extractor output contract)
  corpus/         # literature fetchers (PMC, bioRxiv, ...)
  extractors/     # per-modality extraction
  matcher/        # cross-paper reconciliation of ParameterRecords
  metamodel/      # (reconciler internals populate the MetaModel shape)
  qc/             # dimensional checks + GRADE rating
  units/          # pint-backed harmonization
  emitter/        # framework-specific lowering (iDynoMiCS 2 first)
  runner/         # supervised execution harness
  comparator/     # post-run comparison vs meta-model
frontend/         # Next.js + Supabase + Cloud Run spec (separate deploy)
docs/             # deployment and architecture notes
tests/            # unit, integration, canonical
```

## Distribution paths

1. **Standalone web frontend** — Next.js + Supabase + Cloud Run. Four-zone panel IDE (canvas, parameters, runs, chat) + meta-model navigator. See `frontend/README.md`.
2. **Ara cloud folder** — the whole `simtool/` package deployed to an Ara VM; all interaction via chat messages. Text renderers in `simtool.ara.render_text`; image renderers in `simtool.ara.render_image`. See `docs/ara-deployment.md`.

Both consume the same Python substrate. Scientific content changes go through the meta-model suggestion flow regardless of surface.

## Quickstart

```bash
# core install
pip install -e '.[dev]'

# Ara deployment (adds matplotlib for chart images)
pip install -e '.[ara]'

# run tests (three-layer: unit + integration + canonical)
pytest
```

## Design posture

- Don't silently pool conflicting measurements — reconciled parameters surface disagreement as distributions with conflict flags.
- Every implicit assumption the system makes during lowering is in the assumption ledger, user-approved before execution.
- Every run produces a reproducibility protocol (ODD for agent-based models) as its primary scientific artifact.
- Frozen panels lock to a meta-model version for publication reproducibility.
- Parameters with no usable consensus are MISSING — they block runs until the user supplies a value or accepts a speculative default marked as such.

## Test layers

- `test_connector_*.py`, `test_metamodel_*.py`, `test_panels.py`, `test_panel_workflows.py`, `test_ara_rendering.py`, `test_units_harmonize.py`, `test_fetcher.py`, `test_settings.py` — unit.
- `test_integration_meta_panel_connector.py` — cross-module seams.
- `test_canonical_*.py` — user-shaped end-to-end workflows (trivial / realistic / adversarial / recommend / adjust / fit).

Canonical tests encode user stories. If any of them breaks, the build is broken regardless of what the unit tests say.
