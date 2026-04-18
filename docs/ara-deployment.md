# Deploying Simtool in Ara

Ara has no frontend surface. The user interacts with the AI purely through chat messages (and image attachments). Simtool ships to Ara as a plain Python folder on the cloud VM that backs the AI agent.

## What gets deployed

The whole `simtool/` package plus a minimum runtime:

```
cloud_vm/
  simtool-manage/
    simtool/
      metamodel/      # shared scientific substrate
      panels/         # per-user workspaces
      connector/      # IR + plugin + run harness
      frameworks/     # iDynoMiCS 2 plugin (and future frameworks)
      runner/         # orchestrator that drives the plugin contract
      comparator/     # post-run output vs meta-model comparison
      persistence/    # JSON save/load for meta-models and panels
      ara/            # text + image renderers + AraSession
      ...
    pyproject.toml
```

Python 3.11+. Install with `pip install -e '.[ara]'` so matplotlib is present for chart images.

### iDynoMiCS 2 runtime

To actually run simulations, also install:

- **Java 11+** on PATH (or `JAVA_HOME` set).
- **iDynoMiCS 2.0 jar** — download from [the release page](https://github.com/kreft/iDynoMiCS-2/releases). Place the full extracted folder somewhere on the VM (it ships with `lib/`, `default.cfg`, and `config/` that the jar needs).

Point Simtool at the jar with one of:

```bash
# Option 1: environment variable (simplest)
export IDYNOMICS_2_JAR=/path/to/iDynoMiCS-2.0.jar

# Option 2: config file
mkdir -p ~/.simtool
cat > ~/.simtool/idynomics_2.json <<EOF
{"jar_path": "/path/to/iDynoMiCS-2.0.jar"}
EOF

# Option 3: drop the jar at ./vendor/iDynoMiCS-2.0.jar under the repo
```

Verify with:

```python
from simtool.frameworks.idynomics_2 import resolve_jar_path
print(resolve_jar_path())   # should print the absolute jar path
```

If none of the paths resolve, `AraSession.run()` raises `IDynoMiCS2NotAvailable` with install guidance — no silent fallback.

## How the agent uses it

The agent reasons in natural language; when it needs to act or report, it either (a) calls individual renderers for ad-hoc questions, or (b) uses the `AraSession` wrapper for a full working loop.

### Individual renderers

```python
from simtool.ara import render_metamodel_summary, render_recommendation
from simtool.panels import recommend_model

text = render_metamodel_summary(meta_model)
# -> markdown string the agent returns as its reply

rec = recommend_model(meta_model, constraints)
reply_text = render_recommendation(rec)
```

### AraSession — the common path

```python
from simtool.ara import AraSession

session = AraSession(root="~/simtool")   # persistent store

# boot: load a meta-model (by id; latest version by default)
reply = session.open_metamodel("nitrifying_biofilm")
# -> markdown summary with staleness warning if applicable

# create a panel for the user's simulation project
reply = session.create_panel(
    user_id="alice",
    title="my AOB chemostat",
    derived_ir=my_ir,                 # ScientificModel
    required_phenomena=["growth"],
    observable_ids=["thickness"],
)

# three workflows — each returns markdown
reply = session.recommend()
reply = session.adjust(kind="change_parameter", target_id="mu_max")
reply = session.fit(datasets=[...], fit_results=[...])

# run the panel end-to-end through iDynoMiCS 2
reply_text, summary = session.run(auto_approve_assumptions=True)
# summary.progress_reports, summary.output_bundle, summary.protocol_doc_path
# are available for the agent to attach charts to the reply.
```

`AraSession.run()` orchestrates: `validate_ir → lower → approve ledger → execute iDynoMiCS → stream progress → parse outputs → harvest from jar_dir/results → generate ODD protocol → write RunRecord → attach to panel history → persist`.

Every major Simtool object has a text renderer in `simtool.ara.render_text`:

| Object                  | Renderer                                |
|-------------------------|-----------------------------------------|
| MetaModel               | `render_metamodel_summary`              |
| ReconciledParameter     | `render_metamodel_parameter`            |
| All / filtered params   | `render_metamodel_parameters`           |
| Submodel hierarchy      | `render_submodel_hierarchy`             |
| Scope status report     | `render_scope_status`                   |
| Panel                   | `render_panel_summary`                  |
| Panel overrides         | `render_panel_overrides`                |
| ModelRecommendation     | `render_recommendation`                 |
| AdjustmentProposal      | `render_adjustment_proposal`            |
| FitDataResult           | `render_fit_result`                     |
| Suggestion              | `render_suggestion`                     |
| SuggestionLedger        | `render_suggestion_ledger_summary`      |
| AssumptionLedger        | `render_assumption_ledger`              |
| ProgressReport          | `render_progress_line` (single) / `render_progress_stream` (iterable) |
| OutputBundle            | `render_output_bundle`                  |
| ScientificModel (IR)    | `render_ir_compact`                     |

For images, `simtool.ara.render_image` provides:

- `render_scalar_timeseries(series, reference_band=...)` — time series with optional meta-model band overlay.
- `render_output_bundle_overview(bundle)` — up to 4 stacked series from one run.
- `render_distribution(distribution)` — empirical histograms or parametric densities.
- `render_reconciled_parameter(rp)` — auto-routes to distribution or point-estimate view.

Each returns PNG bytes. The agent attaches them to messages.

## Persistence

Ara's VM holds state on disk. Simtool objects are pydantic models with `model_dump_json()` / `model_validate_json()` — persist to files, read back at session start. A conservative layout:

```
~/simtool/
  metamodels/<id>/<version>/metamodel.json
  panels/<panel_id>/panel.json
  runs/<run_id>/          # RunLayout.under(...).ensure()
  suggestions/<metamodel_id>/ledger.json
```

No database needed for single-user deployment.

## Long-running simulations

Runs produce a `RunLayout`-shaped directory under `~/simtool/runs/<run_id>/`. The plugin's `monitor(handle)` yields `ProgressReport` objects; the agent renders each with `render_progress_line` and posts as a chat message. Images (final charts) are attached once `parse_outputs` completes.

The run layout is always:

```
runs/<run_id>/
  inputs/
    protocol.xml          # IR lowered to iDynoMiCS protocol
    default.cfg           # staged if the jar needs it
  outputs/
    <timestamp>_<sim>/    # harvested from iDynoMiCS's default results/ dir
      <sim>_00001.xml     # agent-state snapshots
      ...
  logs/
    idynomics.log         # full stdout+stderr
  protocol/
    ODD.md                # reproducibility protocol (auto-generated)
  metadata/
    run.json              # RunRecord with ledger hash, status, timings
```

The `ODD.md` + `run.json` pair is the citable scientific artifact of each run; the rest is reproducibility machinery.

## What's missing (vs the standalone frontend)

- No live canvas editing — model edits happen via chat ("add a decay process for NOB") invoking `adjust_model`.
- No multi-user collaboration UI — the `Panel.share_with` mechanism still works, but presence/notifications are not exposed through Ara.
- No in-app meta-model browser — replaced by `render_metamodel_*` messages.

These gaps are intentional: the frontend is a separate distribution target. Ara is the text-shaped one.

## The boundary

Ara is a distribution channel, not a replacement for the scientific substrate. The same meta-models, panels, connector pipeline, and canonical tests govern both deployment paths. Changes to scientific content go through the meta-model community suggestion flow regardless of which surface the user reached it from.
