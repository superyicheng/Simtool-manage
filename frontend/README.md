# Simtool Frontend

The standalone web frontend for Simtool. Separate deploy; consumes the Python backend via a REST/JSON API. This README specifies architecture and backend contract; the Next.js project itself gets scaffolded when the Python API freezes.

## Stack

- **Next.js 14** (App Router) on **Vercel** — SSR/edge for the marketing + meta-model browse surfaces; client components for the panel IDE.
- **Supabase** — auth (magic link + GitHub), row-level security, storage for user-uploaded datasets and frozen panel snapshots.
- **Cloud Run / dedicated VPS** — simulation execution (long-running, bursty). Frontend orchestrates via a job-queue API; it does not host compute.
- **Python backend** (this repo) — the scientific substrate. Exposed via a thin FastAPI or equivalent layer that wraps the existing modules. The frontend never imports Python; it speaks JSON.

## Four-zone panel UI

The panel is the primary work surface. The layout has four coexisting zones, each driven by the same underlying `Panel` JSON plus live streams from long-running jobs:

1. **Model canvas**  
   Visual rendering of the panel's `derived_ir` (entities as nodes, processes as typed connectors, boundaries as surface annotations). Editable two ways: (a) direct manipulation (drag a Monod process onto AOB), (b) AI chat ("add a maintenance process for NOB"). Writes go through the adjustment workflow endpoint.

2. **Parameters & assumptions**  
   A browsable ledger. For every `ParameterBinding` in the IR, shows: canonical unit, point estimate or distribution, source note, provenance DOIs (click to see the reconciliation), and whether it's inherited from the meta-model or overridden. Assumption ledger sits alongside with per-item Approve/Reject/Alternative controls.

3. **Runs & analysis**  
   List of `RunHistoryEntry` with status, timestamps, links to RunRecord metadata. Selecting a run opens a chart view overlaying the `OutputBundle` against the meta-model's literature-reported expected ranges (drawn from `ReconciledParameter` ranges and plugin-specific comparators). Anomalies highlighted.

4. **AI conversation**  
   Scoped to the panel. Messages carry structured tool calls (e.g. `recommend_model`, `adjust_model`, `fit_data`); tool outputs render as rich cards in the other zones, not as prose.

## Meta-model navigator

The meta-model is a separate, global view reachable from any panel via an "up arrow." It shows:

- Scope contract (what's in-scope, what's not).
- Submodel hierarchy (tree, rendered by complexity rank).
- Approximation operators (edges between submodels).
- Reconciled parameters (sortable table; click → provenance).
- Changelog (credits accepted suggesters).
- Pending community suggestions on every artifact.
- Staleness banner when `is_stale(meta_model)` is true.

## Publication state

Panel UI exposes four states: `DRAFT`, `SHARED`, `PUBLISHED`, `FROZEN`. Freezing is a one-click action that locks `meta_model_version_pin`; `FROZEN` panels show a green "reproducible" badge next to their citation block. Unfreezing is a single button with a confirm — or the user forks.

## Suggestion affordance

Every displayed meta-model artifact (parameter cell, reconciliation tooltip, submodel node, scope line) has an inline "suggest" icon. Clicking opens a modal pre-populated with the artifact's target kind/id/context. The modal enforces `Evidence` — the submit button stays disabled until at least one `Evidence` entry (DOI / personal data / domain argument) is attached. Submitted suggestions are public by default.

## Propagation notifications

When a meta-model advances:

- `PATCH` / `MINOR`: a toast appears on every open panel pinned to the old version: "meta-model nitrifying-biofilm moved 1.2.0 → 1.3.0 (minor). 2 parameters updated." Click to see diff. Auto-applied.
- `MAJOR`: a persistent banner replaces the toast, with "Review & confirm" as the only action. Nothing propagates until the user acknowledges.
- `FROZEN` panels: no notification; frozen panels are out of this flow by design.

## Backend API surface the frontend consumes

The Python backend must expose (minimum set):

```
GET  /metamodels                              list metamodels
GET  /metamodels/{id}                         full MetaModel JSON
GET  /metamodels/{id}/versions                version list
GET  /metamodels/{id}/parameters              flattened reconciled params
GET  /metamodels/{id}/staleness               {is_stale, warning, last_ingestion_at}
GET  /metamodels/{id}/suggestions             SuggestionLedger
POST /metamodels/{id}/suggestions             submit Suggestion
POST /metamodels/{id}/suggestions/{sid}/review   maintainer action

POST /metamodels/{id}/ingestion/run           manual trigger
GET  /metamodels/{id}/ingestion/jobs          recent IngestionJob list

GET  /panels                                  user's panels
POST /panels                                  create (empty) or fork
GET  /panels/{id}                             full Panel JSON
PATCH /panels/{id}                            edit notes, tags, constraints
POST /panels/{id}/freeze                      freeze
POST /panels/{id}/unfreeze                    unfreeze
POST /panels/{id}/share                       add collaborator

POST /panels/{id}/workflows/recommend         ModelRecommendation
POST /panels/{id}/workflows/adjust            AdjustmentProposal
POST /panels/{id}/workflows/fit               FitDataResult
POST /panels/{id}/workflows/fit/submit-suggestions  push suggestions to meta-model

POST /panels/{id}/runs                        schedule a run (returns job id)
GET  /panels/{id}/runs/{run_id}               RunRecord
GET  /panels/{id}/runs/{run_id}/stream        SSE stream of ProgressReport
GET  /panels/{id}/runs/{run_id}/outputs       OutputBundle
GET  /panels/{id}/runs/{run_id}/protocol      protocol document
```

Every endpoint carries the meta-model version pin in response headers (`X-MetaModel-Version`) so the UI can detect and surface mismatches.

## Build order

1. Python backend: FastAPI wrapper exposing the above surface. Panels + MetaModel + connector already implemented; this layer is serialization + routes.
2. Next.js scaffold: auth, meta-model browse, panel shell.
3. Four-zone panel IDE in order of load-bearing-ness: parameters → runs → canvas → chat.
4. Suggestion flow.
5. Propagation notifications + freeze UX.

## Skill-later note

Once the standalone frontend stabilizes, the orchestration logic (panel workflows, run scheduling) gets packaged as a **skill** for agent harnesses like Ara. The skill is a thin wrapper over the same backend endpoints — it is additive distribution, not a replacement for the standalone frontend.
