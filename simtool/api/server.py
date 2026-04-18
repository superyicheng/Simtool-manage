"""FastAPI backend for the Simtool desktop app.

The Electron main process spawns this server (bound to 127.0.0.1 on an
ephemeral port) and reads the chosen port from stdout. Endpoints are
intentionally narrow: everything drives off the on-disk MetaModel JSON
and the extractions JSON. Workflow endpoints wrap the Python panel
workflows. Run endpoints drive either the demo plugin or the real
iDynoMiCS 2 plugin (when its jar is configured).

This is NOT the `frontend/` (Next.js + Vercel) API surface. The desktop
app is a local convenience wrapper around the same Python modules.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterator, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from simtool.api.demo_plugin import DemoPlugin
from simtool.connector.ir import (
    AgentPopulation,
    BoundaryCondition,
    ComputeBudget,
    InitialCondition,
    MonodGrowthProcess,
    Observable,
    ParameterBinding,
    SamplingSpec,
    ScientificModel,
    Solute,
    SpatialDomain,
    Surface,
)
from simtool.connector.runs import OutputBundle, ProgressReport, RunLayout, RunStatus
from simtool.connector.skill import SourceKind, SourceRef
from simtool.connector.plugin import DocSources
from simtool.frameworks.idynomics_2.config import resolve_jar_path
from simtool.frameworks.idynomics_2.doctor import run_health_check
from simtool.frameworks.idynomics_2.plugin import IDynoMiCS2Plugin
from simtool.metamodel.library import MetaModel
from simtool.panels.panel import (
    Panel,
    UserConstraints,
)
from simtool.panels.workflows import (
    AdjustmentRequest,
    adjust_model,
    recommend_model,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
METAMODEL_PATH = REPO_ROOT / "data" / "metamodel" / "nitrifying_biofilm_v0_1_0.json"
EXTRACTIONS_PATH = REPO_ROOT / "data" / "extractions" / "nitrifying_biofilm.json"
PLOTS_DIR = Path.home() / "Downloads" / "simtool-metamodel-plots"


# ---------------------------------------------------------------------------
# App + state
# ---------------------------------------------------------------------------

app = FastAPI(title="Simtool Local API", version="0.1.0")

# Electron loads renderer from file://; keep CORS permissive for localhost.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class _State:
    metamodel: Optional[MetaModel] = None
    metamodel_dict: Optional[dict[str, Any]] = None
    extractions: list[dict[str, Any]] = []
    progress: dict[str, list[ProgressReport]] = {}
    run_threads: dict[str, threading.Thread] = {}
    run_framework: dict[str, str] = {}
    default_panel: Optional[Panel] = None


state = _State()


def _reload_metamodel() -> None:
    if not METAMODEL_PATH.exists():
        return
    raw = json.loads(METAMODEL_PATH.read_text())
    state.metamodel_dict = raw
    state.metamodel = MetaModel.model_validate(raw)


def _reload_extractions() -> None:
    if not EXTRACTIONS_PATH.exists():
        return
    state.extractions = json.loads(EXTRACTIONS_PATH.read_text()).get("extractions", [])


@app.on_event("startup")
def _on_startup():
    _reload_metamodel()
    _reload_extractions()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "metamodel_loaded": state.metamodel is not None,
        "n_reconciled_params": len(state.metamodel.reconciled_parameters) if state.metamodel else 0,
        "n_extractions": len(state.extractions),
    }


# ---------------------------------------------------------------------------
# MetaModel
# ---------------------------------------------------------------------------


@app.get("/api/metamodel")
def get_metamodel():
    if state.metamodel is None:
        raise HTTPException(404, f"No metamodel at {METAMODEL_PATH}. Run `simtool build-metamodel` first.")
    return state.metamodel_dict


@app.get("/api/metamodel/parameters")
def list_parameters():
    if state.metamodel is None:
        raise HTTPException(404, "No metamodel loaded.")
    out = []
    for rp in state.metamodel.reconciled_parameters:
        b = rp.binding
        row: dict[str, Any] = {
            "parameter_id": rp.parameter_id,
            "context_keys": rp.context_keys,
            "canonical_unit": b.canonical_unit,
            "quality_rating": rp.quality_rating.value,
            "supporting_record_dois": rp.supporting_record_dois,
            "conflict_flags": rp.conflict_flags,
            "source_note": b.source_note,
        }
        if b.point_estimate is not None:
            row["point_estimate"] = b.point_estimate
            row["samples"] = None
        elif b.distribution is not None and b.distribution.shape == "empirical":
            row["point_estimate"] = None
            row["samples"] = b.distribution.samples
        else:
            row["point_estimate"] = None
            row["samples"] = None
        out.append(row)
    return out


@app.get("/api/metamodel/plot/{name}")
def metamodel_plot(name: str):
    safe = {
        "overview": "1_metamodel_overview.png",
        "data_mu_max": "2_data_mu_max.png",
        "data_K_s": "2_data_K_s.png",
        "data_Y_XS": "2_data_Y_XS.png",
        "data_b": "2_data_b.png",
        "data_D_biofilm": "2_data_D_biofilm.png",
        "data_S_bulk_initial": "2_data_S_bulk_initial.png",
        "quality": "3_quality_summary.png",
        "coverage": "4_coverage_matrix.png",
    }
    fname = safe.get(name)
    if fname is None:
        raise HTTPException(400, f"Unknown plot name '{name}'. Available: {sorted(safe)}.")
    path = PLOTS_DIR / fname
    if not path.exists():
        raise HTTPException(404, f"Plot not generated yet: {path}. Run the plot-generation script.")
    return FileResponse(path, media_type="image/png")


# ---------------------------------------------------------------------------
# Data (raw extractions)
# ---------------------------------------------------------------------------


@app.get("/api/data/extractions")
def list_extractions(doi: Optional[str] = None, parameter_id: Optional[str] = None):
    items = state.extractions
    if doi:
        items = [e for e in items if e.get("doi") == doi]
    if parameter_id:
        items = [e for e in items if e.get("parameter_id") == parameter_id]
    return items


@app.get("/api/data/papers")
def list_papers():
    """One row per DOI with extraction count."""

    by_doi: dict[str, int] = {}
    for e in state.extractions:
        d = e.get("doi", "")
        by_doi[d] = by_doi.get(d, 0) + 1
    return [{"doi": d, "extraction_count": c} for d, c in sorted(by_doi.items())]


# ---------------------------------------------------------------------------
# Paper search (PMC E-utilities, no auth)
# ---------------------------------------------------------------------------


_PMC_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_PMC_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


class PaperSearchResult(BaseModel):
    pmc_id: str
    title: str
    journal: str = ""
    year: int = 0
    authors: str = ""


@app.get("/api/papers/search")
def search_papers(q: str = Query(..., min_length=2), retmax: int = 25):
    import requests

    try:
        esearch = requests.get(
            _PMC_ESEARCH,
            params={"db": "pmc", "term": q, "retmode": "json", "retmax": retmax, "sort": "relevance"},
            timeout=30,
        )
        esearch.raise_for_status()
        ids = esearch.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return {"query": q, "results": []}
        esummary = requests.get(
            _PMC_ESUMMARY,
            params={"db": "pmc", "id": ",".join(ids), "retmode": "json"},
            timeout=30,
        )
        esummary.raise_for_status()
        summary = esummary.json().get("result", {})
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"PMC E-utilities error: {exc}")

    out: list[dict[str, Any]] = []
    for uid in ids:
        rec = summary.get(uid)
        if not isinstance(rec, dict):
            continue
        pubdate = rec.get("pubdate", "")
        year = 0
        try:
            year = int(pubdate.split(" ")[0])
        except (ValueError, IndexError):
            pass
        authors = ", ".join(a.get("name", "") for a in rec.get("authors", [])[:3])
        out.append(
            {
                "pmc_id": f"PMC{uid}",
                "title": rec.get("title", ""),
                "journal": rec.get("fulljournalname", rec.get("source", "")),
                "year": year,
                "authors": authors,
            }
        )
    return {"query": q, "results": out}


# ---------------------------------------------------------------------------
# Workflows: recommend + adjust (thin wrappers over simtool.panels.workflows)
# ---------------------------------------------------------------------------


class RecommendRequest(BaseModel):
    required_phenomena: list[str] = []
    excluded_phenomena: list[str] = []
    predictive_priorities: list[str] = []
    time_horizon_days: float = 5.0
    compute_budget_wall_time_hours: Optional[float] = None
    compute_budget_memory_gb: Optional[float] = None
    note: str = ""


@app.post("/api/workflows/recommend")
def workflow_recommend(req: RecommendRequest):
    if state.metamodel is None:
        raise HTTPException(404, "No metamodel loaded.")

    try:
        constraints = UserConstraints(
            predictive_priorities=req.predictive_priorities,
            measurement_capabilities=[],
            required_phenomena=req.required_phenomena,
            excluded_phenomena=req.excluded_phenomena,
            time_horizon_s=float(req.time_horizon_days) * 86400.0,
            compute_budget_wall_time_s=(
                req.compute_budget_wall_time_hours * 3600.0
                if req.compute_budget_wall_time_hours is not None else None
            ),
            compute_budget_memory_gb=req.compute_budget_memory_gb,
            note=req.note,
        )
    except Exception as exc:
        raise HTTPException(400, f"Invalid constraints: {exc}")

    try:
        rec = recommend_model(state.metamodel, constraints)
    except Exception as exc:
        raise HTTPException(500, f"recommend_model failed: {exc}")

    if not state.metamodel.submodels:
        return {
            "recommendation": None,
            "reasoning": [
                {
                    "kind": "note",
                    "note": (
                        "The current meta-model has no submodel hierarchy populated. "
                        "Seed submodels (e.g. well-mixed ODE / 1D biofilm / 3D agent-based) "
                        "before recommendations can be computed."
                    ),
                }
            ],
            "unmet_constraints": [],
            "assumptions_introduced": [],
        }
    return rec.model_dump(mode="json")


class AdjustRequestBody(BaseModel):
    kind: str = "change_parameter"  # change_parameter | add_process | add_entity | change_scope | remove_process
    target_id: str = ""
    spec: dict[str, Any] = {}
    user_note: str = ""


def _default_panel() -> Panel:
    """Lazy-built demo Panel for workflows that need one."""

    if state.default_panel is not None:
        return state.default_panel
    if state.metamodel is None:
        raise HTTPException(404, "No metamodel loaded.")

    from datetime import datetime, timezone

    ir = _build_nitrifying_ir()
    state.default_panel = Panel(
        id="default",
        title="Default demo panel",
        user_id="local",
        created_at=datetime.now(timezone.utc),
        meta_model_id=state.metamodel.id,
        meta_model_version_pin=str(state.metamodel.version),
        derived_ir=ir,
        constraints=UserConstraints(
            predictive_priorities=[],
            measurement_capabilities=[],
            required_phenomena=[],
            excluded_phenomena=[],
            time_horizon_s=5 * 86400.0,
        ),
        parameter_overrides=[],
        run_history=[],
        tags=[],
        collaborators=[],
    )
    return state.default_panel


@app.post("/api/workflows/adjust")
def workflow_adjust(req: AdjustRequestBody):
    if state.metamodel is None:
        raise HTTPException(404, "No metamodel loaded.")

    panel = _default_panel()
    adj = AdjustmentRequest(
        kind=req.kind,
        target_id=req.target_id,
        spec=req.spec,
        user_note=req.user_note,
    )
    try:
        proposal = adjust_model(panel, state.metamodel, adj)
    except Exception as exc:
        raise HTTPException(500, f"adjust_model failed: {exc}")
    return proposal.model_dump(mode="json")


# ---------------------------------------------------------------------------
# IR builder — uses the canonical IR shape (same one panels & plugins expect)
# ---------------------------------------------------------------------------


def _binding(pid: str, unit: str, v: float, **ctx: str) -> ParameterBinding:
    return ParameterBinding(
        parameter_id=pid,
        canonical_unit=unit,
        context_keys=dict(ctx),
        point_estimate=v,
    )


def _build_nitrifying_ir() -> ScientificModel:
    """A small but valid 1-D AOB biofilm IR.

    Used by both the demo plugin (which ignores most of it) and the
    iDynoMiCS 2 plugin (which lowers it to a real protocol.xml). Kept
    purposefully minimal so the smoke path runs in seconds.
    """

    aob = AgentPopulation(id="AOB", name="AOB")
    nh4 = Solute(id="NH4", name="ammonium")
    o2 = Solute(id="O2", name="oxygen")
    dom = SpatialDomain(id="dom", dimensionality=1, extent_um=(64.0,))
    top = Surface(id="top", name="top", axis=0, position="high")
    bot = Surface(id="bot", name="bot", axis=0, position="low")

    growth = MonodGrowthProcess(
        id="AOB_growth",
        growing_entity="AOB",
        consumed_solutes=["NH4", "O2"],
        parameters={
            "mu_max": _binding("mu_max", "1/day", 0.75, species="AOB"),
            "K_s_NH4": _binding("K_s", "mg/L", 0.5, species="AOB", substrate="ammonium"),
            "K_s_O2": _binding("K_s", "mg/L", 0.3, species="AOB", substrate="oxygen"),
            "Y_XS_NH4": _binding(
                "Y_XS", "g_biomass/g_substrate", 0.13,
                species="AOB", substrate="ammonium",
            ),
        },
    )

    bcs = [
        BoundaryCondition(
            id="NH4_top",
            target_entity="NH4",
            surface="top",
            kind="dirichlet",
            value=_binding("S_bulk_initial", "mg/L", 5.0, substrate="ammonium"),
        ),
        BoundaryCondition(
            id="NH4_bot",
            target_entity="NH4",
            surface="bot",
            kind="no_flux",
        ),
    ]
    obs = [
        Observable(
            id="effluent_NH4_mgL",
            name="effluent ammonium",
            kind="scalar_time_series",
            target="NH4_bulk",
            sampling=SamplingSpec(interval_s=3600.0),
        ),
        Observable(
            id="biofilm_thickness_um",
            name="biofilm thickness",
            kind="scalar_time_series",
            target="biofilm_thickness",
            sampling=SamplingSpec(interval_s=3600.0),
        ),
    ]
    return ScientificModel(
        id="demo-nitrifying-biofilm",
        title="Demo nitrifying biofilm",
        domain="microbial_biofilm",
        formalism="agent_based",
        entities=[aob, nh4, o2, dom, top, bot],
        processes=[growth],
        boundary_conditions=bcs,
        initial_conditions=[],
        observables=obs,
        compute=ComputeBudget(time_horizon_s=5 * 86400.0, wall_time_budget_s=600.0),
    )


# ---------------------------------------------------------------------------
# Framework runtimes
# ---------------------------------------------------------------------------


class _IDynomicsRuntime:
    """Adapter that gives the formal iDynoMiCS plugin the same surface as
    DemoPlugin: start_run / stop_run / status / stream_progress / build_output_bundle.

    The formal plugin uses lower → execute → monitor → parse_outputs; we
    drive that lifecycle on a fresh per-run RunLayout under a temp dir.
    """

    framework_id = "idynomics_2"
    framework_name = "iDynoMiCS 2"

    def __init__(self) -> None:
        self._plugin = IDynoMiCS2Plugin()
        self._skill = self._plugin.parse_docs(DocSources(sources=[
            SourceRef(kind=SourceKind.REFERENCE_PROSE, uri="bundled minimal skill"),
        ]))
        self._handles: dict[str, Any] = {}
        self._layouts: dict[str, RunLayout] = {}
        self._statuses: dict[str, RunStatus] = {}
        self._roots: dict[str, Path] = {}

    def is_available(self) -> bool:
        return resolve_jar_path() is not None

    def start_run(self, ir: ScientificModel, *, n_steps: int = 12) -> str:
        if not self.is_available():
            raise RuntimeError(
                "iDynoMiCS 2 jar not configured. Set IDYNOMICS_2_JAR or run "
                "`simtool idynomics set-jar <path>`."
            )
        report = self._plugin.validate_ir(ir, self._skill)
        if not report.ok:
            msgs = "; ".join(i.message for i in report.errors())
            raise RuntimeError(f"IR not expressible by iDynoMiCS 2: {msgs}")

        run_id = uuid.uuid4().hex[:12]
        root = Path(tempfile.mkdtemp(prefix=f"simtool-run-{run_id}-"))
        layout = RunLayout.under(root)
        layout.ensure()
        artifact = self._plugin.lower(ir, self._skill)
        for rel, blob in artifact.files:
            dest = layout.inputs / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(blob)
        try:
            self._plugin.generate_protocol(ir, layout)
        except Exception as exc:  # noqa: BLE001
            logger.warning("generate_protocol failed for %s: %s", run_id, exc)
        handle = self._plugin.execute(artifact, layout)

        self._handles[run_id] = handle
        self._layouts[run_id] = layout
        self._roots[run_id] = root
        self._statuses[run_id] = RunStatus.RUNNING
        return run_id

    def stop_run(self, run_id: str) -> None:
        h = self._handles.get(run_id)
        if h is not None:
            self._plugin.terminate(h, reason="user-requested")
        self._statuses[run_id] = RunStatus.TERMINATED

    def status(self, run_id: str) -> RunStatus:
        return self._statuses.get(run_id, RunStatus.FAILED)

    def stream_progress(self, run_id: str) -> Iterator[ProgressReport]:
        h = self._handles.get(run_id)
        if h is None:
            return
        try:
            for report in self._plugin.monitor(h):
                yield report
            self._statuses[run_id] = RunStatus.SUCCEEDED
        except Exception as exc:  # noqa: BLE001
            logger.exception("iDynoMiCS run %s failed: %s", run_id, exc)
            self._statuses[run_id] = RunStatus.FAILED

    def build_output_bundle(self, run_id: str) -> OutputBundle:
        layout = self._layouts.get(run_id)
        if layout is None:
            raise KeyError(run_id)
        return self._plugin.parse_outputs(layout)


# Runtimes — keyed by framework_id. Demo is always present; iDynoMiCS only
# functional when the jar is configured but the runtime object is always
# available so the UI can show its check status.
_demo_runtime = DemoPlugin()
_idynomics_runtime = _IDynomicsRuntime()


def _runtime_for(framework_id: str):
    if framework_id == "demo":
        return _demo_runtime
    if framework_id == "idynomics_2":
        return _idynomics_runtime
    raise HTTPException(400, f"Unknown framework '{framework_id}'. Use 'demo' or 'idynomics_2'.")


# ---------------------------------------------------------------------------
# Frameworks endpoint
# ---------------------------------------------------------------------------


@app.get("/api/frameworks")
def list_frameworks():
    """Surface available frameworks + a health summary for each.

    The renderer uses this to populate the framework selector and to flag
    when iDynoMiCS isn't configured (so the user sees what to fix).
    """

    idynomics_health = run_health_check()
    return {
        "frameworks": [
            {
                "id": "demo",
                "name": "Demo (synthetic)",
                "available": True,
                "details": "Always available. Synthetic outputs for UI smoke testing.",
            },
            {
                "id": "idynomics_2",
                "name": "iDynoMiCS 2",
                "available": _idynomics_runtime.is_available(),
                "details": idynomics_health.render(),
                "checks": [
                    {"name": c.name, "ok": c.ok, "detail": c.detail}
                    for c in idynomics_health.checks
                ],
            },
        ],
    }


# ---------------------------------------------------------------------------
# Simulation runs (demo OR iDynoMiCS 2)
# ---------------------------------------------------------------------------


class RunStartRequest(BaseModel):
    n_steps: int = 12
    framework: str = "demo"   # "demo" | "idynomics_2"


def _run_in_background(run_id: str, framework_id: str):
    """Consume the runtime's progress iterator into the state buffer."""

    runtime = _runtime_for(framework_id)
    try:
        for report in runtime.stream_progress(run_id):
            state.progress.setdefault(run_id, []).append(report)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Background run %s crashed: %s", run_id, exc)


@app.post("/api/runs")
def start_run(req: RunStartRequest):
    runtime = _runtime_for(req.framework)
    ir = _build_nitrifying_ir()
    try:
        if req.framework == "demo":
            run_id = runtime.start_run(ir, n_steps=req.n_steps)
        else:
            run_id = runtime.start_run(ir, n_steps=req.n_steps)
    except RuntimeError as exc:
        raise HTTPException(400, str(exc))

    state.progress[run_id] = []
    state.run_framework[run_id] = req.framework
    t = threading.Thread(target=_run_in_background, args=(run_id, req.framework), daemon=True)
    state.run_threads[run_id] = t
    t.start()
    return {"run_id": run_id, "framework_id": req.framework}


@app.get("/api/runs/{run_id}/progress")
def get_progress(run_id: str):
    framework = state.run_framework.get(run_id, "demo")
    runtime = _runtime_for(framework)
    reports = state.progress.get(run_id, [])
    status = runtime.status(run_id).value
    return {
        "run_id": run_id,
        "framework_id": framework,
        "status": status,
        "reports": [r.model_dump(mode="json") for r in reports],
    }


@app.get("/api/runs/{run_id}/stream")
def stream_run(run_id: str):
    """Server-Sent Events stream of progress reports."""

    framework = state.run_framework.get(run_id, "demo")
    runtime = _runtime_for(framework)

    def _gen():
        last = 0
        while True:
            reports = state.progress.get(run_id, [])
            while last < len(reports):
                payload = reports[last].model_dump(mode="json")
                yield f"data: {json.dumps(payload)}\n\n"
                last += 1
            status = runtime.status(run_id)
            if status in {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.TERMINATED}:
                yield f"data: {json.dumps({'run_id': run_id, 'status': status.value, 'terminal': True})}\n\n"
                return
            time.sleep(0.2)

    return StreamingResponse(_gen(), media_type="text/event-stream")


@app.post("/api/runs/{run_id}/cancel")
def cancel_run(run_id: str):
    framework = state.run_framework.get(run_id, "demo")
    runtime = _runtime_for(framework)
    runtime.stop_run(run_id)
    return {"run_id": run_id, "cancellation_requested": True}


@app.get("/api/runs/{run_id}/output")
def run_output(run_id: str):
    framework = state.run_framework.get(run_id, "demo")
    runtime = _runtime_for(framework)
    try:
        bundle = runtime.build_output_bundle(run_id)
    except KeyError:
        raise HTTPException(404, f"run_id {run_id} not found")
    analysis = _analyze_output(bundle)
    return {
        "framework_id": framework,
        "bundle": bundle.model_dump(mode="json"),
        "analysis": analysis,
    }


def _analyze_output(bundle) -> dict[str, Any]:
    """Lightweight comparator: for each scalar series, compare the mean of
    the last third to the meta-model's reconciled range for any parameter
    with a matching key name."""

    findings: list[dict[str, Any]] = []
    if state.metamodel is None:
        return {"findings": findings, "note": "no metamodel loaded"}

    ranges: dict[str, tuple[float, float]] = {}
    for rp in state.metamodel.reconciled_parameters:
        b = rp.binding
        vs: list[float] = []
        if b.point_estimate is not None:
            vs = [b.point_estimate]
        elif b.distribution is not None and b.distribution.shape == "empirical" and b.distribution.samples:
            vs = list(b.distribution.samples)
        if vs:
            ranges[rp.parameter_id] = (min(vs), max(vs))

    for obs, series in bundle.scalar_time_series.items():
        if not series:
            continue
        tail = series[-max(1, len(series) // 3):]
        tail_mean = sum(v for _t, v in tail) / len(tail)
        expected = None
        if "NH4" in obs and "K_s" in ranges:
            expected = ranges["K_s"]
        elif "biofilm_thickness" in obs and "D_biofilm" in ranges:
            expected = ranges["D_biofilm"]
        findings.append(
            {
                "observable": obs,
                "tail_mean": round(tail_mean, 4),
                "expected_range": {"lo": expected[0], "hi": expected[1]} if expected else None,
                "within_expected": (
                    expected is not None and expected[0] <= tail_mean <= expected[1]
                ),
            }
        )
    return {"findings": findings}


# ---------------------------------------------------------------------------
# Static frontend (Electron renderer)
# ---------------------------------------------------------------------------

RENDERER_DIR = REPO_ROOT / "app" / "renderer"

if RENDERER_DIR.exists():
    from fastapi.staticfiles import StaticFiles

    app.mount("/ui", StaticFiles(directory=str(RENDERER_DIR), html=True), name="ui")


# ---------------------------------------------------------------------------
# Entrypoint — Electron main spawns this with `python -m simtool.api.server`
# ---------------------------------------------------------------------------


def main() -> int:
    import socket

    import uvicorn

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    print(f"SIMTOOL_API_PORT={port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
