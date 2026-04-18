"""Stream iDynoMiCS 2 progress as structured ProgressReports.

iDynoMiCS writes progress to stdout in lines like:

    ----> Computing time step X at time Y
    time step Y ended at wall time Z

We parse what we can and synthesize a ProgressReport per timestep, plus
a closing one when the process exits. Unrecognized lines are kept in
``message`` on the latest report so the monitor never goes silent.
"""

from __future__ import annotations

import re
import time
from typing import Iterator, Optional

from simtool.connector.plugin import RunHandle
from simtool.connector.runs import ProgressReport
from simtool.frameworks.idynomics_2.execute import IDynoMiCS2Backend


# iDynoMiCS 2 progress line: `[HH:MM] #N time: T step: S end: E`
_STEP_RE = re.compile(
    r"#(?P<step>\d+)\s+time:\s+(?P<time>[\d.eE+-]+)\s+"
    r"step:\s+(?P<dt>[\d.eE+-]+)\s+end:\s+(?P<end>[\d.eE+-]+)"
)
# Agent-count line: `<compartment> contains N agents`
_AGENTS_RE = re.compile(r"contains\s+(?P<n>\d+)\s+agents")
# Generic time reference, kept as fallback.
_SIM_TIME_RE = re.compile(
    r"time\s+(?P<t>\d+(?:\.\d+)?)\s*(?:d|day|h|hour|s|sec)?", re.IGNORECASE
)


def monitor_handle(
    handle: RunHandle,
    *,
    heartbeat_s: float = 5.0,
) -> Iterator[ProgressReport]:
    """Yield a ProgressReport per parsed line (or per heartbeat when no
    output arrives for ``heartbeat_s`` seconds)."""
    backend: IDynoMiCS2Backend = handle.backend
    queue = backend.line_queue
    last_report = ProgressReport(run_id=handle.run_id, message="launched")
    yield last_report
    last_emit = time.monotonic()

    while True:
        try:
            line = queue.get(timeout=heartbeat_s)
        except Exception:
            # heartbeat: the process may still be running but quiet.
            if backend.process.poll() is not None:
                break
            hb = _make_heartbeat(handle, backend)
            yield hb
            last_emit = time.monotonic()
            continue
        if line is None:  # EOF sentinel from reader thread
            break
        report = _parse_line(handle, backend, line, last_report)
        if report is not None:
            last_report = report
            yield report
            last_emit = time.monotonic()
        elif time.monotonic() - last_emit >= heartbeat_s:
            hb = _make_heartbeat(handle, backend, message=line)
            yield hb
            last_emit = time.monotonic()

    # Final summary.
    final = ProgressReport(
        run_id=handle.run_id,
        wall_time_elapsed_s=time.time() - backend.started_at,
        message=(
            f"process exited (code {backend.process.returncode})"
        ),
    )
    yield final


def _parse_line(
    handle: RunHandle,
    backend: IDynoMiCS2Backend,
    line: str,
    last: ProgressReport,
) -> Optional[ProgressReport]:
    step_m = _STEP_RE.search(line)
    agents_m = _AGENTS_RE.search(line)
    time_m = _SIM_TIME_RE.search(line) if not step_m else None
    if not (step_m or agents_m or time_m):
        return None
    idx = last.timestep_index
    total = last.timestep_total
    sim_time_s: Optional[float] = last.sim_time_s
    sim_horizon_s: Optional[float] = last.sim_time_horizon_s
    observables = dict(last.observables)

    if step_m is not None:
        idx = int(step_m.group("step"))
        sim_time_s = float(step_m.group("time"))
        dt = float(step_m.group("dt"))
        end = float(step_m.group("end"))
        sim_horizon_s = end
        if dt > 0:
            total = int(end / dt)
    elif time_m is not None:
        t_val = float(time_m.group("t"))
        unit_chunk = line[time_m.end("t"):].lstrip()
        if unit_chunk.startswith(("d", "day")):
            sim_time_s = t_val * 86400.0
        elif unit_chunk.startswith(("h", "hour")):
            sim_time_s = t_val * 3600.0
        elif unit_chunk.startswith(("s",)):
            sim_time_s = t_val

    if agents_m is not None:
        observables["n_agents"] = float(agents_m.group("n"))

    return ProgressReport(
        run_id=handle.run_id,
        sim_time_s=sim_time_s,
        sim_time_horizon_s=sim_horizon_s,
        timestep_index=idx,
        timestep_total=total,
        wall_time_elapsed_s=time.time() - backend.started_at,
        observables=observables,
        message=line.strip(),
    )


def _make_heartbeat(
    handle: RunHandle,
    backend: IDynoMiCS2Backend,
    *, message: str = "",
) -> ProgressReport:
    return ProgressReport(
        run_id=handle.run_id,
        wall_time_elapsed_s=time.time() - backend.started_at,
        message=message or "(running; no new output)",
    )
