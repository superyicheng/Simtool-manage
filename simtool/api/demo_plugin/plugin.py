"""Demo plugin — scripted runs with synthetic outputs.

Not a scientific implementation; its only purpose is to exercise the run
+ monitor + analysis pipeline end-to-end inside the desktop app. Swap
for the real iDynoMiCS 2 plugin when available.
"""

from __future__ import annotations

import math
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Iterator, Optional

from simtool.connector.ir import ScientificModel
from simtool.connector.runs import (
    OutputBundle,
    ProgressReport,
    RunStatus,
)


@dataclass
class _RunHandle:
    run_id: str
    ir: ScientificModel
    n_steps: int
    seed: int
    status: RunStatus = RunStatus.RUNNING
    stop_requested: bool = False
    started_at: float = field(default_factory=time.time)


class DemoPlugin:
    framework_id = "demo"
    framework_name = "Demo (synthetic)"

    def __init__(self):
        self._runs: dict[str, _RunHandle] = {}

    def start_run(
        self,
        ir: ScientificModel,
        *,
        n_steps: int = 12,
        seed: Optional[int] = None,
    ) -> str:
        run_id = uuid.uuid4().hex[:12]
        self._runs[run_id] = _RunHandle(
            run_id=run_id,
            ir=ir,
            n_steps=max(4, n_steps),
            seed=seed if seed is not None else random.randint(1, 1 << 30),
        )
        return run_id

    def stop_run(self, run_id: str) -> None:
        handle = self._runs.get(run_id)
        if handle:
            handle.stop_requested = True

    def status(self, run_id: str) -> RunStatus:
        h = self._runs.get(run_id)
        return h.status if h else RunStatus.FAILED

    def stream_progress(self, run_id: str, *, step_seconds: float = 0.35) -> Iterator[ProgressReport]:
        handle = self._runs.get(run_id)
        if handle is None:
            return
        rng = random.Random(handle.seed)
        start = time.time()
        for step in range(handle.n_steps):
            if handle.stop_requested:
                handle.status = RunStatus.FAILED
                yield ProgressReport(
                    run_id=run_id,
                    timestep_index=step,
                    timestep_total=handle.n_steps,
                    wall_time_elapsed_s=time.time() - start,
                    message="Cancelled by user.",
                )
                return
            time.sleep(step_seconds * (1.8 if step == 0 else 1.0))
            yield ProgressReport(
                run_id=run_id,
                timestep_index=step,
                timestep_total=handle.n_steps,
                sim_time_s=step * 3600 * 10,
                sim_time_horizon_s=(handle.n_steps - 1) * 3600 * 10,
                wall_time_elapsed_s=time.time() - start,
                memory_rss_gb=0.12,
                cpu_percent=18 + rng.uniform(-3, 3),
                observables={
                    "biofilm_thickness_um": round(10 + 5 * math.log1p(step) + rng.uniform(-0.5, 0.5), 2),
                    "effluent_NH4_mgL": round(5.0 * math.exp(-0.25 * step) + rng.uniform(-0.05, 0.05), 3),
                },
                message=_stage_for(step, handle.n_steps),
            )
        handle.status = RunStatus.SUCCEEDED
        yield ProgressReport(
            run_id=run_id,
            timestep_index=handle.n_steps,
            timestep_total=handle.n_steps,
            wall_time_elapsed_s=time.time() - start,
            message="Done.",
        )

    def build_output_bundle(self, run_id: str) -> OutputBundle:
        handle = self._runs[run_id]
        rng = random.Random(handle.seed + 7)
        end_time = 5 * 86400  # 5 simulated days in seconds
        n = 60
        times = [end_time * (i / (n - 1)) for i in range(n)]

        def _series(mid, jitter, decay=0.0):
            return [
                (t, max(0.0, mid * math.exp(-decay * (t / 86400)) + rng.uniform(-jitter, jitter)))
                for t in times
            ]

        return OutputBundle(
            run_id=run_id,
            scalar_time_series={
                "effluent_NH4_mgL": _series(0.4, 0.03),
                "effluent_NO2_mgL": _series(0.05, 0.01),
                "effluent_NO3_mgL": _series(5.2, 0.2),
                "biofilm_thickness_um": _series(55.0, 2.0),
                "AOB_fraction": _series(0.42, 0.02),
                "NOB_fraction": _series(0.18, 0.015),
            },
            notes="Synthetic demo output; values drawn near literature ranges for nitrifying biofilms.",
        )


def _stage_for(step: int, total: int) -> str:
    pct = step / max(1, total - 1)
    if pct < 0.15:
        return "Initializing spatial domain + seeding biomass..."
    if pct < 0.35:
        return "Equilibrating substrate fields..."
    if pct < 0.65:
        return "Integrating growth + detachment..."
    if pct < 0.9:
        return "Approaching pseudo-steady state..."
    return "Finalizing outputs..."
