"""Execute an iDynoMiCS 2 run as a subprocess.

We Popen ``java -jar <iDynoMiCS-2.0.jar> <protocol.xml>`` with the run's
layout directory as cwd. The process runs in its own process group so
the harness can terminate it cleanly. stdout + stderr are merged and
written to ``<layout.logs>/idynomics.log`` while being tee'd to the
monitor stream.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from typing import Optional

from simtool.connector.assumptions import AssumptionLedger
from simtool.connector.plugin import LoweredArtifact, RunHandle
from simtool.connector.runs import RunLayout
from simtool.frameworks.idynomics_2.config import (
    IDynoMiCS2Config,
    IDynoMiCS2NotAvailable,
    require_runtime,
)


@dataclass
class IDynoMiCS2Backend:
    """What lives inside ``RunHandle.backend`` for this plugin."""

    config: IDynoMiCS2Config
    process: subprocess.Popen
    protocol_path: Path
    log_path: Path
    line_queue: "Queue[Optional[str]]"  # None sentinel = EOF
    reader_thread: threading.Thread
    started_at: float = field(default_factory=time.time)

    def is_running(self) -> bool:
        return self.process.poll() is None


def execute_artifact(
    artifact: LoweredArtifact,
    layout: RunLayout,
    *,
    runtime: Optional[IDynoMiCS2Config] = None,
) -> RunHandle:
    """Materialize the lowered artifact and launch iDynoMiCS 2.

    Raises:
        IDynoMiCS2NotAvailable - if the jar or Java is missing.
        RuntimeError - if the assumption ledger is not fully approved.
    """
    if not artifact.assumptions.is_ready_to_run():
        raise RuntimeError(
            "cannot execute: assumption ledger not fully approved. "
            f"Blocking: {artifact.assumptions.blocking_reasons()}"
        )

    cfg = runtime or require_runtime()
    layout.ensure()

    for rel, content in artifact.files:
        dest = layout.inputs / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)

    protocol_path = layout.inputs / artifact.entrypoint
    if not protocol_path.is_file():
        raise RuntimeError(
            f"entrypoint '{artifact.entrypoint}' was not written under inputs"
        )

    # iDynoMiCS-2.0.jar declares a relative Class-Path in its manifest
    # (`lib/*/...`) AND loads `default.cfg` from cwd. Only running with
    # cwd = <jar_dir> satisfies both. We rewrite the protocol's
    # outputfolder to an absolute path under the run layout so outputs
    # still land where the rest of the pipeline expects them.
    _rewrite_outputfolder_absolute(protocol_path, layout.outputs)

    log_path = layout.logs / "idynomics.log"
    popen_kwargs = dict(
        args=cfg.as_command(protocol_path.resolve()),
        cwd=str(cfg.jar_path.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if hasattr(os, "setsid"):
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(**popen_kwargs)

    line_queue: "Queue[Optional[str]]" = Queue(maxsize=1024)
    reader = threading.Thread(
        target=_reader_loop,
        args=(proc, line_queue, log_path),
        daemon=True,
    )
    reader.start()

    backend = IDynoMiCS2Backend(
        config=cfg,
        process=proc,
        protocol_path=protocol_path,
        log_path=log_path,
        line_queue=line_queue,
        reader_thread=reader,
    )
    return RunHandle(
        run_id=f"idyno-{proc.pid}-{int(backend.started_at)}",
        layout=layout,
        backend=backend,
    )


def _rewrite_outputfolder_absolute(protocol_path: Path, outputs_dir: Path) -> None:
    """Replace ``outputfolder="..."`` in the protocol.xml with the absolute
    path to the run's outputs directory.

    We use a minimal regex rather than full XML round-trip to avoid
    reformatting the file (iDynoMiCS accepts either; preserving format
    keeps inputs/protocol.xml byte-identical to what lower() emitted
    apart from this one attribute)."""
    outputs_dir.mkdir(parents=True, exist_ok=True)
    import re
    text = protocol_path.read_text(encoding="utf-8")
    abs_out = str(outputs_dir.resolve())
    new_text, n = re.subn(
        r'outputfolder="[^"]*"',
        f'outputfolder="{abs_out}"',
        text,
        count=1,
    )
    if n == 0:
        # No outputfolder attribute at all — inject one on the <simulation> tag.
        new_text = re.sub(
            r"(<simulation\b[^>]*)>",
            lambda m: m.group(1) + f' outputfolder="{abs_out}">',
            text,
            count=1,
        )
    protocol_path.write_text(new_text, encoding="utf-8")


def terminate_handle(handle: RunHandle, reason: str = "") -> None:
    """Request a clean termination. Safe to call repeatedly."""
    backend: IDynoMiCS2Backend = handle.backend
    proc = backend.process
    if proc.poll() is not None:
        return
    try:
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
    except (OSError, ProcessLookupError):
        return
    try:
        proc.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        try:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
        except (OSError, ProcessLookupError):
            pass


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _reader_loop(
    proc: subprocess.Popen, queue: "Queue[Optional[str]]", log_path: Path
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("w", encoding="utf-8") as fh:
            assert proc.stdout is not None
            for line in proc.stdout:
                fh.write(line)
                fh.flush()
                try:
                    queue.put(line.rstrip("\n"), timeout=5.0)
                except Exception:
                    # If the monitor isn't consuming, drop lines rather
                    # than block forever. stdout still lands in the log.
                    pass
    finally:
        proc.wait()
        queue.put(None)
