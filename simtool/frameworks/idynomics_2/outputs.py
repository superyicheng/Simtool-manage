"""Read an iDynoMiCS 2 output directory into an OutputBundle.

iDynoMiCS writes snapshots into ``<outputfolder>/`` as XML and optionally
CSV files. We don't assume a fixed schema — we scan the directory, pick
out the easiest scalar time series (if any), and keep every other file
as a ``spatial_field_paths`` reference for downstream tooling.

iDynoMiCS's default storage config (``ignore_protocol_out=TRUE``) makes
it ignore the protocol's outputfolder and write under
``<jar_dir>/results/<timestamp>_<simulation_name>/`` regardless. We
post-locate that directory and copy it into ``layout.outputs`` so the
rest of the pipeline sees outputs in the canonical place.

Tolerates incomplete/failed runs: missing or partial outputs produce
empty channels rather than raising.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Optional

from simtool.connector.runs import OutputBundle, RunLayout


def harvest_jar_results(
    jar_dir: Path, simulation_name: str, layout: RunLayout,
    *, cutoff_mtime: Optional[float] = None,
) -> Optional[Path]:
    """If iDynoMiCS wrote its outputs under ``jar_dir/results/``, copy the
    newest matching directory into ``layout.outputs``.

    Returns the path to the copied directory under ``layout.outputs``, or
    None if nothing matched (run failed before writing anything, or
    user-provided storage.cfg redirected outputs somewhere else).
    """
    results_root = jar_dir / "results"
    if not results_root.is_dir():
        return None
    candidates: list[Path] = []
    for child in results_root.iterdir():
        if not child.is_dir():
            continue
        if simulation_name not in child.name:
            continue
        if cutoff_mtime is not None and child.stat().st_mtime < cutoff_mtime:
            continue
        candidates.append(child)
    if not candidates:
        return None
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    dest = layout.outputs / newest.name
    if dest.exists():
        return dest
    layout.outputs.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(newest, dest)
    except OSError:
        return None
    return dest


def extract_simulation_name(protocol_xml_path: Path) -> Optional[str]:
    if not protocol_xml_path.is_file():
        return None
    try:
        text = protocol_xml_path.read_text(encoding="utf-8")
    except OSError:
        return None
    m = re.search(r'<simulation\b[^>]*\bname="([^"]+)"', text)
    return m.group(1) if m else None


def parse_output_dir(layout: RunLayout, run_id: Optional[str] = None) -> OutputBundle:
    bundle = OutputBundle(run_id=run_id or layout.root.name)
    outputs_dir = layout.outputs
    if not outputs_dir.is_dir():
        return bundle

    # Scan for CSV time series.
    for csv_path in outputs_dir.rglob("*.csv"):
        series = _read_csv_scalar(csv_path)
        if series:
            key = csv_path.stem
            bundle.scalar_time_series[key] = series

    # Every XML/snapshot file gets recorded as a spatial field reference.
    for ext in ("*.xml", "*.vtu", "*.vtk", "*.dat", "*.nc"):
        for path in outputs_dir.rglob(ext):
            try:
                rel = str(path.relative_to(layout.root))
            except ValueError:
                rel = str(path)
            bundle.spatial_field_paths[path.stem] = rel

    if not bundle.scalar_time_series and not bundle.spatial_field_paths:
        bundle.notes = "no output files found under outputs/"

    return bundle


def _read_csv_scalar(path: Path) -> list[tuple[float, float]]:
    """Best-effort CSV reader. Accepts files with a header row naming
    something like ``time`` and one other numeric column."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return []
    header = [h.strip().lower() for h in _split_csv(lines[0])]
    # Find a time-ish column + a value-ish column.
    t_idx: Optional[int] = None
    v_idx: Optional[int] = None
    for i, h in enumerate(header):
        if t_idx is None and ("time" in h or h in {"t", "t_s", "t_d"}):
            t_idx = i
        elif v_idx is None and i != t_idx:
            v_idx = i
    if t_idx is None or v_idx is None:
        # fall back to first two columns
        if len(header) >= 2:
            t_idx, v_idx = 0, 1
        else:
            return []
    series: list[tuple[float, float]] = []
    for ln in lines[1:]:
        cols = _split_csv(ln)
        if len(cols) <= max(t_idx, v_idx):
            continue
        try:
            t = float(cols[t_idx])
            v = float(cols[v_idx])
        except ValueError:
            continue
        series.append((t, v))
    return series


def _split_csv(line: str) -> list[str]:
    # Tolerate comma or whitespace separation; no fancy quoting.
    if "," in line:
        return [c.strip() for c in line.split(",")]
    return line.split()
