"""Image renderers for Ara deployment.

Ara accepts image attachments. These renderers emit PNG bytes (or write
to a path) so the agent can attach them to messages.

matplotlib is a lazy import — text renderers work without it. If
matplotlib isn't installed, image functions raise a clear
``ImageRenderingNotAvailable`` with install guidance.

Design: every renderer takes a plain-typed payload (dict, list, tuple),
not our pydantic types. Callers extract the fields they want from
``OutputBundle`` / ``Distribution`` / etc. and pass them in. This keeps
the renderers decoupled from the rest of the package and trivially
testable.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional, Union

from simtool.connector.ir import Distribution
from simtool.connector.runs import OutputBundle
from simtool.metamodel import ReconciledParameter


class ImageRenderingNotAvailable(RuntimeError):
    """Raised when matplotlib cannot be imported."""


def _require_mpl():
    try:
        import matplotlib  # noqa: F401
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise ImageRenderingNotAvailable(
            "image rendering needs matplotlib. Install with "
            "`pip install 'simtool-manage[ara]'` or `pip install matplotlib`."
        ) from exc


def _save(fig, out: Optional[Union[Path, str]]) -> bytes:
    """Save a matplotlib figure to PNG — to path or to bytes."""
    if out is not None:
        out = Path(out)
        fig.savefig(out, dpi=120, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
        return out.read_bytes()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------


def render_scalar_timeseries(
    series: list[tuple[float, float]],
    *,
    title: str = "",
    x_label: str = "time (s)",
    y_label: str = "value",
    reference_band: Optional[tuple[float, float]] = None,
    out: Optional[Union[Path, str]] = None,
) -> bytes:
    """Render one scalar series. ``reference_band`` adds a shaded band
    (e.g. meta-model's reconciled range) behind the curve.

    Returns the PNG bytes. If ``out`` is given, also writes to disk.
    """
    plt = _require_mpl()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    if series:
        xs, ys = zip(*series)
        ax.plot(xs, ys, marker="o", linewidth=1.5)
    if reference_band is not None:
        lo, hi = reference_band
        ax.axhspan(lo, hi, alpha=0.15, label="literature range")
        ax.legend(loc="best")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return _save(fig, out)


def render_output_bundle_overview(
    bundle: OutputBundle,
    *,
    out: Optional[Union[Path, str]] = None,
) -> bytes:
    """One figure with up to 4 scalar series from an OutputBundle."""
    plt = _require_mpl()
    items = list(bundle.scalar_time_series.items())[:4]
    n = max(1, len(items))
    fig, axes = plt.subplots(n, 1, figsize=(6, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, (key, series) in zip(axes, items):
        if series:
            xs, ys = zip(*series)
            ax.plot(xs, ys, marker="o", linewidth=1.5)
        ax.set_ylabel(key)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"run {bundle.run_id}")
    return _save(fig, out)


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------


def render_distribution(
    distribution: Distribution,
    *,
    title: str = "",
    out: Optional[Union[Path, str]] = None,
) -> bytes:
    """Visualize an IR Distribution — empirical samples become a histogram,
    parametric shapes render their density."""
    plt = _require_mpl()
    fig, ax = plt.subplots(figsize=(6, 3.5))

    if distribution.shape == "empirical" and distribution.samples:
        ax.hist(distribution.samples, bins=max(5, len(distribution.samples) // 2))
        ax.set_xlabel("value")
        ax.set_ylabel("count")
    elif distribution.shape == "uniform":
        lo = distribution.params["low"]
        hi = distribution.params["high"]
        ax.axvspan(lo, hi, alpha=0.3)
        ax.set_xlim(lo - (hi - lo) * 0.2, hi + (hi - lo) * 0.2)
        ax.set_ylabel("density (uniform)")
    elif distribution.shape in {"normal", "lognormal", "triangular"}:
        import math
        p = distribution.params
        if distribution.shape == "normal":
            mean, std = p["mean"], p["stddev"]
            xs = [mean - 4 * std + i * (8 * std) / 200 for i in range(201)]
            ys = [
                math.exp(-0.5 * ((x - mean) / std) ** 2)
                / (std * math.sqrt(2 * math.pi))
                for x in xs
            ]
        elif distribution.shape == "lognormal":
            mu, sigma = p["mu"], p["sigma"]
            xs = [
                math.exp(mu - 4 * sigma + i * (8 * sigma) / 200)
                for i in range(201)
            ]
            ys = [
                (1 / (x * sigma * math.sqrt(2 * math.pi)))
                * math.exp(-0.5 * ((math.log(x) - mu) / sigma) ** 2)
                if x > 0 else 0.0
                for x in xs
            ]
        else:  # triangular
            low, mode, high = p["low"], p["mode"], p["high"]
            xs = [low + i * (high - low) / 200 for i in range(201)]
            ys = []
            for x in xs:
                if x < low or x > high:
                    ys.append(0.0)
                elif x <= mode:
                    ys.append(
                        2.0 * (x - low) / ((high - low) * (mode - low))
                        if mode > low else 0.0
                    )
                else:
                    ys.append(
                        2.0 * (high - x) / ((high - low) * (high - mode))
                        if high > mode else 0.0
                    )
        ax.plot(xs, ys)
        ax.set_xlabel("value")
        ax.set_ylabel("density")
    else:
        ax.text(0.5, 0.5, f"shape {distribution.shape} (not plottable)",
                ha="center", va="center", transform=ax.transAxes)

    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return _save(fig, out)


def render_reconciled_parameter(
    rp: ReconciledParameter,
    *,
    out: Optional[Union[Path, str]] = None,
) -> bytes:
    """Render a ReconciledParameter's binding as a distribution chart or
    a single-value marker."""
    if rp.binding.distribution is not None:
        title = f"{rp.parameter_id} ({', '.join(rp.context_keys.values())})"
        return render_distribution(rp.binding.distribution, title=title, out=out)
    # Point estimate — render as a single vertical marker.
    plt = _require_mpl()
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.axvline(rp.binding.point_estimate, linewidth=2)
    ax.set_xlabel(f"value ({rp.binding.canonical_unit})")
    ax.set_title(
        f"{rp.parameter_id} ({', '.join(rp.context_keys.values())}) — "
        f"point estimate"
    )
    ax.set_yticks([])
    return _save(fig, out)
