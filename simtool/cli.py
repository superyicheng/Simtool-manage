"""Minimal CLI for simtool.

Subcommands:
    simtool config set-key / clear-key / status   — API-key management
    simtool fetch [...]                           — PDF fetching pipeline
"""

from __future__ import annotations

import argparse
import getpass
import logging
import sys
from pathlib import Path

from simtool import settings


# --- `config` subcommands ---------------------------------------------------


def _cmd_config_set_key(_args: argparse.Namespace) -> int:
    try:
        key = getpass.getpass("Anthropic API key (input hidden): ")
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.", file=sys.stderr)
        return 130
    try:
        settings.set_anthropic_api_key(key)
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print("Stored in OS credential store.")
    return 0


def _cmd_config_clear_key(_args: argparse.Namespace) -> int:
    removed = settings.clear_anthropic_api_key()
    print("Removed stored key." if removed else "No stored key to remove.")
    return 0


def _cmd_config_status(_args: argparse.Namespace) -> int:
    s = settings.status()
    print(f"Keyring available:      {s['keyring_available']}")
    print(f"python-dotenv available:{s['dotenv_available']}")
    print(f"Env var set:            {s['has_env_key']}")
    print(f"Stored in keyring:      {s['has_stored_key']}")
    suffix = s["effective_suffix"]
    if suffix:
        print(f"Effective key ends in:  ...{suffix}")
    else:
        print("Effective key ends in:  (none configured)")
    return 0


# --- `fetch` ----------------------------------------------------------------


def _cmd_fetch(args: argparse.Namespace) -> int:
    # Local imports to keep the CLI cheap when only config is used.
    from simtool.corpus.fetcher import fetch_corpus, default_contact_email
    from simtool.corpus.manifest import AccessStatus, load_log
    from simtool.corpus.pmc_fetcher import PmcFetcher

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        return 1
    log = load_log(manifest_path)

    output_dir = Path(args.output)
    contact_email = args.email or default_contact_email()

    pmc_fetcher = PmcFetcher(
        output_dir=output_dir,
        delay_seconds=args.delay,
        timeout_seconds=args.timeout,
        contact_email=contact_email,
    )
    registry = {
        AccessStatus.OPEN_PMC: pmc_fetcher,
        # open_biorxiv / open_other / institutional fetchers plug in here later.
    }

    summary = fetch_corpus(
        log,
        output_dir=output_dir,
        fetcher_registry=registry,
        max_fetches=args.max,
        skip_existing=not args.no_skip_existing,
        show_progress=not args.no_progress,
    )

    print()
    print("Fetch summary")
    print("-" * 40)
    for status, count in sorted(summary.by_status().items()):
        print(f"  {status:<26}{count}")
    print("-" * 40)
    print(f"Output dir: {output_dir.resolve()}")
    failures = [
        r for r in summary.results if r.status.value.startswith("failed_")
    ]
    if failures:
        print(f"\nFailures ({len(failures)}):")
        for r in failures:
            print(f"  {r.doi}  [{r.status.value}]  {r.reason}")
    return 0


# --- parser wiring ----------------------------------------------------------


def _cmd_extract(args: argparse.Namespace) -> int:
    import json as _json

    from simtool.extractors import ChemostatExtractor, ExtractorConfig
    from simtool.extractors.base import BaseExtractor, MethodProfile
    from simtool.extractors.client import default_client
    from simtool.schema.parameter_record import MeasurementMethod

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        return 1

    method = MeasurementMethod(args.method)
    extractor_cls: type[BaseExtractor]
    if method == MeasurementMethod.CHEMOSTAT:
        extractor_cls = ChemostatExtractor
    else:
        # Build a generic subclass for any other method on the fly.
        class _Generic(BaseExtractor):
            method_profile = MethodProfile(method=method)

        extractor_cls = _Generic

    client = default_client()
    config = ExtractorConfig(model=args.model)
    extractor = extractor_cls(client=client, config=config)
    result = extractor.extract(pdf_path, doi_hint=args.doi)

    payload = {
        "pdf": str(result.pdf_path),
        "method": result.method.value,
        "stop_reason": result.stop_reason,
        "n_extractions": len(result.extractions),
        "extractions": [r.model_dump(mode="json") for r in result.extractions],
    }
    print(_json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _cmd_extract_corpus(args: argparse.Namespace) -> int:
    """Run the extractor against every downloaded PDF in the manifest."""

    import json as _json

    from simtool.corpus.manifest import load_log
    from simtool.extractors import ChemostatExtractor, ExtractorConfig
    from simtool.extractors.client import default_client

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        return 1
    prisma = load_log(manifest_path)

    download_log_path = Path(args.pdfs_dir) / "download_log.json"
    if not download_log_path.exists():
        print(
            f"No download log at {download_log_path}. Run `simtool fetch` first.",
            file=sys.stderr,
        )
        return 1
    with open(download_log_path, "r", encoding="utf-8") as f:
        fetched = _json.load(f).get("fetched", {})

    entries_by_doi = {e.doi: e for e in prisma.entries}
    doi_pdf_pairs: list[tuple[str, Path]] = []
    for doi, pdf_str in fetched.items():
        pdf = Path(pdf_str)
        if not pdf.exists():
            continue
        if doi not in entries_by_doi:
            continue
        doi_pdf_pairs.append((doi, pdf))

    if args.max is not None:
        doi_pdf_pairs = doi_pdf_pairs[: args.max]

    if not doi_pdf_pairs:
        print("Nothing to extract. Check download_log.json and manifest.")
        return 1

    client = default_client()
    config = ExtractorConfig(model=args.model)
    extractor = ChemostatExtractor(client=client, config=config)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: preserve any prior output for DOIs we've already processed.
    existing: dict[str, list] = {}
    if out_path.exists() and not args.no_skip_existing:
        try:
            existing_payload = _json.loads(out_path.read_text())
            for rec in existing_payload.get("extractions", []):
                existing.setdefault(rec["doi"], []).append(rec)
        except Exception:
            pass

    all_items: list[dict] = []
    # Re-emit previously extracted ones (resume)
    for doi, items in existing.items():
        all_items.extend(items)
    processed = set(existing.keys())

    for doi, pdf in doi_pdf_pairs:
        if doi in processed and not args.no_skip_existing:
            print(f"[skip] {doi} (already in output)")
            continue
        print(f"[extract] {doi} -> {pdf.name}")
        result = extractor.extract(pdf, doi_hint=doi)
        for rec in result.extractions:
            payload = rec.model_dump(mode="json")
            payload["doi"] = doi
            all_items.append(payload)
        # Incremental save — a crash mid-run doesn't lose everything.
        out_path.write_text(
            _json.dumps(
                {
                    "model": config.model,
                    "extractor_method": extractor.method_profile.method.value,
                    "extractions": all_items,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        processed.add(doi)

    print(f"\n{len(all_items)} extractions saved to {out_path}")
    return 0


def _cmd_build_metamodel(args: argparse.Namespace) -> int:
    """Reconcile saved extractions into a MetaModel JSON artifact."""

    import json as _json

    from simtool.extractors.schemas import RawExtraction
    from simtool.metamodel.library import MetaModel
    from simtool.metamodel.reconciler import ExtractionWithDoi, reconcile
    from simtool.metamodel.versioning import SemVer

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    extractions_path = Path(args.extractions)
    if not extractions_path.exists():
        print(
            f"Extractions file not found: {extractions_path}. Run `simtool extract-corpus` first.",
            file=sys.stderr,
        )
        return 1
    payload = _json.loads(extractions_path.read_text())
    raw_items = payload.get("extractions", [])

    records: list[ExtractionWithDoi] = []
    for item in raw_items:
        doi = item.pop("doi", None)
        if doi is None:
            continue
        records.append(ExtractionWithDoi(extraction=RawExtraction.model_validate(item), doi=doi))

    reconciled, summary = reconcile(records)

    mm = MetaModel(
        id=args.id,
        title=args.title,
        scientific_domain=args.domain,
        version=SemVer(major=0, minor=1, patch=0),
        reconciled_parameters=reconciled,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(mm.model_dump_json(indent=2))

    print(f"MetaModel written to {out_path}")
    print("-" * 40)
    print(f"  Input records:       {summary.input_count}")
    print(f"  Harmonized:          {summary.harmonized_count}")
    print(f"  Dropped (units):     {summary.dropped_harmonization}")
    print(f"  Dropped (range):     {summary.dropped_range_check}")
    print(f"  Reconciled params:   {summary.reconciled_parameters}")
    print(f"  Conflict params:     {summary.conflict_parameters}")
    if reconciled:
        print()
        print("Parameters:")
        for rp in reconciled:
            ctx = ", ".join(f"{k}={v}" for k, v in rp.context_keys.items()) or "(no-context)"
            if rp.binding.point_estimate is not None:
                val_desc = f"{rp.binding.point_estimate:.4g} {rp.binding.canonical_unit}"
            else:
                samples = rp.binding.distribution.samples or []
                val_desc = (
                    f"n={len(samples)} empirical ("
                    f"min {min(samples):.3g}, max {max(samples):.3g}) {rp.binding.canonical_unit}"
                )
            flag = " ⚠ conflict" if rp.conflict_flags else ""
            print(f"  {rp.parameter_id:<20} [{ctx}] = {val_desc}  quality={rp.quality_rating.value}{flag}")
    return 0


# --- `idynomics` subcommands -----------------------------------------------


def _cmd_idynomics_check(_args: argparse.Namespace) -> int:
    from simtool.frameworks.idynomics_2 import run_health_check

    report = run_health_check()
    print(report.render())
    return 0 if report.ok else 1


def _cmd_idynomics_set_jar(args: argparse.Namespace) -> int:
    from simtool.frameworks.idynomics_2 import save_jar_path

    try:
        cfg_path = save_jar_path(Path(args.path))
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Stored jar path in {cfg_path}")
    return 0


def _cmd_idynomics_run(args: argparse.Namespace) -> int:
    """One-shot execution of an existing protocol.xml — sanity-check that
    the runtime is wired up end-to-end without touching the IR/panel path."""
    from simtool.frameworks.idynomics_2 import (
        IDynoMiCS2NotAvailable,
        IDynoMiCS2Plugin,
    )
    from simtool.connector.assumptions import AssumptionLedger
    from simtool.connector.plugin import LoweredArtifact
    from simtool.connector.runs import RunLayout

    protocol_path = Path(args.protocol)
    if not protocol_path.is_file():
        print(f"Protocol not found: {protocol_path}", file=sys.stderr)
        return 1

    run_root = Path(args.run_root).expanduser()
    layout = RunLayout.under(run_root / f"adhoc-{protocol_path.stem}")
    layout.ensure()

    artifact = LoweredArtifact(
        entrypoint=protocol_path.name,
        files=[(protocol_path.name, protocol_path.read_bytes())],
        assumptions=AssumptionLedger(
            ir_id=f"adhoc-{protocol_path.stem}",
            framework="idynomics_2",
            framework_version="2.0.0",
        ),
    )

    plugin = IDynoMiCS2Plugin()
    try:
        handle = plugin.execute(artifact, layout)
    except IDynoMiCS2NotAvailable as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(f"Launched iDynoMiCS against {protocol_path}")
    print(f"Run layout: {layout.root}")
    print("Streaming progress (Ctrl-C to stop monitoring, process keeps running):\n")
    for report in plugin.monitor(handle):
        if report.timestep_index is not None:
            suffix = ""
            if report.observables:
                suffix = " " + " ".join(
                    f"{k}={v:g}" for k, v in report.observables.items()
                )
            print(
                f"  step {report.timestep_index}"
                + (f"/{report.timestep_total}" if report.timestep_total else "")
                + f" t={report.sim_time_s:g}s"
                + suffix
            )
        elif report.message:
            print(f"  {report.message[:120]}")
    bundle = plugin.parse_outputs(layout)
    print("\n--- outputs ---")
    for k, series in bundle.scalar_time_series.items():
        print(f"  {k}: {len(series)} points")
    for k in bundle.spatial_field_paths:
        print(f"  {k}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="simtool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # config
    config = subparsers.add_parser("config", help="Manage API keys and settings.")
    config_sub = config.add_subparsers(dest="subcommand", required=True)

    p_set = config_sub.add_parser("set-key", help="Store Anthropic API key in OS credential store.")
    p_set.set_defaults(func=_cmd_config_set_key)

    p_clear = config_sub.add_parser("clear-key", help="Remove any stored Anthropic API key.")
    p_clear.set_defaults(func=_cmd_config_clear_key)

    p_status = config_sub.add_parser("status", help="Show where the API key is resolved from.")
    p_status.set_defaults(func=_cmd_config_status)

    # fetch
    fetch = subparsers.add_parser(
        "fetch",
        help="Fetch open-access PDFs listed in the corpus manifest.",
    )
    fetch.add_argument(
        "--manifest",
        default="simtool/corpus/seed_nitrifying_biofilm.yaml",
        help="Path to the corpus manifest YAML.",
    )
    fetch.add_argument(
        "--output",
        default="data/pdfs",
        help="Directory to write PDFs to (default: data/pdfs).",
    )
    fetch.add_argument(
        "--max",
        type=int,
        default=None,
        help="Stop after N attempted fetches (default: all entries).",
    )
    fetch.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Polite-pool delay between downloads in seconds (default: 1.0).",
    )
    fetch.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Per-request timeout in seconds (default: 60).",
    )
    fetch.add_argument(
        "--email",
        default=None,
        help="Contact email for the polite-pool User-Agent (or set SIMTOOL_CONTACT_EMAIL).",
    )
    fetch.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-download even if entries are in the download log.",
    )
    fetch.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar.",
    )
    fetch.set_defaults(func=_cmd_fetch)

    # extract
    extract = subparsers.add_parser(
        "extract",
        help="Run the parameter extractor on a single PDF using Anthropic Claude.",
    )
    extract.add_argument("--pdf", required=True, help="Path to the PDF to extract from.")
    extract.add_argument(
        "--method",
        default="chemostat",
        choices=[m.value for m in __import__("simtool.schema.parameter_record", fromlist=["MeasurementMethod"]).MeasurementMethod],
        help="Measurement method to focus on (default: chemostat).",
    )
    extract.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="Anthropic model id (default: claude-sonnet-4-6).",
    )
    extract.add_argument(
        "--doi",
        default=None,
        help="Optional DOI hint for the model's bookkeeping.",
    )
    extract.set_defaults(func=_cmd_extract)

    # extract-corpus
    extract_corpus = subparsers.add_parser(
        "extract-corpus",
        help="Run the extractor on every downloaded PDF listed in download_log.json.",
    )
    extract_corpus.add_argument(
        "--manifest",
        default="simtool/corpus/seed_nitrifying_biofilm.yaml",
        help="Corpus manifest YAML (default: nitrifying biofilm seed).",
    )
    extract_corpus.add_argument(
        "--pdfs-dir",
        default="data/pdfs",
        help="Directory holding downloaded PDFs + download_log.json (default: data/pdfs).",
    )
    extract_corpus.add_argument(
        "--output",
        default="data/extractions/nitrifying_biofilm.json",
        help="Where to write extraction JSON (default: data/extractions/nitrifying_biofilm.json).",
    )
    extract_corpus.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="Anthropic model id.",
    )
    extract_corpus.add_argument(
        "--max",
        type=int,
        default=None,
        help="Process at most N PDFs (for cost-controlled dry runs).",
    )
    extract_corpus.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-extract DOIs already present in the output file.",
    )
    extract_corpus.set_defaults(func=_cmd_extract_corpus)

    # build-metamodel
    build_mm = subparsers.add_parser(
        "build-metamodel",
        help="Reconcile extractions into a MetaModel artifact.",
    )
    build_mm.add_argument(
        "--extractions",
        default="data/extractions/nitrifying_biofilm.json",
        help="Input extraction JSON (from `simtool extract-corpus`).",
    )
    build_mm.add_argument(
        "--output",
        default="data/metamodel/nitrifying_biofilm_v0_1_0.json",
        help="Where to write the MetaModel JSON.",
    )
    build_mm.add_argument(
        "--id",
        default="nitrifying_biofilm",
        help="MetaModel id (stable slug).",
    )
    build_mm.add_argument(
        "--title",
        default="Nitrifying biofilm reference model (v0.1, auto-extracted)",
        help="Human-readable title.",
    )
    build_mm.add_argument(
        "--domain",
        default="microbial_biofilm",
        help="Scientific domain label.",
    )
    build_mm.set_defaults(func=_cmd_build_metamodel)

    # idynomics
    idyn = subparsers.add_parser(
        "idynomics",
        help="iDynoMiCS 2 plugin setup + runtime checks.",
    )
    idyn_sub = idyn.add_subparsers(dest="subcommand", required=True)

    p_check = idyn_sub.add_parser(
        "check",
        help="Report jar + Java + sibling-file status. Exit 0 if all green.",
    )
    p_check.set_defaults(func=_cmd_idynomics_check)

    p_setjar = idyn_sub.add_parser(
        "set-jar",
        help="Persist the iDynoMiCS 2 jar path in ~/.simtool/idynomics_2.json.",
    )
    p_setjar.add_argument("path", help="Path to iDynoMiCS-2.0.jar.")
    p_setjar.set_defaults(func=_cmd_idynomics_set_jar)

    p_run = idyn_sub.add_parser(
        "run",
        help="Run an existing protocol.xml under iDynoMiCS 2 and stream progress.",
    )
    p_run.add_argument("protocol", help="Path to an iDynoMiCS protocol.xml.")
    p_run.add_argument(
        "--run-root", default="~/simtool/runs",
        help="Where to materialize the run layout (default: ~/simtool/runs).",
    )
    p_run.set_defaults(func=_cmd_idynomics_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
