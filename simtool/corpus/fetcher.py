"""PDF fetcher base class and PMC implementation.

Patterned after the OpenAlex Data extractor's `PDFDownloader`
(see /Users/yichengli/.../research-search-engine/OpenAlex data/download_pdfs.py):

  - session with `urllib3.util.retry.Retry` on 429/5xx, backoff_factor=2
  - resumable JSON download log (skip already-fetched entries)
  - polite delay between downloads
  - atomic writes (stream to .part, verify size, rename)
  - content-type sanity check
  - per-source subclass overrides URL resolution

Adapted for our corpus: dispatches by `CorpusEntry.access` — PMC entries use
`PmcFetcher`; paywalled entries are logged as skipped, never attempted.
"""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from simtool.corpus.manifest import AccessStatus, CorpusEntry, PrismaLog

logger = logging.getLogger(__name__)


class FetchStatus(str, Enum):
    DOWNLOADED = "downloaded"
    SKIPPED_EXISTING = "skipped_existing"
    SKIPPED_PAYWALLED = "skipped_paywalled"
    SKIPPED_VERIFY_DOI = "skipped_verify_doi"
    FAILED_NO_URL = "failed_no_url"
    FAILED_HTTP = "failed_http"
    FAILED_CONTENT = "failed_content"
    FAILED_TOO_SMALL = "failed_too_small"


@dataclass
class FetchResult:
    doi: str
    status: FetchStatus
    path: Optional[Path] = None
    url: Optional[str] = None
    reason: Optional[str] = None


@dataclass
class FetchSummary:
    results: list[FetchResult] = field(default_factory=list)

    def by_status(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.results:
            out[r.status.value] = out.get(r.status.value, 0) + 1
        return out


def _build_session(
    *,
    timeout: int,
    max_retries: int,
    user_agent: str,
) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=max_retries,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": user_agent})
    # attach timeout convenience
    session.request = _with_default_timeout(session.request, timeout)  # type: ignore[assignment]
    return session


def _with_default_timeout(request_fn, default_timeout):
    def wrapper(method, url, **kwargs):
        kwargs.setdefault("timeout", default_timeout)
        return request_fn(method, url, **kwargs)

    return wrapper


class FetcherBase(ABC):
    """Base PDF fetcher. Subclass and implement `_resolve_pdf_url`."""

    # Minimum acceptable PDF size (bytes). Smaller is almost certainly an error page.
    MIN_PDF_BYTES = 10 * 1024  # 10 KB

    def __init__(
        self,
        output_dir: Path,
        *,
        delay_seconds: float = 1.0,
        timeout_seconds: int = 60,
        max_retries: int = 3,
        contact_email: Optional[str] = None,
        tool_name: str = "simtool-manage",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay_seconds
        self.contact_email = contact_email
        self.tool_name = tool_name
        self.download_log_path = self.output_dir / "download_log.json"
        self._fetched_dois = self._load_download_log()

        ua_parts = [f"{tool_name}/0.1"]
        if contact_email:
            ua_parts.append(f"(mailto:{contact_email})")
        self.session = _build_session(
            timeout=timeout_seconds,
            max_retries=max_retries,
            user_agent=" ".join(ua_parts),
        )

    # ----- persistence -----

    def _load_download_log(self) -> dict[str, str]:
        if not self.download_log_path.exists():
            return {}
        try:
            with open(self.download_log_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return dict(data.get("fetched", {}))
        except Exception as exc:
            logger.warning("Could not load download log %s: %s", self.download_log_path, exc)
            return {}

    def _save_download_log(self) -> None:
        try:
            with open(self.download_log_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "fetched": self._fetched_dois,
                        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    },
                    f,
                    indent=2,
                )
        except Exception as exc:
            logger.warning("Could not save download log %s: %s", self.download_log_path, exc)

    # ----- public API -----

    def fetch(self, entry: CorpusEntry, *, skip_existing: bool = True) -> FetchResult:
        # Guard: DOI-verification required before any fetch attempt.
        if entry.verify_doi:
            return FetchResult(
                doi=entry.doi,
                status=FetchStatus.SKIPPED_VERIFY_DOI,
                reason="verify_doi=True; resolve DOI before fetching",
            )

        if skip_existing and entry.doi in self._fetched_dois:
            return FetchResult(
                doi=entry.doi,
                status=FetchStatus.SKIPPED_EXISTING,
                path=Path(self._fetched_dois[entry.doi]),
                reason="already in download log",
            )

        url = self._resolve_pdf_url(entry)
        if url is None:
            return FetchResult(
                doi=entry.doi,
                status=FetchStatus.FAILED_NO_URL,
                reason="could not resolve a PDF URL",
            )

        filename = self._filename_for(entry)
        output_path = self.output_dir / filename
        if skip_existing and output_path.exists() and output_path.stat().st_size >= self.MIN_PDF_BYTES:
            self._fetched_dois[entry.doi] = str(output_path)
            self._save_download_log()
            return FetchResult(
                doi=entry.doi,
                status=FetchStatus.SKIPPED_EXISTING,
                path=output_path,
                url=url,
                reason="file already on disk",
            )

        return self._download(entry, url, output_path)

    def _download(self, entry: CorpusEntry, url: str, output_path: Path) -> FetchResult:
        part_path = output_path.with_suffix(output_path.suffix + ".part")
        try:
            with self.session.get(url, stream=True, allow_redirects=True) as resp:
                resp.raise_for_status()
                content_type = (resp.headers.get("Content-Type") or "").lower()
                if "pdf" not in content_type and "octet-stream" not in content_type:
                    # Many redirecting PDF endpoints still return text/html wrappers;
                    # flag but continue — size check will catch junk.
                    logger.debug("Unexpected content-type for %s: %s", entry.doi, content_type)
                with open(part_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=64 * 1024):
                        if chunk:
                            f.write(chunk)
        except requests.exceptions.HTTPError as exc:
            _cleanup(part_path)
            return FetchResult(
                doi=entry.doi,
                status=FetchStatus.FAILED_HTTP,
                url=url,
                reason=f"HTTP {exc.response.status_code if exc.response is not None else '?'}: {exc}",
            )
        except requests.exceptions.RequestException as exc:
            _cleanup(part_path)
            return FetchResult(
                doi=entry.doi,
                status=FetchStatus.FAILED_HTTP,
                url=url,
                reason=f"request error: {exc}",
            )

        if part_path.stat().st_size < self.MIN_PDF_BYTES:
            _cleanup(part_path)
            return FetchResult(
                doi=entry.doi,
                status=FetchStatus.FAILED_TOO_SMALL,
                url=url,
                reason=f"response under {self.MIN_PDF_BYTES} bytes (likely error page)",
            )

        # Magic-number check: PDFs start with %PDF-
        with open(part_path, "rb") as f:
            head = f.read(5)
        if head != b"%PDF-":
            _cleanup(part_path)
            return FetchResult(
                doi=entry.doi,
                status=FetchStatus.FAILED_CONTENT,
                url=url,
                reason=f"file does not start with %PDF- (got {head!r})",
            )

        part_path.replace(output_path)
        self._fetched_dois[entry.doi] = str(output_path)
        self._save_download_log()
        time.sleep(self.delay)
        return FetchResult(
            doi=entry.doi,
            status=FetchStatus.DOWNLOADED,
            path=output_path,
            url=url,
        )

    # ----- hooks for subclasses -----

    @abstractmethod
    def _resolve_pdf_url(self, entry: CorpusEntry) -> Optional[str]:
        """Return a URL to stream the PDF from, or None if resolution fails."""

    def _filename_for(self, entry: CorpusEntry) -> str:
        """Default: use PMC id if present, else a DOI-safe slug."""

        if entry.pmc_id:
            return f"{entry.pmc_id}.pdf"
        safe = entry.doi.replace("/", "_").replace(":", "_")
        return f"{safe}.pdf"


def _cleanup(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_PAYWALLED = {AccessStatus.PAYWALLED_LIKELY, AccessStatus.PAYWALLED_NO_ACCESS}


def fetch_corpus(
    log: PrismaLog,
    output_dir: Path,
    *,
    fetcher_registry: "dict[AccessStatus, FetcherBase]",
    max_fetches: Optional[int] = None,
    skip_existing: bool = True,
    show_progress: bool = True,
) -> FetchSummary:
    """Iterate entries in `log` and dispatch each to the right fetcher.

    Paywalled entries are logged as SKIPPED_PAYWALLED without a network call.
    Unknown access status logs a warning and is skipped.
    """

    summary = FetchSummary()

    iterable = log.entries
    if show_progress:
        try:
            from tqdm import tqdm

            iterable = tqdm(log.entries, desc="Fetching PDFs")
        except ImportError:
            pass

    attempted = 0
    for entry in iterable:
        if entry.access in _PAYWALLED:
            summary.results.append(
                FetchResult(
                    doi=entry.doi,
                    status=FetchStatus.SKIPPED_PAYWALLED,
                    reason=f"access={entry.access.value}",
                )
            )
            continue

        fetcher = fetcher_registry.get(entry.access)
        if fetcher is None:
            logger.warning("No fetcher registered for access=%s (doi=%s)", entry.access.value, entry.doi)
            summary.results.append(
                FetchResult(
                    doi=entry.doi,
                    status=FetchStatus.FAILED_NO_URL,
                    reason=f"no fetcher for access={entry.access.value}",
                )
            )
            continue

        result = fetcher.fetch(entry, skip_existing=skip_existing)
        summary.results.append(result)
        if result.status == FetchStatus.FAILED_HTTP:
            logger.warning("HTTP failure for %s: %s", entry.doi, result.reason)
        elif result.status == FetchStatus.DOWNLOADED:
            logger.info("downloaded %s -> %s", entry.doi, result.path)
        attempted += 1
        if max_fetches is not None and attempted >= max_fetches:
            break

    return summary


def default_contact_email() -> Optional[str]:
    """Pick up the user's polite-pool contact email from env."""

    return os.environ.get("SIMTOOL_CONTACT_EMAIL") or None
