"""Unit tests for the PDF fetcher — HTTP is mocked; no network calls."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

import pytest

from simtool.corpus.fetcher import FetchStatus
from simtool.corpus.manifest import AccessStatus, CorpusEntry, InclusionDecision
from simtool.corpus.pmc_fetcher import (
    PmcFetcher,
    _parse_s3_listing_keys,
    _pick_main_pdf,
)


_VALID_PDF = b"%PDF-1.5\n" + os.urandom(40 * 1024)  # >10 KB, starts with %PDF-


def _make_entry(**over) -> CorpusEntry:
    base = dict(
        doi="10.1234/test",
        title="Test paper",
        year=2020,
        access=AccessStatus.OPEN_PMC,
        pmc_id="PMC0000001",
        decision=InclusionDecision.SCREENED_IN,
        decision_reason="test",
        decision_date="2026-04-18",
        decision_by="unit_test",
    )
    base.update(over)
    return CorpusEntry(**base)


def _listing_xml(keys: list[str]) -> str:
    contents = "".join(
        f"<Contents><Key>{k}</Key><Size>100</Size></Contents>" for k in keys
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Name>pmc-oa-opendata</Name>
  <KeyCount>{len(keys)}</KeyCount>
  {contents}
</ListBucketResult>"""


class _FakeResponse:
    def __init__(self, status: int, body: bytes, content_type: str = "application/octet-stream"):
        self.status_code = status
        self._body = body
        self.headers = {"Content-Type": content_type}
        self.text = body.decode("utf-8", errors="ignore")

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            err = requests.exceptions.HTTPError(f"HTTP {self.status_code}")
            err.response = self  # type: ignore[attr-defined]
            raise err

    def iter_content(self, chunk_size: int = 64 * 1024):
        view = memoryview(self._body)
        for i in range(0, len(view), chunk_size):
            yield bytes(view[i : i + chunk_size])

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _s3_router(pmc_id: str, keys: list[str], pdf_bytes: bytes | None):
    """Route session.get: S3 listing → XML, PDF key URL → pdf_bytes."""

    def _route(url, *args, **kwargs):
        if "list-type=2" in url and f"prefix={pmc_id}" in url:
            return _FakeResponse(200, _listing_xml(keys).encode("utf-8"), "application/xml")
        for key in keys:
            if url.endswith(key) and pdf_bytes is not None:
                return _FakeResponse(200, pdf_bytes)
        return _FakeResponse(404, b"not found")

    return _route


# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------


def test_pick_main_pdf_single_version():
    keys = [
        "PMC0000001.1/PMC0000001.1.pdf",
        "PMC0000001.1/PMC0000001.1.xml",
        "PMC0000001.1/Image_1.jpg",
    ]
    assert _pick_main_pdf("PMC0000001", keys) == "PMC0000001.1/PMC0000001.1.pdf"


def test_pick_main_pdf_prefers_highest_version():
    keys = [
        "PMC0000001.1/PMC0000001.1.pdf",
        "PMC0000001.3/PMC0000001.3.pdf",
        "PMC0000001.2/PMC0000001.2.pdf",
    ]
    assert _pick_main_pdf("PMC0000001", keys) == "PMC0000001.3/PMC0000001.3.pdf"


def test_pick_main_pdf_ignores_supplementary():
    # Only supplementary files, no <PMCID>.<ver>.pdf main file
    keys = [
        "PMC0000001.1/Data_Sheet_1.pdf",
        "PMC0000001.1/supplementary.pdf",
        "PMC0000001.1/Image_1.jpg",
    ]
    assert _pick_main_pdf("PMC0000001", keys) is None


def test_pick_main_pdf_empty_keys():
    assert _pick_main_pdf("PMC0000001", []) is None


def test_parse_s3_listing_keys():
    xml = _listing_xml(["PMC1/PMC1.1.pdf", "PMC1/PMC1.1.xml"])
    assert _parse_s3_listing_keys(xml) == ["PMC1/PMC1.1.pdf", "PMC1/PMC1.1.xml"]


def test_parse_s3_listing_keys_empty():
    xml = """<?xml version="1.0"?><ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><KeyCount>0</KeyCount></ListBucketResult>"""
    assert _parse_s3_listing_keys(xml) == []


def test_parse_s3_listing_keys_malformed():
    assert _parse_s3_listing_keys("not xml") == []


# ---------------------------------------------------------------------------
# End-to-end fetch flow
# ---------------------------------------------------------------------------


def test_download_success(tmp_path):
    fetcher = PmcFetcher(output_dir=tmp_path, delay_seconds=0)
    entry = _make_entry()
    keys = ["PMC0000001.1/PMC0000001.1.pdf", "PMC0000001.1/PMC0000001.1.xml"]

    with mock.patch.object(
        fetcher.session,
        "get",
        side_effect=_s3_router("PMC0000001", keys, _VALID_PDF),
    ):
        result = fetcher.fetch(entry)

    assert result.status == FetchStatus.DOWNLOADED
    assert result.path == tmp_path / "PMC0000001.pdf"
    assert (tmp_path / "PMC0000001.pdf").exists()
    assert (tmp_path / "download_log.json").exists()


def test_listing_returns_no_main_pdf_fails_no_url(tmp_path):
    fetcher = PmcFetcher(output_dir=tmp_path, delay_seconds=0)
    entry = _make_entry()
    # Only supplementary files in the listing
    keys = ["PMC0000001.1/Data_Sheet_1.docx", "PMC0000001.1/supp.pdf"]

    with mock.patch.object(
        fetcher.session,
        "get",
        side_effect=_s3_router("PMC0000001", keys, _VALID_PDF),
    ):
        result = fetcher.fetch(entry)

    assert result.status == FetchStatus.FAILED_NO_URL


def test_listing_returns_empty_fails_no_url(tmp_path):
    fetcher = PmcFetcher(output_dir=tmp_path, delay_seconds=0)
    entry = _make_entry()

    with mock.patch.object(
        fetcher.session,
        "get",
        side_effect=_s3_router("PMC0000001", [], None),
    ):
        result = fetcher.fetch(entry)

    assert result.status == FetchStatus.FAILED_NO_URL


def test_pdf_magic_number_check_rejects_html(tmp_path):
    fetcher = PmcFetcher(output_dir=tmp_path, delay_seconds=0)
    entry = _make_entry()
    keys = ["PMC0000001.1/PMC0000001.1.pdf"]
    html_blob = b"<html><body>error</body></html>" + os.urandom(40 * 1024)

    with mock.patch.object(
        fetcher.session,
        "get",
        side_effect=_s3_router("PMC0000001", keys, html_blob),
    ):
        result = fetcher.fetch(entry)

    assert result.status == FetchStatus.FAILED_CONTENT


def test_verify_doi_entries_are_skipped_before_network(tmp_path):
    fetcher = PmcFetcher(output_dir=tmp_path, delay_seconds=0)
    entry = _make_entry(verify_doi=True)

    with mock.patch.object(
        fetcher.session, "get", side_effect=AssertionError("should not be called")
    ):
        result = fetcher.fetch(entry)

    assert result.status == FetchStatus.SKIPPED_VERIFY_DOI


def test_resume_skips_entries_in_download_log(tmp_path):
    (tmp_path / "download_log.json").write_text(
        json.dumps({"fetched": {"10.1234/test": str(tmp_path / "PMC0000001.pdf")}})
    )
    (tmp_path / "PMC0000001.pdf").write_bytes(_VALID_PDF)

    fetcher = PmcFetcher(output_dir=tmp_path, delay_seconds=0)
    entry = _make_entry()

    with mock.patch.object(
        fetcher.session, "get", side_effect=AssertionError("should not be called")
    ):
        result = fetcher.fetch(entry)

    assert result.status == FetchStatus.SKIPPED_EXISTING


def test_paywalled_entries_never_hit_network(tmp_path):
    from simtool.corpus.fetcher import fetch_corpus
    from simtool.corpus.manifest import PrismaLog

    entry = _make_entry(access=AccessStatus.PAYWALLED_LIKELY)
    log = PrismaLog(target_system="test", entries=[entry])
    fetcher = PmcFetcher(output_dir=tmp_path, delay_seconds=0)

    with mock.patch.object(
        fetcher.session, "get", side_effect=AssertionError("no network for paywalled")
    ):
        summary = fetch_corpus(
            log,
            output_dir=tmp_path,
            fetcher_registry={AccessStatus.OPEN_PMC: fetcher},
            show_progress=False,
        )

    assert summary.by_status() == {FetchStatus.SKIPPED_PAYWALLED.value: 1}
