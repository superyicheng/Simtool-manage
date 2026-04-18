"""PMC-specific PDF fetcher.

Resolution strategy (April 2026):

  In early 2026 NCBI migrated the PMC Article Dataset distribution to
  AWS-hosted buckets. The legacy FTP oa_package/ paths (what the
  `oa.fcgi` service still returns) now 404, and the direct
  `pmc.ncbi.nlm.nih.gov/articles/<PMCID>/pdf/` URLs gate behind a
  JavaScript Proof-of-Work challenge.

  Current clean endpoint:

      https://pmc-oa-opendata.s3.amazonaws.com/<PMCID>.<ver>/<PMCID>.<ver>.pdf

  One S3 ListObjectsV2 call lets us resolve the current version without
  guessing — we list by `prefix=<PMCID>` and pick the key that matches
  `<PMCID>.<ver>/<PMCID>.<ver>.pdf`. No authentication; the bucket is
  public. Docs: https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/

  Note: the S3 bucket is broader than the legacy PMC OA Subset —
  papers like Kits et al. 2017 (Nature, comammox kinetics) that
  `oa.fcgi` used to reject with `idDoesNotExist` are available here.

If a paper is missing from the AWS bucket we return FAILED_NO_URL; a
later pass can try Europe PMC's fullTextXML (text-only).
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from typing import Optional

from simtool.corpus.fetcher import FetcherBase
from simtool.corpus.manifest import CorpusEntry

logger = logging.getLogger(__name__)

_S3_BUCKET = "https://pmc-oa-opendata.s3.amazonaws.com"
_S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


class PmcFetcher(FetcherBase):
    """Fetch open-access PDFs from the PMC AWS bucket."""

    def _resolve_pdf_url(self, entry: CorpusEntry) -> Optional[str]:
        if not entry.pmc_id:
            logger.warning("PmcFetcher: entry %s has no pmc_id", entry.doi)
            return None

        keys = _list_keys(self.session, entry.pmc_id)
        if not keys:
            logger.info("PMC AWS: no keys for %s", entry.pmc_id)
            return None

        pdf_key = _pick_main_pdf(entry.pmc_id, keys)
        if pdf_key is None:
            logger.info("PMC AWS: no main PDF among keys for %s: %s", entry.pmc_id, keys[:10])
            return None
        return f"{_S3_BUCKET}/{pdf_key}"

    def _filename_for(self, entry: CorpusEntry) -> str:
        """Stable per-PMCID filename, independent of the S3 version suffix."""

        return f"{entry.pmc_id}.pdf" if entry.pmc_id else super()._filename_for(entry)


def _list_keys(session, pmc_id: str, *, max_keys: int = 200) -> list[str]:
    """Return all S3 keys whose name starts with `<pmc_id>`."""

    url = f"{_S3_BUCKET}/?list-type=2&prefix={pmc_id}&max-keys={max_keys}"
    try:
        resp = session.get(url)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        logger.warning("PMC S3 listing failed for %s: %s", pmc_id, exc)
        return []
    return _parse_s3_listing_keys(resp.text)


def _parse_s3_listing_keys(xml_text: str) -> list[str]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.debug("Failed to parse S3 listing XML: %s", exc)
        return []
    return [
        k.text
        for c in root.findall("s3:Contents", _S3_NS)
        for k in [c.find("s3:Key", _S3_NS)]
        if k is not None and k.text
    ]


_MAIN_PDF_RE = re.compile(r"^(?P<pmcid>PMC\d+)\.(?P<ver>\d+)/(?P=pmcid)\.(?P=ver)\.pdf$")


def _pick_main_pdf(pmc_id: str, keys: list[str]) -> Optional[str]:
    """Among the S3 listing keys, select the main article PDF.

    The main-article key matches `<PMCID>.<ver>/<PMCID>.<ver>.pdf`. When
    multiple versions exist, pick the highest version.
    """

    candidates: list[tuple[int, str]] = []
    for key in keys:
        m = _MAIN_PDF_RE.match(key)
        if m and m.group("pmcid") == pmc_id:
            candidates.append((int(m.group("ver")), key))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]
