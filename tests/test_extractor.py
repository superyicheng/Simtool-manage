"""Tests for the extractor base class and chemostat subclass.

The Anthropic SDK is not invoked — a dummy client captures the request
kwargs and returns a canned response. This lets us verify:
  - the request carries a PDF document block with base64 bytes
  - the system prompt has a cache_control boundary
  - tool_choice forces our `report_parameters` tool
  - tool-call responses are parsed and validated into RawExtraction
  - extractions failing validation are dropped, not crashed on
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from simtool.extractors.base import BaseExtractor, ExtractorConfig, MethodProfile
from simtool.extractors.methods import ChemostatExtractor
from simtool.extractors.schemas import REPORT_PARAMETERS_TOOL, RawExtraction
from simtool.schema.parameter_record import ExtractionModality, MeasurementMethod


_TINY_PDF = b"%PDF-1.4\n" + b"tiny-pdf-body\n" + b"%%EOF\n"


def _write_pdf(tmp_path: Path) -> Path:
    p = tmp_path / "sample.pdf"
    p.write_bytes(_TINY_PDF)
    return p


class _DummyClient:
    """Captures the last messages_create call and returns a canned response."""

    def __init__(self, response: Any):
        self.response = response
        self.last_kwargs: dict[str, Any] | None = None

    def messages_create(self, **kwargs: Any) -> Any:
        self.last_kwargs = kwargs
        return self.response


def _tool_response(extractions: list[dict[str, Any]]) -> dict[str, Any]:
    """Mimic the shape of an Anthropic response containing a tool_use block."""

    return {
        "content": [
            {
                "type": "tool_use",
                "name": REPORT_PARAMETERS_TOOL["name"],
                "input": {"extractions": extractions},
            }
        ],
        "stop_reason": "tool_use",
    }


def _valid_extraction(**over) -> dict[str, Any]:
    base = {
        "parameter_id": "mu_max",
        "value": 1.3,
        "unit": "d-1",
        "method": MeasurementMethod.CHEMOSTAT.value,
        "context": {
            "species": "Nitrosomonas europaea",
            "substrate": "NH4",
            "temperature_c": 25.0,
        },
        "span": {
            "page": 3,
            "text_excerpt": "mu_max was 1.3 d^-1 at 25C",
            "modality": ExtractionModality.PROSE.value,
        },
        "self_confidence": 0.85,
    }
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# request shape
# ---------------------------------------------------------------------------


def test_extract_sends_pdf_as_base64_document_block(tmp_path):
    client = _DummyClient(_tool_response([_valid_extraction()]))
    extractor = ChemostatExtractor(client=client)
    pdf = _write_pdf(tmp_path)

    extractor.extract(pdf)

    kwargs = client.last_kwargs
    assert kwargs is not None
    user_msg = kwargs["messages"][0]
    assert user_msg["role"] == "user"
    doc = user_msg["content"][0]
    assert doc["type"] == "document"
    assert doc["source"]["media_type"] == "application/pdf"
    assert doc["source"]["type"] == "base64"
    import base64

    decoded = base64.standard_b64decode(doc["source"]["data"])
    assert decoded == _TINY_PDF


def test_system_prompt_has_cache_control(tmp_path):
    client = _DummyClient(_tool_response([]))
    ChemostatExtractor(client=client).extract(_write_pdf(tmp_path))

    system = client.last_kwargs["system"]
    assert isinstance(system, list)
    assert system[0]["cache_control"] == {"type": "ephemeral"}


def test_tool_choice_forces_report_parameters(tmp_path):
    client = _DummyClient(_tool_response([]))
    ChemostatExtractor(client=client).extract(_write_pdf(tmp_path))

    kwargs = client.last_kwargs
    assert kwargs["tool_choice"] == {
        "type": "tool",
        "name": REPORT_PARAMETERS_TOOL["name"],
    }
    assert kwargs["tools"][0]["name"] == REPORT_PARAMETERS_TOOL["name"]


def test_temperature_zero_by_default(tmp_path):
    client = _DummyClient(_tool_response([]))
    ChemostatExtractor(client=client).extract(_write_pdf(tmp_path))

    assert client.last_kwargs["temperature"] == 0.0


def test_method_profile_text_in_system_prompt(tmp_path):
    client = _DummyClient(_tool_response([]))
    ChemostatExtractor(client=client).extract(_write_pdf(tmp_path))

    system_text = client.last_kwargs["system"][0]["text"]
    assert "chemostat" in system_text.lower()
    assert "mu_max" in system_text  # from vocab section
    assert "iDynoMiCS 2" in system_text


# ---------------------------------------------------------------------------
# response parsing
# ---------------------------------------------------------------------------


def test_valid_extraction_parsed(tmp_path):
    client = _DummyClient(_tool_response([_valid_extraction()]))
    result = ChemostatExtractor(client=client).extract(_write_pdf(tmp_path))

    assert len(result.extractions) == 1
    rec = result.extractions[0]
    assert isinstance(rec, RawExtraction)
    assert rec.parameter_id == "mu_max"
    assert rec.value == 1.3
    assert rec.method == MeasurementMethod.CHEMOSTAT
    assert result.stop_reason == "tool_use"


def test_invalid_extraction_is_dropped_not_raised(tmp_path):
    bad = _valid_extraction()
    bad["self_confidence"] = 1.5  # out of [0,1]
    client = _DummyClient(_tool_response([_valid_extraction(), bad]))
    result = ChemostatExtractor(client=client).extract(_write_pdf(tmp_path))

    # The valid one is kept; the invalid one is silently dropped.
    assert len(result.extractions) == 1
    assert result.extractions[0].self_confidence == 0.85


def test_missing_method_field_filled_from_profile(tmp_path):
    item = _valid_extraction()
    del item["method"]
    client = _DummyClient(_tool_response([item]))
    result = ChemostatExtractor(client=client).extract(_write_pdf(tmp_path))

    assert len(result.extractions) == 1
    assert result.extractions[0].method == MeasurementMethod.CHEMOSTAT


def test_no_tool_use_block_returns_empty(tmp_path):
    response = {
        "content": [{"type": "text", "text": "Sorry, cannot comply."}],
        "stop_reason": "end_turn",
    }
    client = _DummyClient(response)
    result = ChemostatExtractor(client=client).extract(_write_pdf(tmp_path))

    assert result.extractions == []
    assert result.stop_reason == "end_turn"


def test_tool_input_as_string_is_json_decoded(tmp_path):
    """Anthropic usually returns tool input as a dict, but some code paths
    stringify it — make sure we handle both."""

    import json

    response = {
        "content": [
            {
                "type": "tool_use",
                "name": REPORT_PARAMETERS_TOOL["name"],
                "input": json.dumps({"extractions": [_valid_extraction()]}),
            }
        ],
        "stop_reason": "tool_use",
    }
    client = _DummyClient(response)
    result = ChemostatExtractor(client=client).extract(_write_pdf(tmp_path))

    assert len(result.extractions) == 1


def test_custom_method_profile_subclass(tmp_path):
    """BaseExtractor is usable as-is with a custom MethodProfile — no need
    to write a new subclass per method."""

    class _BatchExtractor(BaseExtractor):
        method_profile = MethodProfile(
            method=MeasurementMethod.BATCH_FIT,
            prose_hints=["Batch culture with Monod fit to depletion curve."],
            expected_parameters=["mu_max", "K_s", "Y_XS"],
        )

    item = _valid_extraction(method=MeasurementMethod.BATCH_FIT.value)
    client = _DummyClient(_tool_response([item]))
    result = _BatchExtractor(client=client).extract(_write_pdf(tmp_path))

    assert result.method == MeasurementMethod.BATCH_FIT
    assert len(result.extractions) == 1
    assert result.extractions[0].method == MeasurementMethod.BATCH_FIT


def test_extractor_config_overrides(tmp_path):
    client = _DummyClient(_tool_response([]))
    cfg = ExtractorConfig(model="claude-opus-4-7", temperature=0.3, max_tokens=2048)
    ChemostatExtractor(client=client, config=cfg).extract(_write_pdf(tmp_path))

    kwargs = client.last_kwargs
    assert kwargs["model"] == "claude-opus-4-7"
    assert kwargs["temperature"] == 0.3
    assert kwargs["max_tokens"] == 2048
