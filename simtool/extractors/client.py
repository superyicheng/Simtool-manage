"""Anthropic client factory + adapter.

The real `anthropic` SDK exposes `client.messages.create(...)`. Our
`BaseExtractor` uses a `messages_create(...)` protocol so tests can pass
in a plain mock. This module adapts between the two.
"""

from __future__ import annotations

from typing import Any, Optional

from simtool.extractors.base import AnthropicClientLike
from simtool.settings import require_anthropic_api_key


class AnthropicClientAdapter:
    """Wrap the real SDK client so it matches `AnthropicClientLike`."""

    def __init__(self, sdk_client: Any):
        self._sdk = sdk_client

    def messages_create(self, **kwargs: Any) -> Any:
        return self._sdk.messages.create(**kwargs)


def default_client(api_key: Optional[str] = None) -> AnthropicClientLike:
    """Construct a ready-to-use client. Resolves the API key from the
    standard sources (env → .env → OS keyring) unless one is provided."""

    import anthropic  # local import so this module is importable without the SDK installed

    key = api_key or require_anthropic_api_key()
    return AnthropicClientAdapter(anthropic.Anthropic(api_key=key))
