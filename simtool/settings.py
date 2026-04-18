"""Anthropic API-key resolution for simtool.

Resolution order (highest priority first):
  1. `ANTHROPIC_API_KEY` environment variable — explicit overrides always win.
  2. `.env` file in the current working directory (developer convenience).
  3. OS-native credential store via the `keyring` library
     (macOS Keychain / Windows Credential Locker / Linux Secret Service).
  4. None — caller should surface a help message.

Users of a distributed build will typically run `simtool config set-key` once;
the key is encrypted by the OS and never touches the repo.
"""

from __future__ import annotations

import os
from typing import Optional

try:
    import keyring as _keyring  # type: ignore
    import keyring.errors as _keyring_errors  # type: ignore

    _KEYRING_AVAILABLE = True
except Exception:
    _KEYRING_AVAILABLE = False
    _keyring = None  # type: ignore
    _keyring_errors = None  # type: ignore

try:
    from dotenv import load_dotenv as _load_dotenv  # type: ignore

    _DOTENV_AVAILABLE = True
except Exception:
    _DOTENV_AVAILABLE = False

    def _load_dotenv(*_a, **_kw):  # type: ignore
        return False


SERVICE_NAME = "simtool-manage"
KEYRING_ACCOUNT_ANTHROPIC = "anthropic_api_key"
ENV_VAR_ANTHROPIC = "ANTHROPIC_API_KEY"


def _env_key() -> Optional[str]:
    raw = os.environ.get(ENV_VAR_ANTHROPIC, "")
    return raw.strip() or None


def _keyring_key() -> Optional[str]:
    if not _KEYRING_AVAILABLE:
        return None
    try:
        raw = _keyring.get_password(SERVICE_NAME, KEYRING_ACCOUNT_ANTHROPIC)  # type: ignore
    except Exception:
        return None
    if raw is None:
        return None
    stripped = raw.strip()
    return stripped or None


def get_anthropic_api_key() -> Optional[str]:
    """Return the Anthropic API key or None if none is configured."""

    env = _env_key()
    if env:
        return env

    if _DOTENV_AVAILABLE:
        # Do not override an already-set env var; load values from .env into
        # the process environment if present.
        _load_dotenv(override=False)
        env = _env_key()
        if env:
            return env

    return _keyring_key()


def set_anthropic_api_key(key: str) -> None:
    """Persist the key in the OS credential store. Raises if keyring unavailable."""

    if not _KEYRING_AVAILABLE:
        raise RuntimeError(
            "keyring is not available in this environment. "
            f"Set {ENV_VAR_ANTHROPIC} in your shell or a .env file instead."
        )
    trimmed = key.strip()
    if not trimmed:
        raise ValueError("API key is empty.")
    _keyring.set_password(SERVICE_NAME, KEYRING_ACCOUNT_ANTHROPIC, trimmed)  # type: ignore


def clear_anthropic_api_key() -> bool:
    """Remove any stored key. Returns True if one was deleted."""

    if not _KEYRING_AVAILABLE:
        return False
    try:
        _keyring.delete_password(SERVICE_NAME, KEYRING_ACCOUNT_ANTHROPIC)  # type: ignore
        return True
    except Exception:
        return False


def status() -> dict[str, object]:
    """Summary for the CLI `config status` command. Never returns the key itself."""

    env = _env_key()
    stored = _keyring_key() if _KEYRING_AVAILABLE else None
    effective = get_anthropic_api_key() or ""
    return {
        "has_env_key": env is not None,
        "has_stored_key": stored is not None,
        "keyring_available": _KEYRING_AVAILABLE,
        "dotenv_available": _DOTENV_AVAILABLE,
        "effective_suffix": effective[-4:] if len(effective) >= 4 else None,
    }


def require_anthropic_api_key() -> str:
    """Return the key or raise a user-facing error with setup instructions."""

    key = get_anthropic_api_key()
    if key:
        return key
    raise RuntimeError(
        "No Anthropic API key configured. Set one of:\n"
        f"  - export {ENV_VAR_ANTHROPIC}=sk-ant-...\n"
        f"  - put {ENV_VAR_ANTHROPIC}=sk-ant-... in a .env file in your working directory\n"
        "  - run: simtool config set-key   (stores in OS keyring, encrypted)"
    )
