"""iDynoMiCS 2 runtime configuration.

Resolution order for the jar path:
    1. ``IDYNOMICS_2_JAR`` environment variable (absolute path)
    2. ``~/.simtool/idynomics_2.json`` config file, key ``jar_path``
    3. Standard install paths (/opt/idynomics, /usr/local/idynomics, ./vendor)

Java binary resolution: ``JAVA_HOME`` env var then PATH.

If no jar is found, ``require_jar`` raises ``IDynoMiCS2NotAvailable`` with
an install-guidance message — the plugin never silently falls back.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class IDynoMiCS2NotAvailable(RuntimeError):
    """Raised when iDynoMiCS 2 jar or a Java runtime cannot be found."""


_DEFAULT_CONFIG_PATH = Path.home() / ".simtool" / "idynomics_2.json"


_DEFAULT_SEARCH_PATHS: tuple[Path, ...] = (
    Path("/opt/idynomics/iDynoMiCS-2.0.jar"),
    Path("/usr/local/idynomics/iDynoMiCS-2.0.jar"),
    Path.cwd() / "vendor" / "iDynoMiCS-2.0.jar",
    Path.cwd() / "iDynoMiCS-2.0.jar",
)


@dataclass
class IDynoMiCS2Config:
    jar_path: Path
    java_bin: str
    java_opts: tuple[str, ...] = ()

    def as_command(self, protocol_xml_path: Path) -> list[str]:
        # `-protocol` is iDynoMiCS 2's batch-mode flag; omitting it launches
        # the GUI and the simulation never starts headlessly.
        return [
            self.java_bin,
            *self.java_opts,
            "-jar",
            str(self.jar_path),
            "-protocol",
            str(protocol_xml_path),
        ]


def resolve_jar_path(config_path: Path = _DEFAULT_CONFIG_PATH) -> Optional[Path]:
    """Return the first valid jar path found; None if nothing resolves."""
    env_val = os.environ.get("IDYNOMICS_2_JAR", "").strip()
    if env_val:
        p = Path(env_val).expanduser()
        if p.is_file():
            return p

    if config_path.is_file():
        try:
            data = json.loads(config_path.read_text())
            jar = data.get("jar_path")
            if jar:
                p = Path(jar).expanduser()
                if p.is_file():
                    return p
        except (OSError, json.JSONDecodeError):
            pass

    for p in _DEFAULT_SEARCH_PATHS:
        if p.is_file():
            return p

    return None


def resolve_java_bin() -> Optional[str]:
    java_home = os.environ.get("JAVA_HOME", "").strip()
    if java_home:
        candidate = Path(java_home) / "bin" / "java"
        if candidate.is_file():
            return str(candidate)
    which = shutil.which("java")
    return which


def require_runtime() -> IDynoMiCS2Config:
    """Return a valid IDynoMiCS2Config or raise IDynoMiCS2NotAvailable."""
    jar = resolve_jar_path()
    if jar is None:
        raise IDynoMiCS2NotAvailable(
            "iDynoMiCS 2 jar not found.\n"
            "  Set IDYNOMICS_2_JAR=/path/to/iDynoMiCS-2.0.jar,\n"
            "  or create ~/.simtool/idynomics_2.json with {\"jar_path\": \"...\"},\n"
            "  or place the jar at ./vendor/iDynoMiCS-2.0.jar."
        )
    java = resolve_java_bin()
    if java is None:
        raise IDynoMiCS2NotAvailable(
            "Java runtime not found. Install a JDK 11+ and set JAVA_HOME, "
            "or ensure `java` is on PATH."
        )
    return IDynoMiCS2Config(jar_path=jar, java_bin=java)
