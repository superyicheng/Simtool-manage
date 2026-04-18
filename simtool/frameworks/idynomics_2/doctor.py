"""Health check for the iDynoMiCS 2 runtime.

Tells the user — or an Ara agent running ``simtool idynomics check`` —
exactly what is and isn't in place. Each check is independent so partial
setups produce partial-success output rather than a single opaque error.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from simtool.frameworks.idynomics_2.config import (
    _DEFAULT_CONFIG_PATH,
    resolve_jar_path,
    resolve_java_bin,
)


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""

    def line(self) -> str:
        mark = "✓" if self.ok else "✗"
        return f"  [{mark}] {self.name}: {self.detail}"


@dataclass
class HealthReport:
    checks: list[CheckResult]

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def render(self) -> str:
        lines = ["iDynoMiCS 2 runtime check"]
        lines.extend(c.line() for c in self.checks)
        lines.append("")
        lines.append(
            "All green — `simtool idynomics run <protocol.xml>` should work."
            if self.ok
            else "Fix the ✗ items above. See docs/ara-deployment.md for install help."
        )
        return "\n".join(lines)


def run_health_check(config_path: Optional[Path] = None) -> HealthReport:
    checks: list[CheckResult] = []

    cfg_path = config_path or _DEFAULT_CONFIG_PATH
    jar = resolve_jar_path(config_path=cfg_path)
    if jar is None:
        checks.append(CheckResult(
            name="iDynoMiCS 2 jar",
            ok=False,
            detail="not found. Set IDYNOMICS_2_JAR or run "
                   "`simtool idynomics set-jar <path>`.",
        ))
    else:
        checks.append(CheckResult(
            name="iDynoMiCS 2 jar", ok=True, detail=str(jar),
        ))

    java = resolve_java_bin()
    if java is None:
        checks.append(CheckResult(
            name="Java runtime",
            ok=False,
            detail="not found. Install JDK 11+ or set JAVA_HOME.",
        ))
    else:
        version = _java_version(java)
        checks.append(CheckResult(
            name="Java runtime",
            ok=True,
            detail=f"{java}  ({version})",
        ))

    if jar is not None:
        jar_dir = jar.parent
        # default.cfg + config/ are load-bearing: iDynoMiCS reads default.cfg
        # from cwd at startup, and its supplementary_property_files pull from
        # config/. The jar itself is self-contained (fat jar), so `lib/` is
        # NOT required.
        for sibling, hint in (
            ("default.cfg", "needed at cwd=jar_dir for supplementary property loading"),
            ("config", "pulled in via default.cfg's supplementary_property_files"),
        ):
            path = jar_dir / sibling
            exists = path.exists()
            checks.append(CheckResult(
                name=f"iDynoMiCS sibling `{sibling}`",
                ok=exists,
                detail=str(path) if exists else f"missing — {hint}",
            ))

    checks.append(CheckResult(
        name="Simtool config dir",
        ok=cfg_path.parent.exists() or _can_create(cfg_path.parent),
        detail=str(cfg_path.parent),
    ))

    return HealthReport(checks=checks)


def save_jar_path(jar_path: Path, config_path: Optional[Path] = None) -> Path:
    """Persist ``jar_path`` into the per-user config so future sessions
    pick it up without needing IDYNOMICS_2_JAR in the env."""
    jar_path = Path(jar_path).expanduser().resolve()
    if not jar_path.is_file():
        raise FileNotFoundError(f"no such jar: {jar_path}")
    cfg_path = config_path or _DEFAULT_CONFIG_PATH
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, str] = {}
    if cfg_path.is_file():
        try:
            data = json.loads(cfg_path.read_text())
        except (OSError, json.JSONDecodeError):
            data = {}
    data["jar_path"] = str(jar_path)
    cfg_path.write_text(json.dumps(data, indent=2))
    return cfg_path


def _java_version(java_bin: str) -> str:
    try:
        proc = subprocess.run(
            [java_bin, "-version"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "version unknown"
    # `java -version` writes to stderr.
    out = (proc.stderr or proc.stdout or "").splitlines()
    return out[0] if out else "version unknown"


def _can_create(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path.is_dir()
    except OSError:
        return False
