"""iDynoMiCS 2 framework plugin.

Drop the Simtool folder on a cloud VM, install deps with
``pip install -e '.[ara]'``, set ``IDYNOMICS_2_JAR`` to the jar path,
and this plugin can drive an end-to-end run against real iDynoMiCS 2.
"""

from simtool.frameworks.idynomics_2.config import (
    IDynoMiCS2Config,
    IDynoMiCS2NotAvailable,
    require_runtime,
    resolve_jar_path,
)
from simtool.frameworks.idynomics_2.doctor import (
    CheckResult,
    HealthReport,
    run_health_check,
    save_jar_path,
)
from simtool.frameworks.idynomics_2.plugin import IDynoMiCS2Plugin

__all__ = [
    "CheckResult",
    "HealthReport",
    "IDynoMiCS2Config",
    "IDynoMiCS2NotAvailable",
    "IDynoMiCS2Plugin",
    "require_runtime",
    "resolve_jar_path",
    "run_health_check",
    "save_jar_path",
]
