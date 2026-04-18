"""Framework plugins — one subpackage per supported simulation framework.

Each subpackage exports a class implementing ``simtool.connector.plugin.FrameworkPlugin``.
The runner's orchestrator picks a plugin by ``name`` and drives it through
the standard lifecycle (validate_ir -> lower -> execute -> monitor ->
parse_outputs -> generate_protocol).

Currently shipped:
    idynomics_2 - iDynoMiCS 2.0 (individual-based biofilm simulator).
"""

from simtool.frameworks.idynomics_2 import IDynoMiCS2Plugin, IDynoMiCS2NotAvailable

__all__ = ["IDynoMiCS2Plugin", "IDynoMiCS2NotAvailable"]
