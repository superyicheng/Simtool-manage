"""Runner — drives a FrameworkPlugin end-to-end.

The orchestrator is the one component that knows the full lifecycle:

    validate_ir -> lower -> user approves ledger -> execute
         -> monitor (collect structured reports) -> parse_outputs
         -> generate_protocol -> write RunRecord

Plugins implement the pieces; the runner sequences them.
"""

from simtool.runner.orchestrator import (
    RunExecutionSummary,
    run_panel,
    run_scientific_model,
)

__all__ = ["RunExecutionSummary", "run_panel", "run_scientific_model"]
