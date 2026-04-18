"""Disk-backed store for meta-models, panels, suggestions, and run records.

Layout under ``root``:

    metamodels/<id>/<version>/metamodel.json
    panels/<panel_id>/panel.json
    suggestions/<meta_model_id>/ledger.json
    runs/<run_id>/...          # owned by the runner's RunLayout
"""

from simtool.persistence.store import Store

__all__ = ["Store"]
