"""Method-specific extractor subclasses.

Each subclass is a thin wrapper around `BaseExtractor` with a filled-in
`MethodProfile`. Add a new method by writing a new module here (and
exporting its class) — no plumbing changes required.
"""

from simtool.extractors.methods.chemostat import ChemostatExtractor

__all__ = ["ChemostatExtractor"]
