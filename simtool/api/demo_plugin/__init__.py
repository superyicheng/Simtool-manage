"""Demo FrameworkPlugin — stand-in until the real iDynoMiCS 2 plugin lands.

Emits a scripted sequence of ProgressReports and a synthetic OutputBundle
whose scalar time series land inside the meta-model's reconciled ranges.
Lets the Simulate view be demonstrably wired end-to-end before iDynoMiCS
integration is complete.
"""

from simtool.api.demo_plugin.plugin import DemoPlugin

__all__ = ["DemoPlugin"]
