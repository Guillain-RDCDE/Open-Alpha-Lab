"""Study 553 — GitHub-Activity: open-source commit/star velocity as an innovation-intensity nowcast.

Synthetic-only study (no free, survivorship-clean GitHub-to-ticker panel exists on a retail
stack), so the SIGNAL axis is capped at WEAK/NONE by construction. See ``data.py``.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
