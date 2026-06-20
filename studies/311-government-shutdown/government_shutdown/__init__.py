"""Study 311 — Government-Shutdown.

Event study of SPY around US federal government funding-gap shutdowns. The package
exposes a deterministic, offline synthetic generator (``data.synthetic_daily`` plus an
event-planting helper) and a cache-first real loader (``data.load_real``), and an
event-study engine with honest inference (``strategy``).
"""

from __future__ import annotations

from . import data, strategy

__all__ = ["data", "strategy"]
