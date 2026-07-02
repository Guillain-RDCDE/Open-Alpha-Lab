"""Study 571 — Pension-Underfunding.

Does a firm's underfunded defined-benefit pension hole — off-balance-sheet leverage the market
may under-weight — predict *lower* subsequent stock returns (Franzoni & Marin 2006)?

This study is **synthetic-only**: the point-in-time pension-funding footnotes (PBO, plan assets,
the funded status scaled by market cap) that Franzoni-Marin extract from Compustat pension items
are not reachable from a no-key retail stack. So the engine is exercised on a deterministic,
seeded synthetic cross-section (a single knob plants the effect), and the data-availability
limitation is stated openly on the SIGNAL axis. A synthetic-only study can never earn `REAL`
(that needs a robust t >= 2 on a real tape) — it is capped at `WEAK`/`NONE`.
"""

from __future__ import annotations

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
