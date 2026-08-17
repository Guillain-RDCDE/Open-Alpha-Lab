"""Study 935 — Value Averaging (Edleson) versus Dollar-Cost Averaging.

Two modules:

- ``data`` — the real tape (SPY equity, BIL cash), cached parquet, plus the deterministic
  offline synthetic generators used by the whole test-suite.
- ``strategy`` — the value path, the two accumulation engines (VA and DCA) with their cash
  buffer, the rolling-window race, and the inference primitives.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
