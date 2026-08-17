"""Study 933 — Same Issuer, Two Ladders (preferred shares vs exchange-traded baby bonds).

``data``     — the real tape (yfinance daily total-return closes, shared parquet cache)
               plus the deterministic offline synthetic panel.
``strategy`` — the paired ladder race, the HAC inference, the costed long-short and the
               synthetic-control detector.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
