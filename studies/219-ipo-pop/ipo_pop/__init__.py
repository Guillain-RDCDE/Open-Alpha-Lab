"""Study 219 — IPO-Pop: chasing the first-day pop and the long-run hangover.

Two sub-questions:
  (a) First-day pop magnitude — real but unharvestable by public market buyers.
  (b) Long-run drift after the first close — Ritter (1991) long-run underperformance.

Data: hardcoded table of ~40 notable US IPOs (1997-2023) + yfinance for post-close
forward returns at 6/12/36-month horizons.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
