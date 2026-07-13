"""Study 716 — "Short the diamond" (natural-diamond prices collapsing as lab-grown scales).

Natural polished-diamond prices peaked in early 2022 and have fallen materially since,
while lab-grown wholesale prices cratered ~80–90% — a widely-pitched "the diamond is
dying, short the miner / buy the beaten-down miner" idea. We test the strongest *tradable*
version: the natural-diamond price index (hardcoded, cited, **approximate** — a labelled
proxy, not a live feed) and the only listed ways to express the trade — the diamond
jeweler **Signet (SIG)** and a pure-play diamond miner **Lucara (LUC.TO)** — all
benchmarked against ``SPY`` on return, volatility, drawdown, alpha, and the frictions the
pitch never charges (short borrow on an illiquid microcap; the retail→resale haircut on a
physical stone).

See :mod:`diamond_price_index.data` (hardcoded cited price index + yfinance equity proxies
+ a deterministic synthetic *collapse* control) and :mod:`diamond_price_index.strategy`
(CAGR/vol/MDD, annual-excess *t*, Newey-West proxy alpha, the resale haircut)."""

from . import data, strategy

__all__ = ["data", "strategy"]
