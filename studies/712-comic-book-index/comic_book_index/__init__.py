"""Study 712 — "CGC-graded key comics are an asset class" (the comic-slab boom, tested).

Do CGC-graded key comics beat the S&P? We test the strongest tradable version of the
claim: the graded-key-comic price index (hardcoded, cited, **approximate** — a labelled
proxy, not a live feed, because GoCollect gates its indices and Heritage publishes only
per-lot archives) and the only *listed* thing even adjacent to the trade (Funko ``FNKO`` —
CGC/PSA parents and Heritage are private/delisted), all benchmarked against ``SPY`` on
return, volatility, drawdown and — the part the pitch never charges — CGC grading fees +
dealer/auction spread + illiquidity carry.

See :mod:`comic_book_index.data` (hardcoded comic index + the yfinance equity proxy + a
deterministic synthetic bubble control) and :mod:`comic_book_index.strategy` (CAGR/vol/MDD,
annual-excess *t*, Newey-West proxy alpha, and the grading + spread haircut)."""

from . import data, strategy

__all__ = ["data", "strategy"]
