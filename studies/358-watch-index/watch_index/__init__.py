"""Study 358 — "Watches are an asset class" (the luxury-watch-flipping bubble, tested).

Do Rolexes beat the S&P? We test the strongest tradable version of the claim:
the secondary-market resale index (hardcoded, cited, **approximate** — a labelled proxy,
not a live feed) and the only listed ways to buy the trade (Watches of Switzerland
``WOSG.L``, Richemont ``CFR.SW``), all benchmarked against ``SPY`` on return, volatility,
drawdown and — the part the pitch never charges — dealer-spread + illiquidity carry.

See :mod:`watch_index.data` (hardcoded resale index + yfinance equity proxies + a
deterministic synthetic bubble control) and :mod:`watch_index.strategy` (CAGR/vol/MDD,
annual-excess *t*, Newey-West proxy alpha, and the carry haircut)."""

from . import data, strategy

__all__ = ["data", "strategy"]
