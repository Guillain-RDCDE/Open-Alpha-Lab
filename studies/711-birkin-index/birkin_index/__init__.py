"""Study 711 — "A Birkin beats the S&P (and even gold)" (the Hermès-handbag-as-asset claim).

Does an Hermès Birkin really out-return stocks and gold, as the Baghunter/Credit-Suisse-
cited "14.2%/yr, never a down year" number claims? We test the strongest tradable version:
the secondary-market resale index (hardcoded, cited, **approximate** — a labelled proxy,
not a live feed) and the only listed ways to buy the trade (Hermès ``RMS.PA``, LVMH
``MC.PA``, Kering ``KER.PA``), all benchmarked against ``SPY`` and ``GLD`` on return,
volatility, drawdown and — the part the pitch never charges — a ~30% consignment/dealer
spread + illiquidity carry.

See :mod:`birkin_index.data` (hardcoded resale index + yfinance equity/gold proxies + a
deterministic synthetic compounder control) and :mod:`birkin_index.strategy` (CAGR/vol/MDD,
annual-excess *t*, Newey-West proxy alpha, and the consignment haircut)."""

from . import data, strategy

__all__ = ["data", "strategy"]
