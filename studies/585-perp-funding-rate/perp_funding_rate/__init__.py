"""Study 585 — Perp-Funding-Rate.

Extreme perpetual-swap **funding rates** as a contrarian / mean-reversion signal for crypto
forward returns: when funding runs hot (longs pay shorts, crowd is long-and-levered), are the
longs over-positioned and due for a flush?

The core is **synthetic-only** by necessity: free retail data (yfinance) has BTC/ETH *prices* but
NOT the historical **funding-rate** tape, which lives behind paid exchange / derivatives-data APIs
(Binance, Bybit, Coinglass, Amberdata...). So the machinery is validated on a deterministic
synthetic world with a single planted-effect knob, and the data-availability limitation is stated
openly on the SIGNAL axis. A synthetic-only study can never earn `REAL` (that needs a robust
*t* >= 2 on a real tape).
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
