"""Study 797 — FX Value (PPP real-exchange-rate mean reversion).

The FX **value** factor: rank G10 currencies on how far their *real* exchange rate
(nominal FX deflated by relative CPI) sits below/above its own long trailing average,
go long the cheap (undervalued) and short the rich (overvalued). The opposite tilt to
carry (sibling 364), and a real *time series* of the PPP gap — not a single Big-Mac
folklore snapshot (sibling 215).
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
