"""Study 834 — Minimum Backtest Length (MinTRL).

A research-method demonstration of Bailey, Borwein, López de Prado & Zhu (2014):
given a target annualised Sharpe ratio, there is a **minimum track-record length**
(MinTRL, in years) below which you *cannot* reject the null that the true Sharpe is
<= 0 at a chosen confidence — so a short backtest cannot tell skill from luck.

Synthetic / simulation only. No network, no real market data.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
