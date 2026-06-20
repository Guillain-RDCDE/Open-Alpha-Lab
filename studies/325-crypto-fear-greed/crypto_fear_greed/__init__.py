"""Study 325 -- Crypto-Fear-Greed.

A contrarian timing test on Bitcoin using a *reconstructed* Fear & Greed gauge.

The live alternative.me Crypto Fear & Greed API is network-blocked here, so we
do not pretend to have its tick-exact series. Instead we rebuild a transparent
PROXY gauge from BTC's own observables -- realised volatility, drawdown from the
trailing high, and trailing momentum -- the same ingredients the public crypto
F&G index leans on -- and test the folk rule on it: *be greedy when others are
fearful.* See ``data.py`` (the proxy gauge + synthetic control) and
``strategy.py`` (the contrarian overlay + honest controls).
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
