"""Study 295 -- Stablecoin-Supply.

Does stablecoin supply growth fuel the next leg of Bitcoin?

The ``data`` module ships a curated monthly total-stablecoin-supply series
(USDT + USDC + DAI + BUSD + the long tail), plus a deterministic synthetic
generator with a planted "dry-powder" knob and a cache-only BTC-USD tape.
The ``strategy`` module turns supply growth into a forward-return timing
signal and pins it against buy-and-hold with HAC t-stats and honest costs.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
