"""Study 326 — Stablecoin-Depeg.

Event study around the two canonical stablecoin depegs (UST/Luna, May-2022; USDC,
Mar-2023) on crypto-wide spot (BTC, ETH): is breaking the buck a market-wide sell
signal? See ``data.py`` for the tapes and ``strategy.py`` for the event-study engine.
"""

from . import data, strategy  # noqa: F401
