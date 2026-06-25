"""Study 451 — Marubozu (the full-bodied, wickless candle).

A *marubozu* ("bald head" / "shaven" candle) is a session whose real body fills almost the
entire high-low range: a **bullish** marubozu opens ≈ at the low and closes ≈ at the high
(no wicks), a **bearish** one opens ≈ at the high and closes ≈ at the low. Candlestick lore
(Nison, every chart-pattern site) reads it as a sign of *decisive, one-way pressure* that
**continues**: a bullish marubozu is a buy. We encode that as a falsifiable forward-return
study against a drift-matched random-entry baseline and a body-shuffle placebo, with costs.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
