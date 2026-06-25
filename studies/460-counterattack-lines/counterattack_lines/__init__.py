"""Study 460 — Counterattack / Meeting Lines.

A mechanical, falsifiable encoding of the Japanese candlestick "counterattack line"
(a.k.a. *meeting line*): two opposite-colour candles whose **closes meet at ~the same
price**, appearing after a trend. The bullish version — a long black candle in a
downtrend, then a long white candle that gaps down on the open but rallies back to
**close at the prior close** — is taught as a reversal signal. We test the bullish
meeting line as a forward-return study against a drift-matched random-entry baseline,
a close-gap scramble placebo, and costs.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
