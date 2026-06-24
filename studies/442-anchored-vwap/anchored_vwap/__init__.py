"""Study 442 — Anchored VWAP (does an anchored level act as a price magnet?).

The folklore (popularised by Brian Shannon): anchor a volume-weighted average price to a
"significant" event — the session open, a swing high/low, an earnings gap, a high-volume bar —
and the resulting **anchored VWAP (AVWAP)** line becomes a *price magnet*. Price supposedly
"respects" the level: it bounces off it (support/resistance), reverts to it when stretched, and
the level acts as a decision line you can trade around.

We test the strongest, cleanest version on intraday tape: anchor VWAP to the **session open**
each day on liquid ETFs/large-caps (yfinance 5-minute bars, ~60 days), then ask whether price
**respects the AVWAP level** — at a touch/cross, does it revert (bounce) or continue? — and
whether it does so any more than at **randomly placed control levels** drawn from the same
session's price range. The honest arbiter is whether the AVWAP-conditioned move beats the
random-level control on a one-sample / HAC ``t`` after a one-bar execution lag and the spread.

See :mod:`anchored_vwap.data` (real 5m bar loader + a deterministic synthetic control with a
planted "magnet" knob) and :mod:`anchored_vwap.strategy` (AVWAP computation, the touch/cross
event detector, the post-touch reversion/continuation measure, random-level control, costs,
HAC ``t`` and a label-shuffle placebo)."""

from . import data, strategy

__all__ = ["data", "strategy"]
