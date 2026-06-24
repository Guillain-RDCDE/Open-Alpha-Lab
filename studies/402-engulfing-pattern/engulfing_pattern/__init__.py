"""Study 402 — Engulfing Pattern.

Detect the bullish/bearish engulfing candlestick across a fixed basket of liquid US
large-caps + SPY, then ask the only question that matters: does the next day (and the
next 3/5/10 days) actually go the way the textbook promises, versus the unconditional
base rate?

Public surface:

* ``data`` — synthetic_panel (planted-edge knob) + load_real (cache-first yfinance) +
  fingerprint.
* ``strategy`` — engulfing detector (precise OHLC rules), forward-return event study,
  one-sample / HAC inference, label-shuffle placebo, costs, the trend/volume myth-check,
  and a run_experiment orchestrator.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
