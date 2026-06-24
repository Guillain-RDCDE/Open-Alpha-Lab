"""Study 416 — Rounding Bottom (saucer base).

A mechanical, look-ahead-free detector for the "rounding bottom" / "saucer base"
chart figure, and an honest event study of forward returns after the *confirmed
breakout* versus the unconditional base rate.

Public surface:
    data.synthetic_panel(...)   -> (panel, truth) with a planted-edge knob
    data.load_real(...)         -> cache-first real OHLC panel via yfinance
    data.fingerprint(...)       -> sha1[:12] content fingerprint
    strategy.run_experiment(...)-> orchestrator returning the headline dict
"""

from . import data, strategy

__all__ = ["data", "strategy"]
