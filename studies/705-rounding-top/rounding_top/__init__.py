"""Study 705 — Rounding Top (dome distribution).

A mechanical, look-ahead-free detector for the "rounding top" / "dome" chart
figure (the bearish mirror of Study 416's rounding bottom), and an honest event
study of forward returns after the *confirmed breakdown* versus the base rate of
shorting a random day in the same names.

Public surface:
    data.synthetic_panel(...)   -> (panel, truth) with a planted-edge (decline) knob
    data.load_real(...)         -> cache-first real OHLC panel via yfinance
    data.fingerprint(...)       -> sha1[:12] content fingerprint
    strategy.run_experiment(...)-> orchestrator returning the headline dict
"""

from . import data, strategy

__all__ = ["data", "strategy"]
