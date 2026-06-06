"""Falling-Knife — does buying the Nasdaq-100 dip actually pay?

A reproducible, honest stress-test of the "buy NDX after it drops ~3%" idea.
We do not test *one* rule; we test a *family* of rules (four ways to define the
falling knife x a grid of exits) and we measure each against the only yardstick
that matters: does it beat buying on a random day, after realistic costs, and
does it survive when you split history by regime?

Public surface (import from the sub-modules):
    from falling_knife import data, triggers, exits, eventstudy
    from falling_knife import benchmark, backtest, robustness, plots

This package is a research and teaching tool. It is NOT investment advice.
"""

__version__ = "0.1.0"

__all__ = [
    "data",
    "triggers",
    "exits",
    "eventstudy",
    "benchmark",
    "backtest",
    "robustness",
    "sweeps",
    "plots",
]
