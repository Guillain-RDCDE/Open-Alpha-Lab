"""Fear-Gauge — does buying the VIX spike actually pay?

A reproducible, honest stress-test of the "buy stocks when the VIX is high / just
spiked" idea — the twin, in volatility space, of Study 02 (Falling-Knife). We do
not test *one* rule; we test a *family*: the Prof's level rule (VIX >= 30, double
down at 50) and Altucher's one-day-spike chart, split by regime. Each is measured
against the only yardsticks that matter: does it beat buying on a random day, does
it add anything over the price drop, and does it survive once you charge realistic
costs and rerun on a sample that includes 2008.

Public surface (import from the sub-modules):
    from fear_gauge import data, triggers, exits, eventstudy
    from fear_gauge import benchmark, backtest, robustness, plots

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
]
