"""Study 431 — Schaff Trend Cycle (STC).

A "faster MACD": Doug Schaff's 2008 oscillator runs a stochastic over the MACD line,
double-smooths it, and bounds it to 0..100. The folk claim is that it turns earlier than
the MACD and so makes a more profitable trend-timing rule. We turn it into a daily
long/flat (and long/short) timing rule on SPY, race its NET Sharpe against buy-and-hold
excess-vs-excess, and pit it head-to-head against the plain MACD it claims to beat.
"""

from . import data, strategy  # noqa: F401
