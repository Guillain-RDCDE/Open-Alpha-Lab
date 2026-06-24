"""Study 426 — Martin Pring's "Know Sure Thing" (KST) momentum oscillator.

The KST is a long-term momentum summing four smoothed rate-of-change series. The folk
claim: a KST/signal-line crossover times the market better than a plain moving average.
We build it as a long/flat (and long/short) daily timing rule on SPY and race its NET
Sharpe against buy-and-hold (excess-vs-excess), with a HAC t-stat, a permutation placebo,
costs one-way × NAV, a one-day execution lag, and a head-to-head against SMA-200 and MACD.
"""
