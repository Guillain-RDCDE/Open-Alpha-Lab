"""Study 674 — VIDYA (Chande's Variable Index Dynamic Average).

Two modules:

* :mod:`vidya.data` — the real five-ticker basket (yfinance, cache-first) and a
  deterministic synthetic positive control.
* :mod:`vidya.strategy` — the indicator, the mechanism checks (does the smoothing
  constant actually speed up in trending/volatile regimes?), the long/flat timing
  rules (VIDYA vs SMA vs EMA vs buy-and-hold) and the inference machinery (HAC *t*,
  permutation placebo).
"""
