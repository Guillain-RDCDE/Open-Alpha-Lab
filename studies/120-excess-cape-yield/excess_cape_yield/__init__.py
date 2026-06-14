"""Study 120 — Excess-CAPE-Yield (ECY).

Shiller's ECY (= 1/CAPE minus the real 10-year bond yield) as a long-horizon
forecaster of the 10-year equity-minus-bond excess return.  Submodules:

* :mod:`excess_cape_yield.data`     — synthetic generator + Shiller parquet loader
* :mod:`excess_cape_yield.strategy` — regression, bucketed returns, OOS split
"""
