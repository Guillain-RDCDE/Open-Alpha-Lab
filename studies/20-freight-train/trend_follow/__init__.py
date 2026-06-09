"""Study 20 — Freight-Train: does riding the trend across many markets actually pay?

The third study mined from Kakushadze & Serur's *151 Trading Strategies* (strategy 10.4, futures trend
following / time-series momentum). The steelman is one of the most robust facts in empirical finance:
an asset's own past-T return predicts its next return, so sizing a position by ``sign(R_i^T)/sigma_i``
across a diversified basket harvests a real, low-turnover edge (Moskowitz, Ooi & Pedersen 2012). We run
it through the desk's protocol and split, as ever, "is the trend real?" from "can you bank it today?".
The reusable pieces, in the desk's usual split:

    * :mod:`data` — the multi-asset tape: a synthetic generator where each asset's drift is a slow,
      persistent AR(1) (so past returns predict future ones), plus a cache-only reader for a
      diversifying ETF basket. The null (trend_strength=0) is driftless noise where the past predicts
      nothing.
    * :mod:`trend` — the signal and the engine: the trailing-return sign, the inverse-vol scaler, and
      the load-bearing :func:`trend.predictability` -- a pooled time-series-momentum t-stat asking
      whether a positive past return really predicts a positive future one.
    * :mod:`strategy` — the §10.4 book: ``w_i = sign(R_i^T)/sigma_i`` gross-normalised, rebalanced
      monthly (slow signal -> low turnover), net of costs, vs the equal-weight long-only basket.
    * :mod:`decompose` — the inference: a **Newey-West** t-stat that the trend is real, the **alpha vs
      the basket** (genuine timing, not hidden long beta), and **sub-sample / rolling Sharpe** decay
      checks. The verdict it lands depends on the tape -- a real signal whose standalone edge is modest
      and time-varying.
    * :mod:`extension` — the beat-7 worked complement: the **crisis-convexity test** -- does trend pay
      in the months the long-only basket crashes? That diversification, not the standalone Sharpe, is
      what allocators actually buy.
"""
