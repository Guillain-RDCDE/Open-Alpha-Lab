"""Study 24 — Stampede: do past winners keep winning, and what does the herd cost when it turns?

The seventh study mined from Kakushadze & Serur's *151 Trading Strategies* (strategy 3.1, price
momentum). The steelman is the most robust anomaly in finance: rank stocks by their trailing 12-1
return, go long the winners and short the losers, and the spread earns a large, significant premium
(Jegadeesh & Titman 1993). We run it through the desk's protocol and split, as ever, "is the premium
real?" from "can you live with how it pays?". The reusable pieces:

    * :mod:`data` — a synthetic panel where each stock carries a *persistent* relative-performance drift
      (so past winners keep winning), plus a cache-only reader for the current S&P 500 cross-section via
      :mod:`quantlab.universe`. The null (mom_strength=0) has no persistence -- past rank predicts
      nothing.
    * :mod:`momentum` — the signal and the engine: the 12-1 trailing return and the load-bearing
      :func:`momentum.momentum_spread` -- sort into deciles, read each decile's forward return, and ask
      whether winners really out-earn losers.
    * :mod:`strategy` — the dollar-neutral WML (winners-minus-losers) factor and the long-only winners
      book vs the equal-weight market, monthly rebalance, net of cost.
    * :mod:`decompose` — the inference: the **CAPM alpha** with HAC errors (the `REAL` signal), the
      **crash profile** (WML's fat negative skew and the loser-rebound crash that makes it `FRAGILE`),
      and sub-sample decay + a bootstrap CI. The verdict it lands: Signal `REAL`, Tradability `FRAGILE`,
      Crash risk `SEVERE` -- a genuine premium with a catastrophic left tail.
    * :mod:`extension` — the beat-7 worked complement: **risk-managed momentum** -- vol-scaling the WML
      factor (Barroso-Santa-Clara, the [Study 16](../../16-storm-shy/) overlay) to tame the crash.
"""
