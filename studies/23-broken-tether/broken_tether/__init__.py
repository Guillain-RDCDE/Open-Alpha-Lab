"""Study 23 — Broken-Tether: do two assets that move together stay tethered enough to trade?

The sixth study mined from Kakushadze & Serur's *151 Trading Strategies* (strategy 3.8, pairs trading).
The steelman is the oldest statistical-arbitrage trade: find two cointegrated assets, and when their
spread stretches far from its mean, bet on reversion -- short the rich leg, long the cheap one, close
when it snaps back. We run it through the desk's protocol and split, as ever, "is the spread really
mean-reverting?" from "does the relationship survive out of sample?". The reusable pieces:

    * :mod:`data` — two log-prices sharing a trend with a spread that mean-reverts (cointegrated) or
      wanders (spurious), plus a cache-only ETF reader to form real pairs.
    * :mod:`spread` — the engine: the hedge ratio, the spread, its rolling z-score, and the load-bearing
      :func:`spread.half_life` -- a short half-life means a tradable, stationary spread; a near-infinite
      one means a random walk with nothing to revert to.
    * :mod:`strategy` — the causal z-score pairs book (rolling hedge ratio and z-score, enter at ±entry,
      exit near the mean), net of two-leg costs.
    * :mod:`decompose` — the inference: the **in-sample vs out-of-sample** split (does the pair work in
      both halves, or snap?) and the **spurious-pairs** false-positive rate (how many *independent*
      random walks look cointegrated by chance -- the selection trap). The verdict depends on the tape:
      a baked pair is real; real ETF pairs tend to be `FRAGILE`/`MIRAGE` -- they break out of sample and
      most apparent cointegration is selection.
    * :mod:`extension` — the beat-7 worked complement: the **hedge-ratio drift** -- the tether quietly
      stretches as the relationship moves -- and an out-of-sample scan across candidate pairs.
"""
