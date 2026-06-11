"""Study 27 — Steamroller: the carry trade picks up nickels — until the steamroller arrives.

The tenth study mined from Kakushadze & Serur's *151 Trading Strategies* (strategy 8.2, the FX carry
trade). The steelman is one of the most durable anomalies in macro: borrow a low-interest currency, lend
a high-interest one, and pocket the gap, because uncovered interest-rate parity (UIRP) fails -- the
high-rate currency does not depreciate enough to offset its yield. We run it through the desk's protocol.
The reusable pieces:

    * :mod:`data` — a synthetic G10 with a *baked* carry premium punctuated by joint risk-off crashes
      (the steamroller), a full-UIRP null, plus the **real reader**: the desk's shared G10 cache (OECD
      MEI 3-month short rates + yfinance FX, the same tape as Study 36), cache-first and offline; a
      missing cache is refilled via ``tools/fetch_altdata.py`` behind ``fetch=True``.
    * :mod:`carry` — the signal (the interest rate) and the engine: the monthly excess-return
      construction and :func:`carry.carry_premium_by_bucket` -- do high-rate currencies out-earn?
    * :mod:`strategy` — the dollar-neutral carry book (long high-rate, short low-rate), monthly, net of
      (low) cost.
    * :mod:`decompose` — the inference: the **carry premium** with a Newey-West t-stat (the `REAL`
      signal), and the **crash profile** -- the negative skew, worst months and drawdown that make it
      `FRAGILE`, with a downside-concentration read.
    * :mod:`extension` — the beat-7 worked complement: **risk-managed carry** -- vol-scaling (the
      [Study 16](../../16-storm-shy/) overlay) to step off the steamroller's path in high-vol regimes.
"""
