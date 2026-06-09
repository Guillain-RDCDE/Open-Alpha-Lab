"""Study 21 — Fools-Gold: does the 'golden cross' actually beat buy-and-hold?

The fourth study mined from Kakushadze & Serur's *151 Trading Strategies* (strategies 3.11-3.13, moving
averages). The steelman is the most famous chart pattern on retail finance TV: when the fast moving
average crosses above the slow one ("golden cross"), the trend is up and you should be long; below
("death cross"), step aside. We run it through the desk's protocol and split, as ever, "is the
crossover informative?" from "does it beat just holding the thing?". The reusable pieces:

    * :mod:`data` — a daily close: a synthetic generator with a slow *persistent drift* (so the price
      trends and a crossover could ride it), plus a cache-only reader for liquid ETFs. The null
      (trend_strength=0) is a driftless random walk where a crossover catches nothing.
    * :mod:`cross` — the signal and the engine: the fast/slow moving averages, the golden/death state,
      and the load-bearing :func:`cross.signal_value` -- do golden-state days really out-return
      death-state days?
    * :mod:`strategy` — the long/flat timing book (long in golden, cash in death) vs buy-and-hold, with
      turnover and a cost sweep.
    * :mod:`decompose` — the inference: a **Newey-West** t on the golden-minus-death spread (the folk
      claim itself), the **alpha & beta vs buy-and-hold** (is the calmer ride just less exposure?), and
      a **risk-matched cash-blend** control. The verdict it lands: the crossover mostly *de-risks* (it
      holds cash), but it does not beat a plain constant lower exposure -- a `WEAK`/`MIRAGE` folk rule.
    * :mod:`extension` — the beat-7 worked complement: the **parameter-robustness test** -- sweep the
      (fast, slow) grid and see whether 50/200 was informative or just the luckiest cell.
"""
