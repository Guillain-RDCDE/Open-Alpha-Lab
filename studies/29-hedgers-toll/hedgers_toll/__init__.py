"""Study 29 — Hedgers-Toll: are speculators really paid to take the other side of producers' hedges?

The twelfth study mined from Kakushadze & Serur's *151 Trading Strategies* (strategy 9.2, trading on
hedging pressure). The steelman (Keynes' normal backwardation; Cootner 1960; Gorton-Hayashi-Rouwenhorst
2013): when commercial hedgers in a commodity are heavily net short, speculators take the long side and
earn a risk premium -- the "toll" producers pay to offload price risk. We read the hedging pressure off
the CFTC Commitments of Traders report and test whether it predicts commodity futures returns. The
reusable pieces:

    * :mod:`data` — a synthetic commodity panel whose hedging pressure predicts returns by construction
      (plus a null), and a real reader for **CFTC COT** positioning + Yahoo commodity futures (both free).
    * :mod:`hedging` — the engine: the trailing-z-scored hedging-pressure signal and
      :func:`hedging.hedging_premium` (does net hedger short-positioning predict the next-week return?).
    * :mod:`strategy` — the dollar-neutral hedging-pressure book (long where hedgers are net short, short
      where net long), weekly, net of cost, vs the equal-weight commodity basket.
    * :mod:`decompose` — the inference: the long-short premium's **Newey-West t-stat**, its relation to
      the commodity basket, and sub-sample decay.
    * :mod:`extension` — the beat-7 worked complement: a **window-robustness sweep** and a long-only-vs-
      long-short **leg split** (where the premium lives).
"""
