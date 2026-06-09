"""Study 14 — Gamma-Gospel: does dealer gamma foretell the day's character, or is it the VIX in a trenchcoat?

The reusable pieces, in the desk's usual split:

    * :mod:`data` — the **regime panel**, one row per trading day carrying the signed
      gamma exposure (GEX) measured at the *prior* close and the day's realised character
      (a range-based vol estimate and a directional-efficiency / trend-vs-chop ratio), plus the
      prior-close VIX that is the confound the whole study turns on. A synthetic generator bakes
      in a known genuine gamma effect ``beta`` *and* a VIX-driven confound (VIX moves both GEX
      sign and realised character), so the tests have a ground truth; a cache-only reader builds
      the same panel from a real SPY option chain (Alpha Vantage ``HISTORICAL_OPTIONS``), the
      :func:`compute_gex` reducer, and daily SPY/VIX bars.
    * :mod:`signals` — the regime labels the GEX pitch trades on (``neg_gamma`` = dealers short
      gamma = the "amplifier"; ``pos_gamma`` = long gamma = the "shock absorber") and the two
      realised outcomes a study of the claim must keep apart: a vol estimate and a
      directional-efficiency ratio.
    * :mod:`decompose` — the teardown: the raw regime gap (does negative gamma really print more
      vol / more trend?) and then the load-bearing move — a HAC-robust nested regression that
      asks whether the GEX sign adds **anything over the VIX level**, or whether it is the
      volatility regime wearing a trenchcoat.
"""
