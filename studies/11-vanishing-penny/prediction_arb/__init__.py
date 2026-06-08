"""Study 11 — Vanishing-Penny: how fast does a *guaranteed* Polymarket arbitrage close?

The reusable pieces, in the desk's usual split:

    * :mod:`data` — the **gap series** ``g(t) = 1 - (p_yes + p_no)`` per market: a
      synthetic generator with a **baked-in half-life** (offline, deterministic, what
      the tests assert on) and a cache-only reader for real Polymarket CLOB
      ``prices-history``.
    * :mod:`arbitrage` — turn a gap series into **episodes** (an arb opens, then
      decays) and estimate the **half-life** of the mispricing two ways: an
      assumption-light empirical time-to-half, and a pooled log-linear decay fit.
    * :mod:`robustness` — the survival curve, a bootstrap CI on the half-life, the
      **resolution sweep** (what a coarser tape can even *see*), and the
      **retail-capture** fraction (how much of the guaranteed penny is left by the
      time a human reacts).
"""
