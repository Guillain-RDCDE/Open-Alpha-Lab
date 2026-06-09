"""Study 15 — Sigma-Sleight: does length-aware "AdaptiveRSI" beat fixed 70/30, or is the σ-transform a relabel?

The reusable pieces, in the desk's usual split:

    * :mod:`data` — the price tape the indicator runs on: a synthetic generator with a
      **baked-in, single-horizon mean reversion** (an Ornstein–Uhlenbeck log-price, so an RSI
      oversold-bounce strategy has a known, modest edge and there is exactly *one* dominant
      horizon — the ground truth the Rescaled-RSI null leans on), plus a cache-only reader that
      pulls real daily SPY/QQQ closes and reduces them to the same single-column tape.
    * :mod:`rsi` — the indicator and the **AdaptiveRSI transform itself**: Wilder's RSI, the
      logit bridge out of the bounded 0–100 scale, and the load-bearing map
      ``σ = logit(RSI) · √(n−1)/2`` that turns a fixed σ landmark into a length-specific RSI
      threshold (this is what makes ``RSI(14)=70`` land at exactly **+1.53σ**), the adaptive
      zone cheat-sheet, and Rescaled RSI (read one length on another's scale).
    * :mod:`signals` — the one strategy the claim implies: RSI mean reversion (enter when RSI
      dips below a lower band, exit back through the 50 equilibrium), expressed two equivalent
      ways — directly in RSI space against a constant level, and in σ space against a landmark —
      so the decomposition can show they are the *same trades*.
    * :mod:`decompose` — the teardown: (1) the σ↔RSI map is strictly monotone, so within a
      length an adaptive σ-zone is a **constant** RSI threshold and produces **identical
      crossings** — the σ dressing adds zero signal; (2) the cheat-sheet zone values are pure
      arithmetic in ``n``, not empirical calibration; (3) a cost-charged horse race of fixed
      70/30 vs adaptive-σ vs a per-length *re-optimised* constant — does σ-calibration beat
      simply picking a sensible length-matched number? (4) does Rescaled long-horizon RSI add
      any forecast power over native short RSI, or nothing on a single-horizon tape?
"""
