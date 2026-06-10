# References & literature map — Study 31 (Trade-Winds)

## The claim under test — the steelman

Time-series momentum (a.k.a. trend-following / managed futures) is the rare anomaly with a century of
out-of-sample evidence and a clear economic story:

- **The foundational result.** Tobias Moskowitz, Yao Hua Ooi & Lasse Pedersen, *"Time Series Momentum"*,
  **Journal of Financial Economics** 104(2), 2012: across 58 liquid futures, a market's own past 12-month
  excess return predicts its next-month return, long-the-risers/short-the-fallers earns a significant
  premium, and the portfolio delivers positive returns in equity bear markets.
- **A century of evidence.** Brian Hurst, Yao Hua Ooi & Lasse Pedersen, *"A Century of Evidence on
  Trend-Following Investing"*, **Journal of Portfolio Management** 2017: the premium holds back to 1880
  across asset classes, with consistent **crisis alpha** (positive in the worst equity quarters).
- **The behavioural / structural story.** Underreaction to information then delayed overreaction
  (Hong & Stein 1999); the convex, long-volatility payoff of trend (Fung & Hsieh 2001, *"The Risk in
  Hedge Fund Strategies"*, **Review of Financial Studies**) explains the positive skew and the crisis alpha.

## The honest counters — why the verdict is `REAL` / `FRAGILE` / `CONFIRMED`

- **Standalone tradability is fragile.** On our 18-market, 2000–2026 sample the net-of-cost book (Sharpe
  0.30) trails the always-long, equal-risk basket of the *same* markets (0.51): the diversification does
  more work than the trend timing. The well-documented **2011–2019 "trend drought"** (e.g. AQR's own
  commentary; Babu, Hoffman, Levine, Ooi, Stamelos & Stein, *"You Can't Always Trend When You Want"*,
  **JPM** 2020) shows up cleanly in the sub-period Sharpe (−0.28 in the middle third).
- **The value is as a diversifier, not a return engine.** Because the book is ~−0.07 correlated to
  equities with a positive-skew payoff, blending a sleeve of it *raises* a 60/40's Sharpe (0.48→0.56)
  and halves its drawdown (−34%→−20%). That is the `CONFIRMED` crisis-alpha use case, and the honest
  reason to run it — exactly the diversification argument, not a market-timing one.
- **Implementation caveats.** Real managed futures uses more markets (50–100+), roll-aware continuous
  contracts, and dynamic vol targeting; our 18-market Yahoo basket is a faithful but thinner proxy. The
  conclusion (real premium, fragile standalone, valuable as crisis-alpha diversifier) is the robust part.

## The desk's own method — engine and reproducibility

- **Data.** Yahoo continuous front-month futures (18 markets, four asset classes), daily, 2000–2026.
  Daily returns clipped to ±25% (data hygiene). Pinned with [`quantlab.repro`](../../../quantlab/repro.py)
  (as-of + input fingerprint). The offline control is a seeded synthetic regime-switching trend panel
  (`trade_winds.data.synthetic_trends`) plus a random-walk null.
- **Strategy.** `trade_winds.strategy` — blended 1/3/12-month TSMOM sign, per-market vol-scaling to
  equal risk, portfolio vol target; `trade_winds.costs` adds the cost sweep, the long-only benchmark and
  the crisis-alpha measurement.

## Caveats stated in the open (house rule)

- **±25% return clip.** Continuous-futures feeds carry roll glitches and the Apr-2020 negative-WTI print
  (CL went to −$37, which makes a percentage return meaningless); a real future is halted long before a
  clean ±25% day. Stated, not hidden.
- **Front-month continuous, not roll-yield-exact.** A simplification of the true rolled return; the trend
  signal is slow (monthly-ish), so daily front-month returns are an adequate, stated approximation.

---

*Part of [Open-Alpha-Lab](../../../README.md). Not investment advice — research and education.*
