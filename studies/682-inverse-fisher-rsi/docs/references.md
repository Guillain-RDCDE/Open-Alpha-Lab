# References & literature map — Study 682 (Inverse-Fisher-RSI)

## The claim under test

- **The folklore.** Ordinary RSI is bounded in [0, 100] but spends most of its time bunched in
  the middle and *lingers* near its extremes instead of snapping — its 30/70 (or 20/80)
  crossovers are a mushy, delayed tell. John F. Ehlers' fix: run RSI through the **Inverse
  Fisher Transform** (the inverse of the same atanh-based Fisher Transform tested on raw price
  in sibling [183-fisher-transform](../183-fisher-transform/)), which compresses the signal
  toward its bounds ±1 and makes turning points look decisive and "snappy" on a chart.
- **The primary source.** Ehlers, *"Using The Fisher Transform"*, Technical Analysis of Stocks
  & Commodities, November 2002 — introduces the transform. The RSI-specific recipe used here
  (`0.1*(RSI(5)-50)` smoothed with a 9-bar WMA, then passed through the inverse Fisher
  function) is Ehlers' own worked example, published again in *Cybernetic Analysis for Stocks
  and Futures* (Wiley, 2004), Ch. 1, and widely reproduced on retail charting platforms
  (TradingView's "Inverse Fisher Transform on RSI" indicator implements the identical formula).
- **The underlying oscillator.** J. Welles Wilder Jr., *New Concepts in Technical Trading
  Systems* (1978) — the original RSI, and the Wilder-smoothed average gain/loss this study (and
  every RSI-family sibling on the desk) computes it with.
- **The adjacent (distinct) claims.** Connors & Alvarez, *Short Term Trading Strategies That
  Work* (2008/2009) — RSI(2) mean reversion, no Fisher transform involved (sibling
  [75-knee-jerk](../75-knee-jerk/)). Chande & Kroll, *The New Technical Trader* (1994) —
  Stochastic-of-RSI, a *different* second-transform recipe (sibling
  [428-stochastic-rsi](../428-stochastic-rsi/)). None of these apply the Fisher machinery.

## What we measure, and the honesty rails

- **The exact Ehlers recipe**, not a hand-tuned variant: `RSI(5)` (Wilder-smoothed) →
  `v1 = 0.1*(RSI-50)` → `v2 = WMA(v1, 9)` → `IFT-RSI = tanh(v2) = (exp(2v2)-1)/(exp(2v2)+1)`.
  Bullish signal = cross UP through -0.5; bearish = cross DOWN through +0.5 — Ehlers' own
  published thresholds.
- **Three comparisons, one machinery.** Signal-conditional forward returns (5/10/20-day, one
  execution lag — signal known at close *t*, return earned from close *t+1*) vs (a) the
  **unconditional** forward-return distribution of the same six tickers, (b) a **random-signal
  placebo** of matched count, many seeds, and (c) **plain RSI(14)** and **plain RSI(2)**
  reversal signals run through the identical event-study code — the fairest possible test of
  "does the transform *add* anything."
- **A timer with costs** turns the IFT-RSI crossover into an executable long-flat rule, charged
  2 × one-way cost × NAV per round trip, and benchmarked against buy-and-hold *and* a
  random-exposure control matched on time-in-market — not just the gross signal.
- **Synthetic control at the honest horizon.** The AR(1) reversion knob used for the positive
  control operates at a single 1-day lag by construction, so the machinery check runs at h=1
  (where the planted effect lives), separately from the headline h=5/10/20 — documented so the
  choice reads as a design decision, not a snooped horizon that happens to work.

## Data sources

- **Daily total-return closes** (`auto_adjust=True`, dividend + split adjusted) for SPY, QQQ,
  IWM, AAPL, MSFT, NVDA — yfinance (no key), cached under `_cache/` (`ifrsi_<ticker>.csv`),
  2010-01-04 → 2026-06-30. Same universe as sibling
  [669-rsi-divergence](../669-rsi-divergence/) for direct RSI-family comparability.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [183-fisher-transform](../../183-fisher-transform/) — the **plain** Fisher Transform applied
  directly to **price** (not RSI), and proven mathematically monotone in the normalised close
  (its crossovers exactly coincide with a raw-price crossover). This study applies the
  **inverse** transform to **RSI**, a different input and a different (non-monotone,
  bounded-oscillator) transform — the two studies test genuinely different mechanisms, and,
  read together, neither one earns the "sharper" story.
- [75-knee-jerk](../../75-knee-jerk/) — Connors' **RSI(2)** mean-reversion system on its own
  terms (the desk's full treatment of that indicator, `t = +10.70` pooled — one of the desk's
  clearest edges). This study uses plain RSI(2) crossover only as a *comparison baseline* for
  IFT-RSI's own forward-return test — it does not re-run 75's full protocol (no 200-SMA filter,
  no publication-decay split); see 75 for the definitive RSI(2) verdict.
- [428-stochastic-rsi](../../428-stochastic-rsi/) — a **different** second transform
  (Stochastic-of-RSI, not Fisher-of-RSI), tested as a long-flat SPY timer (`Signal: NONE`,
  timing worse than a coin). Complementary "does stacking help" teardown, different math.
- [669-rsi-divergence](../../669-rsi-divergence/) — a **structural** RSI pattern (price/RSI
  swing-low divergence over two confirmed lows), not a threshold crossover — different signal
  definition entirely, same universe/basket for comparability.

None of the siblings apply Ehlers' Inverse Fisher Transform to RSI and test its crossover
thresholds against the plain-RSI baselines head-to-head — that is this study's own axis.
