# References & literature map — Study 483 (Zero-Lag EMA)

## The claim under test

- **The folklore.** A plain exponential moving average lags price by ~ `(L-1)/2` bars. The
  *zero-lag EMA* (ZLEMA) removes that lag by smoothing a **de-lagged** input,
  `close + (close - close[lag])`, which extrapolates the recent move forward so the line
  "catches up" to price. The pitch is that a less-laggy filter is a *timelier* one: "long while
  price > ZLEMA" gets you into trends sooner and out sooner, so it should beat a plain EMA of the
  same length. ZLEMA ships in most charting suites (TradingView, MetaTrader, Thinkorswim) and is
  a staple of "zero-lag / instantaneous trendline" write-ups.
- **The source.** **John F. Ehlers** is the originator of the zero-lag / de-lagged moving-average
  family, developed in his DSP-for-traders work — *Rocket Science for Traders* (2001), *Cybernetic
  Analysis for Stocks and Futures* (2004), and the *Stocks & Commodities* articles on the
  "zero-lag" and "instantaneous trendline" filters. The same `price + (price - price[lag])`
  de-lag trick underlies Patrick Mulloy's earlier **DEMA/TEMA** (Mulloy, *Technical Analysis of
  Stocks & Commodities*, 1994) and Tushar Chande's **Variable Index Dynamic Average** — all are
  attempts to subtract the EMA's phase lag.
- **Why "zero lag" is a half-truth.** Removing phase lag at one frequency necessarily *amplifies*
  high-frequency noise (a basic filter-theory trade-off): the de-lag is a high-pass term, so the
  ZLEMA overshoots and whipsaws on noise. You don't get something for nothing — less lag is paid
  for in extra noise, which is exactly what the upcross foil and the head-to-head expose here.

## Why a high one-sample t is not evidence

- **Drift / beta.** US equity indices have a positive unconditional daily mean; a one-sample
  *t* of a long-only filter against **zero** measures that drift, not the rule. The desk's
  standing rule is *signal-vs-baseline*, never *signal-vs-zero* — here the baseline is a
  drift-matched **random-entry** control, and ZLEMA loses to it at every horizon.
- **The right benchmark is the thing it claims to beat.** ZLEMA's specific promise is "better
  than a plain EMA", so the load-bearing test is the **ZLEMA-minus-EMA** head-to-head (Δ_ema),
  not the level. That delta is tiny and mixed — the de-lag adds nothing.
- **Data snooping on chart tools.** Lo, Mamaysky & Wang (2000, *Foundations of Technical
  Analysis*, Journal of Finance) formalize testing chart/indicator rules against a properly
  matched null; Sullivan, Timmermann & White (1999, *Data-Snooping, Technical Trading Rule
  Performance, and the Bootstrap*, JF) and White (2000, *A Reality Check for Data Snooping*,
  Econometrica) show how trend-fitted rules manufacture significance unless raced against a fair
  benchmark.
- **HAC inference.** Newey & West (1987) standard errors for the one-sample mean; Welch (1947)
  two-sample *t* for the ZLEMA-vs-random difference.

## Method lineage (the desk's shared engine)

- **Causal ZLEMA + plain EMA.** [`strategy.zlema`](../zlema/strategy.py),
  [`strategy.ema`](../zlema/strategy.py) — the de-lagged filter and the head-to-head baseline.
- **Folklore filter + upcross foil.** [`strategy.zlema_entries`](../zlema/strategy.py) (sampled
  `price > ZLEMA` state), [`strategy.zlema_upcross_entries`](../zlema/strategy.py) (the whippy
  cross), [`strategy.ema_entries`](../zlema/strategy.py) (the plain-EMA head-to-head).
- **Forward-return + HAC t + random baseline.** [`strategy.forward_returns`](../zlema/strategy.py),
  [`strategy.hac_t`](../zlema/strategy.py), [`strategy.run_experiment`](../zlema/strategy.py).
- **De-lag placebo.** [`strategy.delag_placebo`](../zlema/strategy.py) — permute the de-lag
  offset, keep its marginal, destroy its alignment.
- **Deterministic synthetic control.** [`data.synthetic_panel`](../zlema/data.py) plants a
  persistent trend regime (knob `edge`); with `edge = 0` the detector must NOT manufacture
  significance — the offline core runs with no network.

## Data sources used here

- **yfinance** daily adjusted (total-return) closes for SPY, QQQ, IWM, DIA, GLD,
  2005-01-03 → 2026-05-29 (As-of 2026-05-31, partial June dropped), cached as parquet under
  `_cache/`. All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced
  by [`examples/verify.py`](../examples/verify.py).

## Related desk studies

- [`../../438-triple-ma`](../../438-triple-ma) and [`../../108-faber-timing`](../../108-faber-timing)
  — the broader moving-average / trend-filter family; most land None × Mirage for the same reason.
- [`../../432-hma`](../../432-hma), [`../../433-kama`](../../433-kama),
  [`../../434-dema-tema`](../../434-dema-tema) — sibling "improved / low-lag" averages built on the
  same de-lag idea; the natural cousins of this teardown.
- The **research-method demos** (data-mining-roulette, look-ahead, curve-fitting) frame why a
  signal-vs-zero *t* is not enough; ZLEMA is a clean live example of beta plus a cosmetic de-lag.
