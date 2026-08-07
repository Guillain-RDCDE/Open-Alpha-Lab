# References & literature map — Study 829 (Global Sovereign-Bond Momentum)

## The claim under test

- **The source paper.** Tobias J. **Moskowitz, Yao Hua Ooi & Lasse Heje Pedersen**,
  *"Time Series Momentum"* (Journal of Financial Economics, 2012). Across 58 liquid
  instruments spanning equity indices, currencies, commodities **and government bonds**,
  a market's own past 12-month excess return positively predicts its next-month return;
  a diversified time-series-momentum (trend) strategy that goes long recent winners and
  short recent losers earns a large, persistent, and largely factor-orthogonal premium.
- **The bond-specific reading.** In fixed income, slow-moving policy-rate and
  term-premium cycles produce *level* trends — multi-year bond bulls and bears — so the
  sign of the trailing 12-1 total return is, in principle, an informative state variable
  for the next month's return. This study asks whether that shows up on *tradable global
  sovereign-bond ETFs* once you cost it.
- **The 12-minus-1 convention.** Following the momentum literature (Jegadeesh & Titman
  1993; Asness, Moskowitz & Pedersen 2013, *"Value and Momentum Everywhere"*), the trend
  is measured over twelve months **skipping the most recent month** (the "12-1" window),
  which sidesteps the short-horizon reversal that contaminates a raw 12-month signal.
- **The specific test here.** Each of five global sovereign-bond ETFs is signed by its own
  12-1 trend known at the close of month `t−1` and held over month `t`; the equal-weight
  strategy return is scored with a Newey-West *t*, a block-rotation placebo, a three-era
  robustness cut, a lookback sweep, a costed backtest, and a 20-seed synthetic control.

## What we measure, and the honesty rails

- **Trend, no free model.** `mom_t = level_{t-1}/level_{t-12} − 1` on month-end
  total-return levels — a pure price-based signal, vectorised as a ratio of shifted frames.
- **Point-in-time positions, one documented lag.** The position is the **sign** of the
  momentum **known at the close of `t−1`** (`.shift(1)`); it is held over month `t`. Zero
  look-ahead (verified in the test-suite by a future-shock invariance check).
- **The right benchmark.** A trend edge must beat *just holding the bonds*; we report the
  naive equal-weight buy-and-hold Sharpe alongside every headline. On this tape the
  long-only "trend" under-earns buy-and-hold — the tell that it is bond beta.
- **Robust inference.** Newey-West (HAC, Bartlett, 6-lag) *t* on the monthly return; a
  one-sample *t* cross-checks; a **3,000-draw block-rotation placebo** breaks the
  trend → forward-return link while preserving each market's own autocorrelation, to test
  whether the signal carries information beyond the return series' own drift.
- **Survivorship named on the Signal axis, power named too.** Only currently listed funds
  enter, so a positive result is a mild upper bound; and the five-ETF panel is small by
  construction, which limits power — stated with the numbers, not buried.
- **The backtest is costed separately.** One-way turnover cost per rebalance plus borrow
  on the short leg — the honest test of whether a thin monthly edge survives friction.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* used on the monthly strategy return).
- **Jegadeesh, N. & Titman, S. (1993)** — the original momentum "skip a month" convention.
- **Asness, C., Moskowitz, T. & Pedersen, L. (2013)** — *"Value and Momentum Everywhere"*,
  the cross-asset framing that includes global government bonds.
- **Wilson, E. B. (1927)** — score interval for a binomial share (hit-rate uncertainty).

## Data sources

- **yfinance daily closes** (`auto_adjust=True`, total-return), five global sovereign-bond
  ETFs (`BWX`, `IGOV`, `BNDX`, `EMB`, `IEF`), resampled to month-end, 2007-01-31 →
  2026-06-30, cached under `_cache/`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [795-corporate-bond-momentum](../../795-corporate-bond-momentum/) — a **cross-sectional**
  momentum sort within a **corporate** bond universe (rank winners vs losers). This study
  is **time-series** (each market signed against its *own* trend, not a cross-sectional
  rank) on **foreign / global sovereigns**.
- [518-time-series-momentum](../../518-time-series-momentum/) — the **general** TSMOM factor
  pooled across broad asset classes. This study isolates the **global-sovereign-bond**
  sleeve on its own tradable ETF tape and costs it standalone.
- [662-em-local-bonds](../../662-em-local-bonds/) — an EM **carry / level** study, not a
  **trend** signal.
- [247-bond-seasonality](../../247-bond-seasonality/) — a **calendar** effect in bonds, not
  a **momentum / trend** signal.

None of the siblings sign a **global sovereign-bond ETF against its own 12-1 total-return
trend** — the time-series-momentum axis this study tests.
