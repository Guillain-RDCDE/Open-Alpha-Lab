# References & literature map — Study 805 (Cokurtosis Premium)

## The claim under test

- **The source paper.** Hongshan **Fang & Tsong-Yue Lai**, *"Co-Kurtosis and Capital
  Asset Pricing"* (The Financial Review, 1997). They extend the CAPM to its **fourth**
  moment: alongside beta (systematic variance) and co-skewness (systematic skewness), a
  security's **cokurtosis with the market** — how strongly its return co-moves with the
  *cube* of the market's deviation — is a priced source of risk. A name that spikes
  precisely when the market has a fat-tailed move loads on **market tail-co-movement**, an
  undesirable exposure, so investors demand a **positive** premium to hold high-cokurtosis
  names.
- **The four-moment CAPM tradition.** The idea sits in a long line: **Kraus & Litzenberger
  (1976)** priced systematic **co-skewness**; **Fang & Lai (1997)** and **Dittmar (2002)**
  add systematic **co-kurtosis**; the pricing-kernel reading is a Taylor expansion of
  marginal utility in which investors dislike variance and kurtosis and like positive
  skewness. Empirically the higher co-moments are notoriously fragile out of sample.
- **The behavioural / risk reading.** High cokurtosis = the stock's biggest moves line up
  with the market's most extreme (tail) days, offering no diversification exactly when it
  is most wanted. Rational pricing says such names must offer a higher expected return.
- **The specific test here.** We take the self-contained daily version: for each name,
  the **trailing standardised cokurtosis with the equal-weight market**
  `E[(r_i-μ_i)(r_m-μ_m)^3] / (σ_i·σ_m^3)` over a 252-day window, then measure the forward
  return of the equal-weight **long-high-cokurtosis / short-low-cokurtosis** book, with a
  Newey-West *t*, a permutation placebo, a two-era robustness cut, a costed timer, and a
  seeded synthetic positive control.

## What we measure, and the honesty rails

- **Cokurtosis, a fourth-order co-moment.** One power on the name's centred return, **three**
  on the centred market return; standardised by `σ_i·σ_m^3`. The market is the
  **equal-weight cross-sectional mean** of the panel. Computed vectorised via rolling raw
  moments (`E[r_i·r_m^3]`, `E[r_i·r_m^2]`, `E[r_i·r_m]`, and the market's own moments) — no
  per-date python loop for the signal.
- **Point-in-time sort, one documented lag.** The ranking signal is the trailing cokurtosis
  **known at the close of `t-1`** (`.shift(1)`); the book is held on day `t`. Zero look-ahead.
- **Robust inference.** Newey-West (HAC, Bartlett, 10-lag) *t* on the daily long-short
  spread — an overlapping-formation signal is serially correlated, so a plain *t* would
  overstate significance. A one-sample *t* and a pooled Welch *t* (high-cokurt book vs
  low-cokurt book) cross-check. A **1,000-permutation placebo** breaks the
  signal → forward-return link to confirm the spread isn't a lucky alignment of the sort.
- **Survivorship is named on the Signal axis.** The universe is a **current-membership** set
  of ~50 liquid mega-caps, pulled through the `quantlab.universe` survivorship guard
  (`allow_survivorship_bias=True`, an explicit opt-in). Delisted / de-rated names are absent,
  so the cross-sectional magnitudes are an **upper bound**.
- **The timer is graded separately.** Costs are one-way × NAV on the long-short book, and the
  short book pays borrow — the honest test of whether a small daily spread survives friction.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* used on the spread series).
- **Wilson, E. B. (1927)** — score interval for a binomial share.
- **Kraus, A. & Litzenberger, R. (1976)** — skewness preference and the three-moment CAPM
  (the co-skewness predecessor tested in study 504).
- **Dittmar, R. (2002)** — nonlinear pricing kernels and cubic marginal utility, the modern
  cokurtosis pricing reference.

## Data sources

- **yfinance daily OHLC** (`auto_adjust=True`, total-return), 50 liquid US large-caps,
  2010-01-04 → 2026-06-30, cached under `_cache/` through `quantlab.universe`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [504-coskewness](../../504-coskewness/) — **systematic co-skewness**, the *third*-order
  co-moment `E[(r_i-μ_i)(r_m-μ_m)^2]` (one power on the name, **two** on the market). This
  study uses the *fourth*-order co-moment with the market **cubed** — one higher power, a
  different tail exposure, and a different paper (Fang-Lai vs Kraus-Litzenberger).
- [238-betting-against-beta](../../238-betting-against-beta/) — the low-beta / BAB trade, a
  sort on the *second*-order co-moment (co-**variance**, i.e. beta). Cokurtosis is a
  higher-moment exposure, not a leverage-constraint beta tilt.
- [803-realized-skewness-reversal](../../803-realized-skewness-reversal/) — a name's **own**
  total realized skewness (a stand-alone third moment read off its own tape), not its
  **co**-movement with the market at all.

None of the siblings sort on the **standardised fourth co-moment of a name's return with
the market** — the Fang-Lai systematic-kurtosis signal — which is this study's own axis.
