# References & literature map — Study 898 (Managed-Vol Equity)

## The claim under test

- **The source paper.** Alan **Moreira & Tyler Muir**, *"Volatility-Managed Portfolios"*
  (Journal of Finance, 2017). Scaling a factor's exposure by the **inverse of its recent
  realized variance** — leaning in when volatility is low, out when it is high — *raises*
  the Sharpe ratio and produces significant positive alpha against the buy-and-hold factor,
  for the market and for several equity factors. The mechanism is that expected returns do
  **not** rise one-for-one with volatility at high frequency (a vol-return "disconnect"), so
  cutting exposure into vol spikes sheds risk you were not being paid to bear.
- **The self-contained version here.** We take the single-asset **vol thermostat** on SPY:
  hold `w = min(2.0, 12% / RV_21d)` of SPY and the rest in **bills (BIL)**, so the book
  targets a roughly constant ~12% annualised volatility. We measure everything **excess of
  cash on both legs** (SPY − BIL vs managed − BIL) and ask the two honest questions: (a) does
  the **Sharpe** rise net of turnover, and (b) does it cut **tail drawdowns** — and we
  **decompose** the managed mean into an exposure (average-leverage) term and a timing (alpha)
  term to see whether any gain is "just" leverage-timing.
- **The vol-targeting caveat.** Constant scaling leaves the Sharpe ratio unchanged, so any
  Sharpe advantage of a vol-managed book is *by construction* the timing component; and a
  low-vol target that mostly de-risks can look like it "improves risk-adjusted return" while
  really just shifting weight into cash. Measuring excess-of-cash on both legs, and running a
  shuffled-vol placebo that holds the weight distribution fixed, guards against both traps
  (the concern documented in study 590, sharpe-hacking).

## What we measure, and the honesty rails

- **Trailing realized vol, no free model.** Rolling 21-day sample std of daily SPY returns
  × √252 — a pure past-only estimate.
- **Point-in-time, one documented lag.** The weight for day *t* uses vol **known at the close
  of *t−1*** (`.shift(1)`); zero look-ahead.
- **Excess-of-cash on both legs.** BIL is the **real** cash leg (a total-return T-bill ETF),
  not an assumed 0%; the managed book earns the bill rate on its un-invested fraction and pays
  a borrow spread on the levered fraction. Sharpe races are excess-vs-excess, so the cash yield
  cancels and the comparison is a clean risk-adjusted one.
- **Robust inference.** A Newey-West (HAC, Bartlett) alpha regression of managed-excess on
  B&H-excess (the Moreira-Muir "did the Sharpe rise?" test); a HAC *t* on the daily
  excess-return difference; a **paired block-bootstrap CI** on the Sharpe advantage; a
  **200-seed shuffled-vol placebo** that breaks the timing alignment while keeping the weight
  distribution; a 3×3 parameter grid; a two-era cut; a seeded synthetic positive control.
- **Short history is named on the Signal axis.** BIL begins in 2007, so the excess-of-cash
  sample is a single ~19-year SPY tape — long enough to span the GFC, 2018, COVID and 2022,
  but a single asset over one macro cycle. The stamp says so.
- **The costed timer is graded separately.** One-way × NAV per rebalance plus borrow on the
  levered leg — the honest test of whether the thin Sharpe edge survives friction.

## Shared method citations

- **Moreira, A. & Muir, T. (2017)** — *Volatility-Managed Portfolios*, JF 72(4). The claim.
- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* used on the alpha and the return difference).
- **Politis, D. & Romano, J. (1994)** — the stationary/circular block bootstrap (the paired
  Sharpe-advantage CI).
- **Wilson, E. B. (1927)** — score interval for a binomial share.
- **Barroso, P. & Santa-Clara, P. (2015)** — *Momentum has its moments*: risk-managing a
  factor by scaling to constant volatility, the sister construction on momentum.

## Data sources

- **yfinance daily total-return closes** (`auto_adjust=True`) for **SPY** and **BIL**,
  2007-05-30 → 2026-06-30, cached under `_cache/spy_bil.parquet`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [16-storm-shy](../../16-storm-shy/) — a **dip-buyer**: it *adds* equity exposure after
  down-days (a mean-reversion timing rule). This study is the opposite posture — a **vol
  thermostat** that *cuts* exposure when realized vol is high, targeting a constant vol.
- [591-vol-managed-portfolio](../../591-vol-managed-portfolio/) — the **broad** Moreira-Muir
  study across several equity **factor** ETFs (a cross-factor 1/RV overlay). This study is the
  single-asset **SPY-only** thermostat with a **real bill cash leg** and an explicit
  leverage-timing decomposition — a narrower, cleaner test of the market-portfolio case.
- [633-btc-vol-targeting](../../633-btc-vol-targeting/) — the same overlay on **Bitcoin** (a
  66%-vol asset, 0% cash). This is the **equity** case with a real T-bill leg and a lower vol
  target.
- [590-sharpe-hacking](../../590-sharpe-hacking/) — the methodological warning that vol-target
  scaling can flatter a Sharpe; this study answers it directly with excess-of-cash races and a
  weight-distribution-preserving placebo.
- [12-paper-prophet](../../12-paper-prophet/) — the paper-vs-live gap demo; cited as the desk's
  reminder that a backtest Sharpe is not a bankable one.

None of the siblings run the **single-asset SPY constant-vol thermostat, excess of a real bill
leg, with a leverage-timing decomposition** — this study's own axis.
