# References & literature map — Study 909 (Preferred Reset Premium)

## The claim under test

- **The reset story.** Traditional preferred stock pays a **fixed** perpetual coupon, so it
  carries long duration — when rates rose sharply in 2022 the fixed-rate preferred complex fell
  double digits, much like a long bond. **Fixed-to-floating** and **variable-rate** preferreds
  instead reset their coupon off a short-rate benchmark (historically 3-month LIBOR, now
  SOFR/Term-SOFR, or a fixed spread over a Treasury on the reset date), so their duration is
  short and their income *rises* with the front end. The folk claim: in a high-rate / rising-rate
  regime, a variable-rate preferred sleeve delivers a **better rate-adjusted carry** than a plain
  fixed-rate preferred sleeve, and holds up through the drawdown that clubs the fixed complex.
- **The 2022 exhibit.** The natural experiment is the 2022 hiking cycle. Invesco's **VRP**
  (Variable Rate Preferred) and Global X's **PFFV** are the liquid variable-rate vehicles; iShares
  **PFF**, Invesco **PGX** and **PGF** are the fixed-rate flagships. In 2022 the fixed sleeve lost
  ~19–20% while the variable sleeve lost ~11% — the exhibit this study puts a *t*-stat on.
- **The specific test here.** We measure the **(variable − fixed)** monthly total-return spread on
  the live ETFs (cash cancels in the difference, so it nets out the shared credit beta), race the
  two sleeves excess-of-cash (minus BIL), cut the sample at the 2022 regime break, bootstrap the
  spread and the Sharpe advantage, and cost both a market-neutral isolation spread and a
  rising-rate regime-switch tilt. The honest question is whether the reset premium is a *standalone
  anomaly* or a *regime-contingent* bet — and whether either version survives costs.

## What we measure, and the honesty rails

- **Excess-of-cash everywhere.** Every Sharpe is computed on the return minus BIL (1-3 month
  T-bill), so the variable-vs-fixed race compares two excess-of-cash streams on the same footing —
  a variable-rate instrument mechanically earns more cash yield when rates rise, and we do not want
  to credit that to the reset story by accident.
- **The spread nets the credit beta.** Preferred stock is junior, equity-like credit; both sleeves
  load heavily on the same credit/equity factor. The (variable − fixed) difference cancels most of
  that common beta and isolates the duration/reset difference — the quantity the thesis is actually
  about.
- **Total return, not price.** `auto_adjust=True` folds the (large) preferred dividends back in —
  mandatory for income instruments whose return is mostly the coupon.
- **Robust inference.** Newey-West (HAC, Bartlett, 6-lag) *t* on the monthly spread — a
  coupon-reset series is serially correlated, so a plain *t* would overstate. A one-sample and a
  Welch *t* cross-check, and a circular block bootstrap (block 6) gives a CI on both the spread
  mean and the Sharpe advantage.
- **Short history is named on the Signal axis.** VRP dates to 2014 and PFFV to 2020; the whole
  premium lives in the single 2022+ hiking cycle. The era cut makes that explicit — the effect is
  Mixed (regime-contingent), not era-robust.
- **The tilt is graded separately, with one lag and real costs.** The regime-switch uses the
  rising-rate signal known at month-end `t−1` to pick the sleeve held over month `t`; the isolation
  spread pays 2 sides × one-way × NAV per rebalance plus borrow on the short fixed leg.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent covariance
  (the HAC *t* used on the monthly spread series).
- **Wilson, E. B. (1927)** — score interval for a binomial share (hit-rate CIs).
- **Politis, D. & Romano, J. (1992)** — the circular block bootstrap used for the spread and
  Sharpe-advantage confidence intervals under serial dependence.

## Data sources

- **yfinance daily closes** (`auto_adjust=True`, total-return) for VRP, PFFV, PFF, PGX, PGF and BIL,
  2006-12-01 → 2026-06-30, cached under `_cache/pref_prices.parquet`.
- Fund mandates and reset mechanics from the issuer fact sheets: **Invesco VRP / PGX / PGF**,
  **iShares PFF**, **Global X PFFV** (variable-rate preferreds reset off a short-rate benchmark;
  fixed-rate preferreds pay a fixed perpetual coupon).
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [338-preferred-stocks](../../338-preferred-stocks/) — tests the preferred asset class **as a
  whole** (the *level*: preferreds vs equities / bonds, the equity-like-credit tradeoff). This study
  works **inside** the preferred sleeve, on the **variable-minus-fixed** spread, not the level.
- [339-convertible-bonds](../../339-convertible-bonds/) — convertibles (equity optionality embedded
  in a bond), a different hybrid; no coupon-reset mechanism.
- [340-bank-loans](../../340-bank-loans/) — senior **floating-rate loans** (BKLN): the same
  "floats-with-rates" story but in the loan complex, tested as a rate-vs-credit identity. This study
  is the **preferred** analogue, and specifically the *variable-vs-fixed within preferreds* contrast.
- [885-ultra-short-credit-pickup](../../885-ultra-short-credit-pickup/) — the front-end
  credit-spread pickup in ultra-short bond funds; a different part of the curve and no reset/duration
  contrast.

None of the siblings measure the **variable-rate-minus-fixed-rate preferred** spread across the rate
regime — this study's own axis.
