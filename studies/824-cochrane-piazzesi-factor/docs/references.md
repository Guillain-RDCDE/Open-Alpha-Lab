# References & literature map — Study 824 (Cochrane-Piazzesi Factor)

## The claim under test

- **The source paper.** John H. **Cochrane & Monika Piazzesi**, *"Bond Risk Premia"*
  (American Economic Review, 95(1), 2005). Regressing each Treasury bond's one-year-ahead
  **excess return** on the full vector of **forward rates**, they find the fitted values
  line up on a **single** predictive factor — one tent-shaped linear combination
  `CP_t = gamma' f_t` (negative on the short forward, rising to a peak in the middle of
  the curve, falling back at the long end) forecasts excess returns of *every* maturity,
  with an R² (~0.35 in their sample) well above the single-slope regressions of Fama-Bliff
  and Campbell-Shiller. The factor is a "return-forecasting factor" that is nearly
  invisible to the level/slope/curvature that price the cross-section of *yields*.
- **The antecedents.** **Fama, E. & Bliss, R. (1987)**, *"The Information in Long-Maturity
  Forward Rates"* — a single forward-spot spread forecasts that bond's excess return.
  **Campbell, J. & Shiller, R. (1991)**, *"Yield Spreads and Interest Rate Movements"* —
  the expectations-hypothesis-rejecting slope regressions. Cochrane-Piazzesi's advance is
  using *all* forwards jointly and collapsing them to one factor.
- **The critique this study lands on.** **Bauer, M. & Hamilton, J. (2018)**, *"Robust Bond
  Risk Premia"* (Review of Financial Studies) — the CP-style predictive regressions use
  **highly persistent** regressors (near-unit-root yields) with **overlapping** annual
  returns, which severely inflates conventional and even Newey-West test statistics and
  R²; once you account for it, much of the apparent predictability is a **spurious-
  regression** artifact and does not survive out of sample. **Thornton, D. & Valente, G.
  (2012)**, *"Out-of-Sample Predictions of Bond Excess Returns and Forward Rates"* — the
  forward-rate factors have **no economic value out of sample** for a real-time investor.
- **The specific test here.** We rebuild the factor from the coarse constant-maturity grid
  a no-key retail stack exposes (`^IRX` 0.25y, `^FVX` 5y, `^TNX` 10y, `^TYX` 30y → four
  implied forwards) and forecast the average 252-day excess return of the `SHY/IEF/TLT`
  bond ETFs, then subject it to exactly the honesty rails the critique demands: a HAC *t*
  with lags scaled to the overlap, a Campbell-Thompson **out-of-sample R²**, and a **block-
  rotation placebo** that shows how much R² pure persistence manufactures.

## What we measure, and the honesty rails

- **Forwards, no free model.** Each constant-maturity yield is a continuously-compounded
  zero (`p(n) = −n·y(n)`); the implied forward between adjacent nodes is
  `f(n1→n2) = (n2·y2 − n1·y1)/(n2 − n1)`. Same-day information only — public at close `t`.
- **The predictive regression.** OLS of the average one-year-ahead ETF excess return on the
  forward vector; the fitted value is the CP factor. In-sample R² is the headline the claim
  advertises.
- **HAC is mandatory but not sufficient.** The 252-day return windows overlap ~252-fold, so
  a plain *t* is meaningless; we use Newey-West with ≈ 1.5× the horizon in lags. **But** —
  and this is the study's spine — even the HAC *t* is size-distorted under near-unit-root
  regressors: our synthetic **null** fires that statistic on 11/20 seeds. So we grade on the
  **out-of-sample R²** (expanding window, Campbell-Thompson 2008) and a **block placebo**,
  which are the honest arbiters.
- **The timer is graded separately.** A costed duration-timing overlay (own TLT when the
  out-of-sample forecast is rich) asks whether any of the in-sample fit is a paycheck.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* on the predictive slope and loadings).
- **Campbell, J. & Thompson, S. (2008)** — out-of-sample R² vs the prevailing mean, the
  bar a predictor must clear to be economically real.
- **Wilson, E. B. (1927)** — score interval for a binomial share.

## Data sources

- **yfinance daily closes** (`auto_adjust=True`): the four constant-maturity yield indices
  (annualised %) and the three bond ETFs (total-return), 2002-01-02 → 2026-06-30, cached
  under `_cache/`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [581-term-premium](../../581-term-premium/) — a **single** ACM-style term-premium proxy
  (`y10 − EWMA(short)`) used to *time TLT*. This study uses the **whole forward-rate vector
  collapsed to one predictive factor** (the CP object), forecasting the *cross-maturity
  average* excess return, not a one-number premium timing a single instrument.
- [132-yield-curve-steepener](../../132-yield-curve-steepener/) — trades the raw **10y−3m
  slope** (a level-and-expectations mix). The CP factor strips the level/slope/curvature and
  isolates the return-forecasting combination the slope alone misses.
- [66-inverted](../../66-inverted/) — the **inversion sign** of the curve as a regime/recession
  signal, a one-bit contrast. This study is a continuous multi-forward regression of bond
  *excess returns*, not an inversion switch.
- [380-curve-roll-down](../../380-curve-roll-down/) — the mechanical **roll-down / carry** of
  sitting on a static curve. The CP factor is a *risk-premium forecaster* (time-varying
  expected excess return), a different object from deterministic roll.

None of the siblings build **the joint forward-rate return-forecasting factor** of
Cochrane-Piazzesi — the single tent that is this study's own axis.
