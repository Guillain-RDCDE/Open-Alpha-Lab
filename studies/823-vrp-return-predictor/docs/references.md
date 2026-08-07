# References & literature map — Study 823 (Variance-Risk-Premium Return Predictor)

## The claim under test

- **The source paper.** Tim **Bollerslev, George Tauchen & Hao Zhou**, *"Expected Stock
  Returns and Variance Risk Premia"* (Review of Financial Studies, 2009). They define the
  variance risk premium as the difference between the option-implied (risk-neutral)
  variance and the expected realized (physical) variance, and show it is a **significant
  time-series predictor of aggregate market excess returns**, with the predictive R²
  **hump-shaped in horizon and peaking at the quarterly frequency** — much stronger than
  the classic price-dividend or P/E predictors at that horizon. Positive VRP → higher
  forward return.
- **The economic reading.** The VRP is compensation for **variance (tail) risk**: when
  investors are especially risk-averse they pay up for variance/option protection (a fat
  VRP), and in equilibrium the market subsequently earns a higher premium. It links the
  equity premium to the price of volatility risk (Bollerslev-Tauchen-Zhou embed it in a
  consumption model with time-varying economic uncertainty / volatility-of-volatility).
- **The specific test here.** We build the self-contained monthly version a retail stack
  can reach: implied variance from the **VIX** (`(VIX/100)²/12`), realized variance from
  the **trailing 21 daily squared SPY log returns**, `VRP = IV − RV`, and regress the
  **forward 1- and 3-month SPY return** on `VRP_t` with a **Newey-West** *t* on the slope,
  a block-bootstrap placebo, a two-era robustness cut, a costed timer, and a seeded
  synthetic positive control. The VIX-based monthly measure is a coarser estimator than the
  paper's model-implied expected realized variance, so this is a conservative reading.

## What we measure, and the honesty rails

- **Implied variance, no free model.** `IV_t = (VIX_t/100)² / 12` — the VIX is a 30-day
  risk-neutral volatility index; squaring and de-annualising gives a monthly variance.
- **Realized variance, no free model.** `RV_t` = the trailing-21-trading-day sum of daily
  squared log returns — a standard non-parametric monthly realized variance.
- **Point-in-time, one documented lag.** `VRP_t` is known at the close of month-end `t`
  (both the VIX print and the trailing-RV window are causal); the forward return runs
  strictly over `t → t+h`. Zero look-ahead.
- **Robust inference.** A Newey-West (HAC, Bartlett) *t* on the **slope** of the predictive
  regression — overlapping multi-month forward returns are strongly serially correlated, so
  a homoskedastic OLS *t* overstates significance (here +2.06 vs a HAC +0.75 at the
  3-month horizon). We stamp on the HAC.
- **Risk-free proxy named on the Signal axis.** BTZ regress *excess* returns; we set the
  bill leg to zero — a near-constant monthly bill shifts the regression intercept, not the
  slope, so the *t*-stat is unaffected.
- **The timer is graded separately.** A VRP-conditioned long/flat market-timer, costed
  one-way × NAV per switch — the honest test of whether the predictor pays after frictions.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance; the HAC *t* used on the regression slope and the timer spread.
- **Newey, W. & West, K. (1994)** — automatic HAC lag-length selection (the data-driven
  bandwidth in `predictive_regression`).
- **Wilson, E. B. (1927)** — score interval for a binomial share.
- **Carr, P. & Wu, L. (2009)** — *"Variance Risk Premiums"*; the model-free synthetic-
  variance-swap construction of the VRP, complementary to the VIX-based measure here.
- **Drechsler, I. & Yaron, A. (2011)** — *"What's Vol Got to Do with It"*; a long-run-risk
  model rationalising the VRP's return predictability.

## Data sources

- **yfinance daily closes** — SPY (`auto_adjust=True`, total-return) + ^VIX (index level),
  1993-01-29 → 2026-06-30, cached under this study's own `_cache/`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [130-vol-risk-premium](../../130-vol-risk-premium/) — tests whether the VRP **exists** /
  is positive on average (its *level*, harvested by shorting variance / selling options).
  This study takes the VRP's existence as given (it is, +10 vol-points here) and asks the
  different question: does its *time variation* **predict the market's forward return**?
- [111-vix-term-structure](../../111-vix-term-structure/) — the **shape** of the VIX
  futures curve (contango/backwardation as a roll-yield signal), not the implied-minus-
  realized variance gap.
- [3-fear-gauge](../../3-fear-gauge/) — the VIX **level** as a contrarian "buy the fear"
  dip signal, not the variance *risk premium* (implied minus realized) as a predictor.

None of the siblings run a **time-series predictive regression of the market's forward
excess return on `IV − RV`** — the Bollerslev-Tauchen-Zhou signal — which is this study's
own axis.
