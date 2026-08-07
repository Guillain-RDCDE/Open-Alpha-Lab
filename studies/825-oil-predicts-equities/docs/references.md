# References & literature map — Study 825 (Oil Predicts Equities)

## The claim under test

- **The source paper.** Gerben **Driesprong, Ben Jacobsen & Benjamin Maat**, *"Striking
  Oil: Another Puzzle?"* (Journal of Financial Economics, 2008). Across the US and a broad
  panel of developed and emerging markets they find that **oil price changes predict stock
  market returns with a one-month lag, negatively**: a rising oil price this month is
  followed by *lower* equity returns next month. They read it as **under-reaction** — the
  market is slow to price a macro shock that raises input costs and signals demand/inflation
  pressure — and note the effect survives standard risk controls and is absent from the
  contemporaneous relation alone.
- **The behavioural / macro reading.** Oil is both an economy-wide **cost** input and a
  **demand** barometer; if investors under-react to oil news, the information diffuses into
  equity prices over the following weeks, producing a lagged negative predictive slope. The
  same slow-diffusion logic underlies gradual-information-diffusion models
  (Hong-Stein 1999).
- **The specific test here.** We take the self-contained monthly version: a single-regressor
  predictive regression of the S&P 500 (SPY) **forward one-month** return on the **trailing
  one-month** oil (USO) return, with a Newey-West HAC *t* on the slope, its sign, its R², a
  Welch tercile cross-check, a 2,000-permutation placebo, a two-era robustness cut, a costed
  monthly timer benchmarked against buy-and-hold, and a seeded synthetic positive control.

## What we measure, and the honesty rails

- **A forecast, not a correlation.** The predictor is the oil return over month `t`; the
  target is the equity return over month `t+1` (`equity.shift(-1)`). One documented
  execution lag, **zero look-ahead** — the whole point is the *gap* between the two returns.
- **Robust inference.** A Newey-West (HAC, Bartlett, 6-lag) *t* on the OLS slope, computed
  from a closed-form sandwich variance of the score `(x−x̄)·resid` — an overlapping macro
  regression is heteroskedastic and serially correlated, so a plain OLS *t* would overstate
  significance. A permutation placebo (shuffle the target, keep the predictor) confirms the
  slope is not a lucky alignment.
- **Sign matters.** The claim fixes the sign a priori (β < 0). A significant *wrong-sign*
  slope would still be a failure to replicate the claim; here the slope is not even
  significant, and its point estimate has the wrong (positive) sign.
- **Survivorship named on the Signal axis.** USO and SPY are continuously-listed ETFs (no
  delisting bias). The construction caveat is that **USO** is a front-month roll vehicle with
  documented contango drag — an ETF proxy for the oil price, not spot or a constant-maturity
  index.
- **The timer is graded separately.** One-way cost × NAV per rebalance leg plus borrow on
  shorts, and — crucially — benchmarked against simply holding SPY, so a book that merely
  harvests the equity premium cannot masquerade as an oil edge.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* used on the regression slope).
- **Hong, H. & Stein, J. (1999)** — gradual-information-diffusion / under-reaction, the
  mechanism Driesprong et al invoke for the lagged oil→equity link.
- **Goyal, A. & Welch, I. (2008)** — the out-of-sample benchmark discipline for predictive
  regressions (a mean-return baseline any predictor must beat); the sibling study 245 runs
  the explicit OOS-R² version.
- **Wilson, E. B. (1927)** — score interval for a binomial share (the hit-rate primitive).

## Data sources

- **yfinance daily adjusted close** (`auto_adjust=True`, total-return), USO + SPY,
  2006-04-10 → 2026-06-30, cached under this study's own `_cache/`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [245-oil-equity-correlation](../../245-oil-equity-correlation/) — the **contemporaneous**
  same-period oil↔equity co-movement (`corr(r_oil[t], r_equity[t])`), the visually
  compelling but untradable chart, plus a multi-horizon regression. This study is strictly
  the **lagged predictive** slope (oil month `t` → equity month `t+1`), the Driesprong
  forecast, at monthly frequency — a *forecast*, not a co-movement.
- [226-crude-seasonality](../../226-crude-seasonality/) — crude's **calendar** seasonality
  (month-of-year effects **in the oil price itself**), not oil predicting a *different*
  asset. Here oil is a cross-asset predictor of equities.
- [85-dr-copper](../../85-dr-copper/) — **copper** as a pro-cyclical growth barometer for
  equities (a *positive* reading, "Dr. Copper"). This study uses **oil**, and the claimed
  sign is *negative* (a cost-push / under-reaction channel), a different commodity and the
  opposite predicted sign.

None of the siblings run the **lagged monthly oil → forward-month equity** predictive
regression — the Driesprong-Jacobsen-Maat signal — which is this study's own axis.
