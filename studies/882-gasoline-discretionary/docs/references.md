# References & literature map — Study 882 (Gas-Price → Discretionary)

## The claim under test

- **The "pump tax" folklore.** A jump in the retail gasoline price acts like a regressive
  consumption tax: households spend a near-fixed number of gallons, so a higher price at the
  pump drains disposable income that would otherwise flow to discretionary purchases (autos,
  apparel, travel, restaurants) while sparing consumer *staples* (food, household goods). The
  textbook top-down trade is therefore to **rotate out of consumer-discretionary (XLY) into
  staples (XLP)** when gasoline rises, and to lean into **energy (XLE)**, whose revenues move
  *with* the pump. We test the tradable, forward-looking version: does *this month's* gas
  move forecast *next month's* XLY−XLP spread, negatively?
- **The macro basis.** Gasoline is a large, salient, high-frequency line item in the consumer
  basket, and gasoline-price shocks have a well-documented drag on real consumption
  (Edelstein & Kilian 2009, *"How sensitive are consumer expenditures to retail energy
  prices?"*, Journal of Monetary Economics). Whether that macro drag translates into a
  *forecastable* sector-rotation edge at monthly frequency — after the market has already
  seen the same gas prices — is the empirical question this study settles.
- **The specific test here.** A single-regressor predictive regression of the
  discretionary-minus-staples (XLY − XLP) **forward one-month** return on the **trailing
  one-month** gasoline (RB=F) return, with a Newey-West HAC *t* on the slope, its sign, its
  R², a Welch tercile cross-check, a 2,000-permutation placebo, a two-era robustness cut, a
  costed monthly spread timer, and a seeded synthetic positive control. A parallel regression
  grades the energy tilt (XLE − SPY).

## What we measure, and the honesty rails

- **A forecast, not a correlation.** The predictor is the gas return over month `t`; the
  target is the spread return over month `t+1` (`spread.shift(-1)`). One documented execution
  lag, **zero look-ahead** — the whole point is the *gap* between the two returns, since the
  contemporaneous co-move is uninteresting (the market already knows the gas price).
- **Robust inference.** A Newey-West (HAC, Bartlett, 6-lag) *t* on the OLS slope, from a
  closed-form sandwich variance of the score `(x−x̄)·resid` — an overlapping macro regression
  is heteroskedastic and serially correlated, so a plain OLS *t* would overstate
  significance. A permutation placebo (shuffle the target, keep the predictor) confirms the
  slope is not a lucky alignment.
- **Sign matters.** The claim fixes the sign a priori (β < 0 for XLY−XLP, β > 0 for
  XLE−SPY). A significant *wrong-sign* slope would be a failure to replicate; here the
  XLY−XLP slope has the *right* sign but is not significant, and the energy slope is a flat
  zero.
- **Survivorship named on the Signal axis.** RB=F, XLY, XLP, XLE and SPY are
  continuously-listed liquid futures/ETFs (no delisting bias). The construction caveat is
  that **RB=F** is a front-month RBOB roll used as a proxy for the retail pump price.
- **The timer is graded separately.** The XLY−XLP spread is a 2× NAV long/short book; we
  charge one-way cost × NAV per rebalance leg on both legs plus borrow on the short leg, so a
  gross whisper cannot masquerade as a net paycheck.

## Data sources

- **yfinance daily adjusted close** (`auto_adjust=True`, total-return), RB=F + XLY + XLP +
  XLE + SPY, 2005-01-03 → 2026-06-30, cached under this study's own `_cache/`.
- **FRED `GASREGW`** (U.S. Regular All Formulations Gas Price, weekly, EIA via the St. Louis
  Fed) is the canonical *retail* pump series and the intended confirmatory tape; it was
  **unreachable from the build host** (DNS to `fred.stlouisfed.org` failed at build time), so
  the traded **RB=F** RBOB futures price — which co-moves ≈0.9 with GASREGW at the monthly
  frequency — carries the signal. This is a data-access limitation of the build environment,
  documented here rather than papered over; the headline is a **real** gasoline price.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* used on the regression slope).
- **Edelstein, P. & Kilian, L. (2009)** — the drag of retail energy prices on consumer
  expenditures, the macro channel behind the pump-tax rotation.
- **Hong, H. & Stein, J. (1999)** — gradual-information-diffusion / under-reaction, the
  mechanism a *lagged* gas→sector predictive slope would require.
- **Wilson, E. B. (1927)** — score interval for a binomial share (the hit-rate primitive).

## Related desk studies (the dedup map — what this study is NOT)

- [825-oil-predicts-equities](../../825-oil-predicts-equities/) — the lagged **crude →
  aggregate-equity** forecast (Driesprong-Jacobsen-Maat). This study swaps crude for the
  **gasoline** (retail pump) price and swaps the whole-market target for a **within-equity
  sector rotation** (discretionary vs staples), a different predictor and a different, sector
  target.
- [245-oil-equity-correlation](../../245-oil-equity-correlation/) — the **contemporaneous**
  same-period oil↔equity co-movement, not a lagged forecast and not a sector spread.
- [226-crude-seasonality](../../226-crude-seasonality/) — crude's **calendar** seasonality in
  the oil price itself, not gas as a cross-asset predictor of a consumer-sector spread.
- [639-gasoline-rvp-seasonality](../../639-gasoline-rvp-seasonality/) — gasoline's **own**
  RVP-driven (summer-blend) calendar seasonality, a property of the gas price, not gas
  predicting a *different* asset. Here gasoline is a cross-asset predictor of the XLY−XLP
  rotation.

None of the siblings run the **lagged monthly gasoline → forward-month
discretionary-vs-staples** predictive regression — the pump-tax rotation — which is this
study's own axis.
