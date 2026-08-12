# References & literature map — Study 875 (Idiosyncratic-Vol Change)

## The claim under test

- **The level puzzle it springs from.** Andrew **Ang, Robert Hodrick, Yuhang Xing &
  Xiaoyan Zhang**, *"The Cross-Section of Volatility and Expected Returns"* (Journal of
  Finance, 2006). Sorting stocks on the **level** of their idiosyncratic (residual,
  market/Fama-French-model) volatility, they find the *high*-idio-vol names go on to earn
  **lower** returns — the "idiosyncratic volatility puzzle" (study 501 on this desk).
- **The change, not the level.** This study asks a distinct question: does the **change**
  in idiosyncratic vol carry information the level does not? The behavioural reading is
  that a **rising** idio-vol signals a *deteriorating information environment* / rising
  disagreement — the residual noise around a name is growing — which, if that noise is
  over-priced or foreshadows bad news, should precede **lower** returns; a **falling**
  idio-vol (tightening consensus) re-rates. So a long **falling-idio-vol** / short
  **rising-idio-vol** book should earn a positive spread. This mirrors the
  disagreement / information-quality literature (e.g. **Diether, Malloy & Scherbina
  2002** on analyst-forecast dispersion, and the **information-uncertainty** return
  literature) applied to a *change* in residual vol.
- **The specific test here.** For each name we estimate the market-model residual vol
  over a **recent** 21-day window and a **prior** (non-overlapping) 21-day window; the
  **delta-IVOL** is the recent minus the prior residual vol. We sort a liquid US
  cross-section on the delta, hold the equal-weight long-falling / short-rising book, and
  read a Newey-West *t*, a permutation placebo, a two-era cut, an **additivity regression
  against the idio-vol level sort (501)**, a costed timer, and a seeded synthetic
  positive control.

## What we measure, and the honesty rails

- **Residual vol, no free model.** The market factor is the equal-weight cross-sectional
  mean return; each name's idiosyncratic vol is the standard deviation of its CAPM
  residual `r - a - b*mkt`, computed **vectorised** via the identity
  `resid_var = var(r) - cov(r, mkt)**2 / var(mkt)` on a rolling window (no per-date
  regression). The **delta-IVOL** is `ivol(recent) - ivol(prior)` over two
  non-overlapping windows.
- **Point-in-time sort, one documented lag.** The ranking signal is the delta-IVOL
  **known at the close of `t-1`** (`.shift(1)`); the book is held on day `t`. Zero
  look-ahead.
- **Robust inference.** Newey-West (HAC, Bartlett, 10-lag) *t* on the daily long-short
  spread — an overlapping-formation signal is serially correlated, so a plain *t* would
  overstate significance. A one-sample *t* and a pooled Welch *t* (falling book vs rising
  book) cross-check. A **1,000-permutation placebo** breaks the signal → forward-return
  link. An **additivity regression** asks whether the *change* adds anything on top of
  the idio-vol *level* (501).
- **Survivorship is named on the Signal axis.** The universe is a **current-membership**
  set of ~50 liquid mega-caps, pulled through the `quantlab.universe` survivorship guard
  (`allow_survivorship_bias=True`, an explicit opt-in). Delisted / de-rated names are
  absent, so the cross-sectional magnitudes are an **upper bound**.
- **The timer is graded separately.** Costs are one-way × NAV on the long-short book, and
  the short book pays borrow — the honest test of whether a small daily spread survives
  friction.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* used on the spread series).
- **Wilson, E. B. (1927)** — score interval for a binomial share.
- **Ang, A., Hodrick, R., Xing, Y. & Zhang, X. (2006)** — the idiosyncratic-volatility
  *level* puzzle this study takes the *change* of.
- **Diether, K., Malloy, C. & Scherbina, A. (2002)** — dispersion-of-opinion / rising
  disagreement predicts lower returns (the behavioural reading of a rising idio-vol).

## Data sources

- **yfinance daily OHLC** (`auto_adjust=True`, total-return), 50 liquid US large-caps,
  2010-01-04 → 2026-06-30, cached under `_cache/` through `quantlab.universe`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [501-idiosyncratic-volatility](../../501-idiosyncratic-volatility/) — the **level** of
  residual (market-model) vol (Ang-Hodrick-Xing-Zhang). This study sorts on the **change**
  in that residual vol and regresses the change *out of* the level (the additivity test);
  on this tape corr(change spread, level spread) is only +0.216.
- [817-realized-volatility-trend](../../817-realized-volatility-trend/) — the trend in
  **total** realized vol (`vol21/vol63 - 1`). This study uses the **residual**
  (idiosyncratic, market-model) vol and a recent-vs-prior **change**, stripping out the
  common market-vol move a total-vol measure still carries.
- [330-low-volatility](../../330-low-volatility/) — the low-**total**-vol *level* anomaly,
  again a level and a total (not residual) vol.

None of the siblings sort on the **change in a name's market-model residual vol** — the
delta-IVOL — which is this study's own axis.
