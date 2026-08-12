# References & literature map — Study 881 (Jobless-Claims Sector Rotation)

## The claim under test

- **The nowcast folk-rule.** Rising **initial jobless claims** are a fast, weekly read on
  a cooling labour market; the sell-side "risk-off rotation" playbook says a claims
  uptick should tilt equities from **cyclicals** (consumer-discretionary, industrials —
  earnings geared to the business cycle) toward **defensives** (consumer-staples,
  utilities — stable demand). Operationalised here: the **4-week change in the 4-week-MA
  initial-claims level** should *negatively* predict the forward **cyclical-minus-
  defensive** sector-return spread.
- **Why it is plausible.** Claims lead payrolls and the NBER cycle; defensives
  historically outperform into and through recessions while cyclicals lead the recovery.
  If claims turn *before* the rotation, a labour-nowcast overlay on sector baskets could
  in principle front-run the risk-off move — a *rotation*, not a market-timer.
- **The specific test here.** Monthly frame: the FRED `IC4WSA` 4-week-MA claims level and
  the four SPDR sector ETFs (XLY, XLI vs XLP, XLU). A predictive Newey-West regression of
  the forward equal-weight cyclical-minus-defensive spread on the claims change, with a
  permutation placebo, a COVID-sensitivity / winsor / Spearman cut, a two-era split, a
  costed long-short rotation timer, and a seeded synthetic positive control.

## What we measure, and the honesty rails

- **The signal is a clean public macro series.** `IC4WSA` is the seasonally-adjusted
  4-week moving average of initial unemployment-insurance claims — the canonical,
  noise-smoothed claims gauge. Its month-on-month change is the "4-week change." The FRED
  CSV endpoint (`fred.stlouisfed.org`) is DNS-unreachable in this build, so the values
  are encoded as a **documented, never-revised public snapshot** (source cited in
  `data.py`), the same convention Study 385 and Study 268 use; `fetch()` still attempts
  the live feed with retries and a DBnomics fallback.
- **Point-in-time, one documented lag.** The month-`t` 4-week MA is fully printed by
  month-end `t`; the position is held over month `t+1` (`.shift(-1)`) — the signal
  strictly precedes the outcome. Zero look-ahead.
- **Robust inference, and the outlier is named.** Newey-West (HAC, Bartlett, 6-lag) *t*
  on the regression **slope** (a monthly overlapping signal is serially correlated, so a
  plain *t* overstates significance). Because the entire claims history is dominated by
  the **2020 spike**, the decisive check is the COVID-drop / winsor / Spearman rank
  triple — a single leverage point is exactly how a spurious macro "signal" is
  manufactured.
- **No survivorship in the outcome.** The four SPDR sector ETFs and SPY have traded
  continuously since 1998; there is no delisting to bias the spread. Survivorship is
  named on the Signal axis for completeness, but the honest hazard here is the outlier,
  not a survivor set.
- **The timer is graded separately.** One-way × NAV per leg on the 2×-NAV long-short
  rotation, plus borrow on the short — the honest test of whether a small monthly edge
  survives friction.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* on the regression slope and on return series).
- **Spearman, C. (1904)** — the rank correlation used as the outlier-robust cross-check.
- **Wilson, E. B. (1927)** — score interval for a binomial share.
- **U.S. Department of Labor, Employment & Training Administration** — the weekly
  Unemployment Insurance Claims report; FRED series `ICSA` (initial claims) and `IC4WSA`
  (its 4-week moving average).

## Data sources

- **FRED `IC4WSA`** — initial claims, SA, 4-week MA (thousands); documented public
  monthly snapshot encoded in `data.py` (FRED CSV host unreachable in this build).
- **yfinance daily OHLC** (`auto_adjust=True`, total-return) — XLY, XLI, XLP, XLU, SPY,
  1998-12-22 → 2026-06-30, cached under `_cache/`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [385-jobless-claims-momentum](../../385-jobless-claims-momentum/) — the same claims
  tape, but the outcome is the **whole market** (SPY): does a claims uptick time
  *equities*? This study's outcome is a **long-short cyclical-vs-defensive sector
  rotation**, a relative-value trade, not a market-timer.
- [268-sahm-rule](../../268-sahm-rule/) — the **unemployment-rate** recession trigger
  (Sahm), a level/threshold recession *call*, not a claims-change *rotation* between
  sectors.
- [626-unemployment-trend-timing](../../626-unemployment-trend-timing/) — trend-timing on
  the **unemployment rate** for market exposure, again a market-timing overlay on a
  monthly labour level, not a claims-driven sector spread.
- [756-challenger-layoffs](../../756-challenger-layoffs/) — the **Challenger job-cut**
  announcement series as a labour signal; a different data source and a different
  outcome, not the 4-week claims change driving cyclical/defensive baskets.

None of the siblings regress a **cyclical-minus-defensive sector spread** on the
**4-week change in initial claims** — this study's own axis.
