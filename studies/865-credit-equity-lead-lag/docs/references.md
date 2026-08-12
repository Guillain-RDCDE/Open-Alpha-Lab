# References & literature map — Study 865 (Credit → Equity Lead-Lag)

## The claim under test

- **The idea.** "Credit leads equity." High-yield credit is a *risk asset* whose price
  embeds the market's appetite for risk, and desk lore holds that it **turns before** the
  stock market at risk-on/off inflections — credit spreads widen (HY sells off) a beat ahead
  of an equity draw-down, and tighten ahead of a rally. Because raw HY total return is
  contaminated by interest-rate duration, the practitioner form measures the credit move
  **duration-hedged** — HYG **in excess of** IEF (7-10y Treasuries). The self-contained,
  testable version of "leads" is a **Granger-style lead-lag**: the trailing 1-4-week
  HY-excess return should carry information about the **next week's** equity return.
- **The specific test here.** We take the weekly form: compute the trailing 1-, 2- and
  4-week **HYG-minus-IEF** total return known at each Friday close, and (a) regress the
  **next** week's SPY return on it (`r_SPY[t+1] = a + b·trend_k[t] + u`), reporting the
  slope, a Newey-West *t*, and R² — the direct Granger-style lead — with a companion
  risk-on/off discrimination, a label-shift placebo and a two-era cut; and (b) test whether a
  **costed** SPY↔IEF timing overlay (long SPY next week when the credit trend is up, else
  IEF) beats a 100%-SPY buy-and-hold (net Sharpe / CAGR / max-drawdown, a Newey-West *t* on
  the active return). A seeded synthetic positive control proves the machinery is unbiased.
- **The source literature.** The lineage is the credit-leads-equity stylised fact, the
  lead-lag / cross-predictability literature, and the standard test for one series leading
  another:
  - **Granger, C. W. J. (1969)**, *"Investigating Causal Relations by Econometric Models and
    Cross-spectral Methods"* (Econometrica) — the canonical framework for "does past X help
    predict future Y beyond Y's own past"; our one-step predictive regression is its
    single-lag form.
  - **Gilchrist, S. & Zakrajšek, E. (2012)**, *"Credit Spreads and Business Cycle
    Fluctuations"* (American Economic Review) — the excess-bond-premium evidence that the
    credit market carries information that *leads* real activity and equity risk, the macro
    basis for using credit as a *leading* risk gauge.
  - **Kwan, S. H. (1996)**, *"Firm-specific information and the correlation between individual
    stocks and bonds"* (Journal of Financial Economics) — early evidence on the
    lead-lag/information linkage between a firm's bonds and its stock (lagged bond returns
    carrying predictive content), the micro root of the desk claim.
  - **Lo, A. W. & MacKinlay, A. C. (1990)**, *"When Are Contrarian Profits Due to Stock
    Market Overreaction?"* (Review of Financial Studies) — the lead-lag cross-predictability
    template and the caution that measured leads are fragile and easily an artefact of
    autocorrelation, not a tradable causal signal.

## What we measure, and the honesty rails

- **Duration-hedged trend, no free model.** The signal is the plain trailing-window total
  return of HYG minus that of IEF — an ETF-implementable, total-return (`auto_adjust`)
  quantity, not a modelled OAS or excess-bond premium.
- **Point-in-time, one documented lag.** The credit trend is **known at the close of week
  `t`** and aligned against the **`t+1`** SPY return (`.shift(-1)` on the target); the
  overlay position is held over week `t+1`. Zero look-ahead.
- **Robust inference.** A slow-moving weekly trend produces persistent risk-on/off runs, so
  a naïve OLS or group *t* would overstate significance. We take the Newey-West (HAC,
  Bartlett, 6-lag) *t* on the regression slope (via the score series `(x−x̄)·û`) and, for the
  discrimination, on a **time-ordered regime-contrast series** whose mean is exactly
  `mean(SPY | on) − mean(SPY | off)`, so the persistence correctly widens the standard error.
  A 1,000-draw **label-shift placebo** confirms the difference is (in)distinguishable from a
  random relabelling.
- **Survivorship / regime honesty on the Signal axis.** The four ETFs are live and
  continuously listed — the *tape* has no delisting bias — but the study rests on **one** US
  credit history (2007-2026); a weekly lead-lag fit on a single realised path is exposed to
  single-sample overfit. The two-era split is the check; it shows a consistently wrong-signed,
  never-significant relation.
- **The overlay is graded separately.** Costs are one-way × NAV on each switch leg; the
  strategy is long-only (SPY or IEF), so no borrow — the honest test of whether the switch
  earns its keep against buy-and-hold.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  (HAC) covariance; the *t* used on the regression slope, the regime-contrast series and the
  active-return series.
- **Wilson, E. B. (1927)** — score interval for a binomial share (risk-on-fraction
  uncertainty).

## Data sources

- **yfinance daily total-return closes** (`auto_adjust=True`) for HYG, IEF, LQD, SPY,
  2007-05-01 → 2026-06-30, cached under `_cache/` as a single parquet, resampled to
  Friday-anchored weekly closes offline.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [115-credit-spreads](../../115-credit-spreads/) — reads the credit-spread **LEVEL**
  (HYG-IEF / HYG-LQD relative to its rolling median) as a *stress warning gauge*. This study
  uses the credit **trend as a forward predictor**: a Granger-style regression of next-week
  SPY on the trailing HY-excess return — a *lead*, not a level.
- [832-high-yield-credit-momentum](../../832-high-yield-credit-momentum/) — grades the HY
  credit trend as its **own daily SPY↔IEF timer on the trend *sign*** (a same-day
  discrimination of the equity excess). This study's headline is a **weekly predictive
  regression** — the explicit *lead-lag* object (`r_SPY[t+1] ~ trend[t]`, slope + NW *t* +
  R²), a forecast, not a sign-switch discrimination.
- [131-utilities-canary](../../131-utilities-canary/) — **utilities** relative strength (a
  defensive-equity sector) as the risk-on/off canary. This study's canary is the **credit
  market** (HY vs Treasuries), a different asset class.
- [379-etf-lead-lag](../../379-etf-lead-lag/) — generic cross-ETF **return** lead-lag among
  a broad ETF set. This study is the **specific credit→equity** lead with the practitioner's
  duration-hedged HY-excess predictor and a costed timing overlay, not a generic ETF grid.

None of the siblings runs the **weekly Granger-style predictive regression of next-week SPY
on the trailing duration-hedged high-yield credit return** — this study's own axis.
