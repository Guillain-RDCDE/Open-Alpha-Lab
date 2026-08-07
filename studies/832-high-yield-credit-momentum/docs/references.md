# References & literature map — Study 832 (High-Yield Credit Momentum)

## The claim under test

- **The idea.** High-yield credit is a *risk asset* whose price embeds the market's risk
  appetite; its **trend** (recent total return) is widely used on trading desks as a
  risk-on/off regime read. Because raw HY total return is contaminated by interest-rate
  duration, the practitioner form measures it **duration-hedged** — HYG **in excess of**
  IEF (7-10y Treasuries) — so that the signal is the *credit* move, not the *rate* move.
  A positive credit trend (HY out-earning duration-matched Treasuries) is read as risk
  appetite rising → be long equities; a negative trend → de-risk into Treasuries.
- **The source literature.** The lineage is time-series momentum applied to credit and the
  credit-leads-equity stylised fact:
  - **Moskowitz, T., Ooi, Y. H. & Pedersen, L. H. (2012)**, *"Time Series Momentum"*
    (Journal of Financial Economics) — trailing 3-12-month total-return trend predicts an
    asset's own next-period return across asset classes; the template for a
    trend-as-timing rule.
  - **Haesen, D., Houweling, P. & van Zundert, J. (2017)**, *"Momentum spillover from
    stocks to corporate bonds"* (Journal of Banking & Finance) — and the broader
    corporate-bond momentum literature (Jostova, Nikolova, Philipov & Stahel, 2013,
    *"Momentum in Corporate Bond Returns"*, RFS) establishing that credit carries its own
    trend.
  - **Gilchrist, S. & Zakrajšek, E. (2012)**, *"Credit Spreads and Business Cycle
    Fluctuations"* (American Economic Review) — the excess-bond-premium evidence that the
    credit market leads real activity and equity risk, the macro basis for using credit as
    a *leading* risk gauge.
- **The specific test here.** We take the self-contained daily version: compute the
  trailing 3-/6-month **HYG-minus-IEF** trend, form the binary risk-on/off switch known at
  the close of `t−1`, and test (a) whether the trend **discriminates** the day-`t`
  equity-excess return `r_SPY − r_IEF` (a Newey-West *t* on the risk-on−risk-off
  difference, a placebo, a two-era cut) and (b) whether the **costed** SPY↔IEF timer beats
  a 100%-SPY buy-and-hold (net Sharpe / CAGR / max-drawdown, a Newey-West *t* on the active
  return). A seeded synthetic positive control proves the machinery is unbiased.

## What we measure, and the honesty rails

- **Duration-hedged trend, no free model.** The signal is the plain trailing-window total
  return of HYG minus that of IEF — an ETF-implementable, total-return (`auto_adjust`)
  quantity, not a modelled OAS.
- **Point-in-time, one documented lag.** The trend is **known at the close of `t−1`**
  (`.shift(1)`); the position is held on day `t`. Zero look-ahead.
- **Robust inference.** A slow-moving conditioning signal produces long risk-on/off runs,
  so a naïve group *t* would overstate significance; we take the Newey-West (HAC, Bartlett,
  10-lag) *t* on a **time-ordered regime-contrast series** whose mean is exactly
  `mean(xs | on) − mean(xs | off)`, so the persistence correctly widens the standard error.
  A 1,000-draw **label-shift placebo** confirms the difference is (in)distinguishable from
  a random relabelling.
- **Survivorship / regime honesty on the Signal axis.** The four ETFs are live and
  continuously listed — the *tape* has no delisting bias — but the study rests on **one**
  US credit history (2007-2026); a binary regime timer fit on a single realised path is
  exposed to single-sample overfit. The two-era split is the check, and it fails (the sign
  flips), which is the honest finding.
- **The timer is graded separately.** Costs are one-way × NAV on each switch leg; the
  strategy is long-only (SPY or IEF), so no borrow — the honest test of whether the switch
  earns its keep against buy-and-hold.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  (HAC) covariance; the *t* used on the regime-contrast and active-return series.
- **Wilson, E. B. (1927)** — score interval for a binomial share (win-rate uncertainty).

## Data sources

- **yfinance daily total-return closes** (`auto_adjust=True`) for HYG, IEF, LQD, SPY,
  2007-05-01 → 2026-06-30, cached under `_cache/` as a single parquet.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [115-credit-spreads](../../115-credit-spreads/) — reads the credit-spread **LEVEL**
  (HYG-IEF / HYG-LQD *below its rolling median* = a stress warning gauge). This study uses
  the credit **TREND** (the trailing *total return* of HYG-excess-over-IEF) as a *timing
  switch* into or out of equities — a direction/momentum read, not a level warning.
- [795-corporate-bond-momentum](../../795-corporate-bond-momentum/) — **cross-sectional**
  momentum *within* the corporate-bond universe (rank bonds/credit sleeves against each
  other). This study is a **single time-series** trend of one duration-hedged credit spread
  used to time a *different* asset (equities).
- [131-utilities-canary](../../131-utilities-canary/) — **utilities** relative strength
  (a defensive-equity sector) as the risk-on/off canary. This study's canary is the
  **credit market** (HY vs Treasuries), a different asset class and a different signal.

None of the siblings uses the **trailing total-return trend of duration-hedged high-yield
credit as an equity risk-on/off timing switch** — this study's own axis.
