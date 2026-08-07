# References & literature map — Study 828 (FX Dollar Factor)

## The claim under test

- **The source paper.** Hanno **Lustig, Nikolai Roussanov & Adrien Verdelhan**, *"Common
  Risk Factors in Currency Markets"* (Review of Financial Studies, 2011). Sorting currencies
  into portfolios on their forward discount (interest differential), they extract two factors:
  **HML_FX** (the high-minus-low carry slope) and **DOL** — the **dollar factor**, the
  equal-weight average excess return of *all* currency portfolios against the USD. DOL is the
  level (common) factor; every currency loads on it near one. Its *unconditional* mean is small,
  but they show the **average forward discount** (the cross-sectional mean interest gap vs the
  USD) forecasts the dollar excess return — DOL is priced **conditionally** ("dollar-timing").
- **The follow-ups.** Lustig, Roussanov & Verdelhan (2014, *"Countercyclical Currency Risk
  Premia"*) formalise the dollar-carry / average-forward-discount timing strategy; Verdelhan
  (2018, *"The Share of Systematic Variation in Bilateral Exchange Rates"*) documents how much of
  bilateral FX variance the dollar and carry factors span.
- **The specific test here.** We build DOL as the equal-weight mean of 7 foreign-currency spot
  returns vs the USD (each pair normalised to USD-per-foreign, inverting the USD-base quotes), test
  its premium with a Newey-West *t*, and run a **dollar-timing predictive regression** of next-month
  DOL on a trailing-dollar-trend conditioning variable — a spot-only **proxy** for the average
  forward discount, since true forward discounts (interest differentials) are not on yfinance.

## What we measure, and the honesty rails

- **DOL, no free model.** The equal-weight average of the foreign-currency monthly spot returns —
  a USD-funded long basket; a positive value means the dollar depreciated. An optional `excess` leg
  adds the average forward-discount **carry proxy** (a fixed, transparent per-currency annualised
  rate differential vs USD, named as a proxy everywhere).
- **The timing conditioning variable is a proxy — flagged.** The LRV average forward discount is
  the cross-sectional mean interest gap; with a static carry proxy that level is constant, so the
  *time-varying* timing signal here is the trailing-12-month DOL (dollar trend). It is a spot-only
  stand-in, and the honest read is that the rate-based LRV signal cannot be reconstructed from spot.
- **Point-in-time, one documented lag.** The timing signal is known at the close of month `t`; the
  position is held over month `t+1` (`.shift`). Zero look-ahead.
- **Robust inference.** Newey-West (HAC, Bartlett, 6-lag) *t* on the DOL mean and on the timing
  regression slope — monthly currency returns are mildly serially correlated. A one-sample *t*
  cross-checks; a **1,000-rotation block-shuffle placebo** breaks the signal → outcome link on the
  timing slope.
- **Survivorship is named on the Signal axis.** The basket is a fixed **current** G10 membership —
  no delisted or hard-pegged legs — so the magnitudes are an **upper bound**.
- **The timer is graded separately.** Costs are one-way × turnover on the static basket and one-way
  per switch on the timed overlay — the honest test of whether a (here absent) premium survives
  friction.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent covariance
  (the HAC *t* used on the DOL mean and the timing slope).
- **Wilson, E. B. (1927)** — score interval for a binomial share (the shared inference set).
- **Fama, E. (1984), "Forward and Spot Exchange Rates"** — the forward-discount / UIP-failure
  bedrock the carry and dollar factors are built on.

## Data sources

- **yfinance daily FX spot** (`auto_adjust=True`), 7 G10 currencies vs USD, sampled to month-end,
  2003-12-31 → 2026-06-30, cached under `_cache/fx_eom.parquet`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [364-fx-carry-trade](../../364-fx-carry-trade/) — the **HML_FX / carry** slope (long high-yield,
  short low-yield currencies), the *cross-sectional* LRV factor. This study tests the **DOL** level
  factor (the basket vs USD) and its dollar-timing, not the high-minus-low carry spread.
- [797-fx-value-ppp](../../797-fx-value-ppp/) — **PPP value** (real-exchange-rate mean reversion),
  a valuation signal from CPI. DOL is a return-average factor, not a value sort.
- [147-fx-momentum](../../147-fx-momentum/) — **cross-sectional FX momentum** (trailing return
  ranking across currencies). DOL is the equal-weight *level* of the whole basket, not a
  winners-minus-losers spread.
- [36-greenback](../../36-greenback/) — a broad **US-dollar-strength** study. This study frames the
  dollar as the LRV *risk factor* DOL and tests its priced premium and forward-discount timing.

None of the siblings tests the **equal-weight dollar factor DOL and whether the average forward
discount times it** — the specific Lustig-Roussanov-Verdelhan claim, which is this study's own axis.
