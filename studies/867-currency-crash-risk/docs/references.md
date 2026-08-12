# References & literature map — Study 867 (Currency Crash Risk)

## The claim under test

- **The source paper.** Markus K. **Brunnermeier, Stefan Nagel & Lasse H. Pedersen**,
  *"Carry Trades and Currency Crashes"* (NBER Macroeconomics Annual, 2008). They document
  that high-interest-rate (carry) currencies are exposed to **crash risk**: their exchange
  rates against the funding currency are **negatively skewed** — long stretches of gentle
  appreciation (earning the interest differential) punctuated by sudden, violent
  depreciations when carry positions unwind. The memorable phrase: carry "goes **up by the
  stairs and down by the elevator**." The negative skewness increases with the interest
  differential and with speculator crowding, and it partly rationalises the forward-premium
  puzzle (UIP failure): the carry premium is compensation for a sold-crash exposure, not a
  free lunch.
- **The mechanism.** Funding-liquidity spirals: when funding tightens, leveraged carry
  traders unwind together, so the high-yield currency falls sharply precisely when risk
  appetite collapses — a peso-problem / sold-insurance tail rather than a diffusion risk.
- **The specific test here.** We build a fixed carry proxy (a per-currency long-run average
  short-rate differential vs USD), rank 8 currencies (G10 + the notorious high-carry MXN),
  and test the two halves of the claim: (a) the **skew-carry cross-section** — does a higher
  carry predict a more negative realized skew? — via an OLS slope and a Spearman rank
  correlation with a permutation *p*; and (b) the **basket crash skew** — is the long-high /
  short-low carry basket negatively skewed? — with a Newey-West *t* on the standardised-cubed
  residual series, a label-shuffle placebo, a crash-conditional split, a two-era cut, a costed
  timer, and a seeded synthetic positive control.

## What we measure, and the honesty rails

- **Realized skewness, no free model.** For each currency (and the basket) the sample
  skewness of weekly simple returns (population third standardised moment). The basket's
  skew gets a **Newey-West (HAC, Bartlett, 6-lag)** *t* via `g_t = ((x−mean)/sd)³`
  (`mean(g)=skew`) — a deliberately conservative moment-estimator *t* that treats the
  standardising mean/sd as known; it is documented as low-powered against rare crashes.
- **The carry is a proxy, named as such.** True overnight-deposit / forward-discount
  differentials are not on yfinance; the carry is a transparent fixed per-currency constant.
  The verdict rests on the realized skewness measured off spot and on the carry *ordering*,
  not on the exact carry magnitudes.
- **Point-in-time, one documented lag.** The carry ranking is known at the close of week
  `t` and the basket is held over `t+1`. The static ranking makes turnover ≈ 0, but the
  convention is stated. Zero look-ahead.
- **Survivorship is named on the Signal axis.** The basket is a **current** fixed membership
  (G10 + MXN) with no defunct or de-pegged legs, so the magnitudes are an **upper bound**.
- **The timer is graded separately.** Costs are 2 bps/side rebalance on the gross book plus
  borrow on the short leg — the honest test of whether the (weak) premium survives friction.

## Shared method citations

- **Newey, W. & West, K. (1987)** — heteroskedasticity- and autocorrelation-consistent
  covariance (the HAC *t* used on the basket skew and premium series).
- **Wilson, E. B. (1927)** — score interval for a binomial share.
- **Menkhoff, Sarno, Schmeling & Schrimpf (2012)** — *"Carry Trades and Global Foreign
  Exchange Volatility"* — the volatility-risk reading of the same carry premium, a companion
  to the crash-risk story tested here.

## Data sources

- **yfinance daily FX spot** (`auto_adjust=True`), 8 currencies vs USD (EUR, GBP, JPY, AUD,
  CAD, CHF, NZD, MXN), resampled to weekly (Friday), 2003-12-12 → 2026-06-26, cached under
  this study's own `_cache/fx_weekly.parquet`.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- [364-fx-carry-trade](../../364-fx-carry-trade/) — tests whether the carry basket earns a
  **premium** (does UIP fail, is the mean positive). This study tests the **crash skew** —
  the *third moment* — that is said to *justify* that premium, not the mean.
- [828-fx-dollar-factor](../../828-fx-dollar-factor/) — the **dollar factor** DOL (the common
  equal-weight average currency move vs USD) and its timing. This study sorts the cross-section
  **high-minus-low carry** and measures skewness, not the common dollar level.
- [27-steamroller](../../27-steamroller/) — the generic "picking up pennies in front of a
  steamroller" short-vol / sold-insurance shape. This study is the **specific FX carry-crash
  instance** of that archetype, measured as realized return skewness.
- [797-fx-value-ppp](../../797-fx-value-ppp/) — the **PPP value** currency anomaly (cheap vs
  expensive on purchasing-power parity), a different currency signal that does not touch carry
  or the crash tail.

None of the siblings sort currencies on **carry** and test the **realized return skewness /
crash tail** of the resulting cross-section — the Brunnermeier-Nagel-Pedersen signal — which
is this study's own axis.
