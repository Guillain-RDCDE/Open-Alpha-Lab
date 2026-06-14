# References & literature map — Study 132 (Yield-Curve-Steepener)

## The claim under test

The yield-curve slope (long yield minus short yield) is the oldest bond-market timing
heuristic. The bond-manager version: *"When the Treasury curve is steep — the 10-year
yield well above the 3-month rate — long bonds are fairly rewarded for duration risk;
when the curve is flat or inverted, term premia have collapsed and long bonds are
expensive. A steepener timing rule — long TLT when slope > 0, cash otherwise — should
outperform buy-and-hold on a risk-adjusted basis."* We test this at the daily resolution
against unconditional buy-and-hold TLT over the full 2002-2026 history.

## Core academic literature on the term structure and bond return predictability

- **Fama & Bliss (1987).** *The Information in Long-Maturity Forward Rates.* American
  Economic Review. The paper that established forward-rate predictability of excess
  bond returns at annual horizons — the foundational result the daily claim leans on.
  Crucially, predictability is documented *monthly and annually*, not daily, and using
  forward rates rather than the raw slope.

- **Campbell & Shiller (1991).** *Yield Spreads and Interest Rate Movements: A Bird's
  Eye View.* Review of Economic Studies. Documents that the yield spread has modest
  predictive power for future rate changes and bond returns at long horizons. Again a
  *long-horizon* result; at the daily level the noise far exceeds the signal.

- **Cochrane & Piazzesi (2005).** *Bond Risk Premia.* American Economic Review. A linear
  combination of five forward rates (the "C-P factor") predicts annual excess bond
  returns with R² ~35% — far stronger than the simple slope. Our study uses the simpler
  slope (TNX−IRX); the C-P factor might yield a stronger but more complex signal.

- **Kim & Wright (2005).** *An Arbitrage-Free Three-Factor Term Structure Model and the
  Recent Behavior of Long-Term Yields and Distant-Horizon Forward Rates.* Fed working
  paper. The Kim-Wright model decomposes the 10Y yield into expected short rates and a
  term premium. Using the model-implied term premium directly (rather than the raw slope)
  might improve signal-to-noise, but the data is available only from the Fed's website.

- **Adrian, Crump & Moench (2013).** *Pricing the Term Structure with Linear Regressions.*
  Journal of Financial Economics. The ACM term-premium model (NY Fed) is the workhorse
  decomposition; its term-premium component directly measures the compensation for
  duration risk rather than the mixture of expectations and premia embedded in the raw slope.

## Why the claim is plausible but breaks down at daily resolution

- **Term premia do vary with the slope.** At monthly and quarterly horizons the yield
  spread predicts excess bond returns with economically meaningful R² (Fama-Bliss: ~15%
  annually). But the signal-to-noise ratio collapses at daily frequency: TLT daily
  volatility (~70 bps/day) dwarfs the daily return differential attributable to the slope.

- **Mean-reversion kills daily timing.** The daily marginal information in "slope > 0
  vs slope ≤ 0" is nearly zero: the slope changes slowly (AR(1) parameter ~0.97) and
  the daily TLT return distribution is essentially the same in both regimes. The timing
  rule captures very little incremental information day to day.

- **Inversion regime is rare and heterogeneous.** In 2002-2026, only 13.5% of days were
  in the inverted regime. Inversions are driven by different mechanisms (Fed tightening
  in 2006-07 vs. 2022-23; growth scares in 2019) and TLT performance varies accordingly.
  The 2022-23 episode saw large TLT losses even after re-steepening; 2019 saw TLT rally.

## Related desk studies

- **[Study 66 — Inverted](../../66-inverted/)**: the inverted yield curve predicting
  *equity* returns (the recession-signal version). The same structural relationship,
  tested on SPY rather than TLT, with similarly weak daily-frequency results.

- **[Study 85 — Dr-Copper](../../85-dr-copper/)**: another macro-momentum signal
  (commodity prices as growth proxies) predicting equity returns — the same family of
  macro-timing claims tested at daily frequency.

- **[Study 56 — Tide-Table](../../56-tide-table/)**: the CAPE / Shiller P/E as an
  equity-return predictor — a valuation signal in the same spirit as the term-premium
  argument here. Also weak at short horizons, stronger at multi-year horizons.

- **[Study 111 — VIX-Term-Structure](../../111-vix-term-structure/)**: the VIX
  term structure (contango/backwardation) predicting SPY returns — structurally the most
  similar study to this one, with a comparable weak/fragile verdict.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy._hac_tstat`](../yield_curve_steepener/strategy.py) and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).

- **Sharpe with robust SE / annualisation.** Lo (2002), *The Statistics of Sharpe
  Ratios* (Financial Analysts Journal) — [`quantlab.analytics.sharpe_with_se`](../../../quantlab/analytics.py).

- **Block bootstrap CI.** Politis & Romano (1994), *The Stationary Bootstrap* (JASA) —
  [`quantlab.stats.sharpe_ci_bootstrap`](../../../quantlab/stats.py).

- **Quintile-rank signal (rolling, out-of-sample).** The percentile-rank approach to
  signal construction avoids assuming a linear relationship and is robust to
  distributional changes in the slope over time (regime changes in the Fed's reaction
  function, structural breaks in the 2022 era).

## Data sources used here

- **Yahoo Finance daily closes** (via `yfinance`): TLT (iShares 20+ Year Treasury Bond
  ETF, since July 2002), IEF (iShares 7-10 Year Treasury Bond ETF), SHY (iShares 1-3
  Year Treasury Bond ETF), ^TNX (CBOE 10-Year Treasury Note Yield Index), ^IRX (CBOE
  13-Week Treasury Bill Rate). Combined frame: 2002-07-31 → 2026-06-12 (6,000 daily
  observations). The offline core and tests run on a deterministic `data.synthetic_daily`
  generator, never the network.
