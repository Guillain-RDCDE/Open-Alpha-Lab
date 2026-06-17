# References & literature map — Study 245 (Oil-Equity-Correlation)

## The claim under test

- **"Crude oil prices lead the stock market."** The idea that oil price changes
  forecast equity returns is a persistent belief in macro investing. The "growth channel"
  story holds that rising oil signals accelerating demand (bullish for equities); the
  "cost-push" story says rising oil squeezes corporate margins (bearish for equities). In
  either variant, oil is supposed to *lead* stocks — giving investors a timing tool.
  We take the strongest testable version: a lagged regression of weekly USO/CL=F log-returns
  on *forward* SPY returns, tested in-sample (HAC t-stats) and out-of-sample (Goyal-Welch
  expanding-window OOS R²), over ~20 years of daily data (2006–2026).

## Why the steelman is coherent — the real economics behind the claim

- **Oil as a real-economy proxy.** Hamilton (2009), *Causes and Consequences of the
  Oil Price Shock of 2007–08* (Brookings Papers), documents that oil price spikes often
  precede recessions. Kilian & Park (2009), *The Impact of Oil Price Shocks on the U.S.
  Stock Market* (International Economic Review), decompose oil shocks into supply and
  demand components: demand-driven oil price increases are actually *positive* for equities,
  while supply-driven spikes are negative. This heterogeneity is one reason a simple
  univariate oil-equity regression struggles.
- **The co-movement channel.** Jones & Kaul (1996), *Oil and the Stock Markets* (Journal
  of Finance), find that the reaction of U.S. and Canadian stock markets to oil shocks can
  be explained rationally by the impact of oil on cash flows. Their result is consistent
  with a contemporaneous co-movement but does not imply a *predictive* lead.
- **The growth-cycle link.** Filis, Degiannakis & Floros (2011), *Dynamic Correlation
  between Stock Market and Oil Prices: The Case of Oil-Importing and Oil-Exporting
  Countries* (International Review of Financial Analysis), confirm that oil and equity
  markets are contemporaneously correlated, especially during global demand shocks. The
  correlation structure is time-varying and regime-dependent.

## Why the predictive claim fails — known evidence

- **Contemporaneous vs predictive: the Goyal-Welch critique.** Goyal & Welch (2008),
  *A Comprehensive Look at the Empirical Performance of Equity Premium Prediction*
  (Review of Financial Studies) — the canonical reference for showing that macro
  variables that look predictive in-sample regularly fail out-of-sample. Our OOS R²
  of −1.8% weekly confirms the Goyal-Welch finding: the oil model loses to the
  historical mean out-of-sample.
- **Oil does not Granger-cause equities at weekly horizons.** The predictive beta of
  −0.009 (HAC t = −0.46) is not only statistically insignificant — it is wrong-signed
  relative to the contemporaneous link (+0.146). By the time you observe the weekly oil
  move, the equity market has already priced in the information.
- **Structural breaks in the oil-equity relationship.** Apergis & Miller (2009),
  *Do Structural Oil-Market Shocks Affect Stock Prices?* (Energy Economics), show that
  the stock-price response to oil shocks has changed substantially over time, undermining
  the stability of any predictive regression. The COVID crash (2020) and the 2022 energy
  shock further complicate any simple linear relationship.
- **ESG and decarbonisation.** Post-2015, the oil-equity correlation has been
  structurally complicated by the rotation into clean-energy sectors; the S&P 500's
  reduced energy-sector weight (from ~13% in 2008 to ~4% by 2022) weakens the
  mechanical transmission from oil prices to aggregate equity returns.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.regression_is`](../oil_equity_correlation/strategy.py) uses inline Bartlett-kernel
  HAC to correct for weekly return autocorrelation.
- **OOS R² (Goyal-Welch).** Goyal & Welch (2008), *A Comprehensive Look at the Empirical
  Performance of Equity Premium Prediction* (Review of Financial Studies) — the expanding-
  window OOS R² in [`strategy.oos_r2`](../oil_equity_correlation/strategy.py) follows their
  exact construction.
- **Diebold-Mariano test.** Diebold & Mariano (1995), *Comparing Predictive Accuracy*
  (JBES) — the DM t-stat on forecast-error differences provides formal inference on the
  OOS comparison.
- **Reproducibility stamp.** [`quantlab/repro.py`](../../../quantlab/repro.py) — the as-of
  freeze and content fingerprint each headline run carries.

## Data sources used here

- **Yahoo! Finance daily closes** (via `yfinance`): USO (United States Oil Fund ETF),
  CL=F (WTI crude oil front-month futures), SPY (S&P 500 ETF). Daily history from
  2006-04-10 (USO inception) to 2026-06-16 (~20 years, 1,052 weekly periods). Every
  headline is pinned with an as-of date and a content fingerprint (see
  [`docs/results.md`](results.md)). The test-suite runs entirely on the deterministic
  [`data.synthetic_daily`](../oil_equity_correlation/data.py) generator, never the network.

## Related desk studies

- **[Study 85 — Dr-Copper](../../85-dr-copper/)**: the copper/gold ratio as a macro
  predictor — the closest sibling study, same Goyal-Welch methodology, same contemporaneous-
  vs-predictive decomposition. Same conclusion: coincident but not predictive.
- **[Study 103 — Turtle](../../103-turtle/)**: a systematic trend-following rule applied to
  commodities — a related approach to extracting equity-timing signals from commodity prices.
- **[Study 67 — Fed-Drift](../../67-fed-drift/)**: macro event windows — a cleaner way to
  test whether a macro indicator genuinely leads prices.
