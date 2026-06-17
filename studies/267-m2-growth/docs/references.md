# References & literature map — Study 267 (M2-Growth)

## The claim under test

The monetarist intuition: money-supply (M2) growth drives asset prices. Fast
money creation should lift equities (more dollars chasing the same assets), and
a money slowdown should foreshadow weak returns. The chart of M2 YoY growth
overlaid on the S&P 500 became a staple of retail commentary during the
2020–2023 money cycle (the +27% YoY COVID spike, then the first-ever negative
prints in 2023).

- **Friedman, M. & Schwartz, A. (1963).** *A Monetary History of the United
  States, 1867–1960.* Princeton University Press. The foundational monetarist
  text: money matters, and large money-supply swings have real macro
  consequences. The leap from "money matters for the macroeconomy" to "M2 growth
  predicts next-month equity returns" is the one this study scrutinizes.

- **Friedman, M. (1970).** "A Theoretical Framework for Monetary Analysis."
  *Journal of Political Economy*, 78(2), 193–238. The quantity theory of money,
  the source of the slogan that inflation is "always and everywhere a monetary
  phenomenon." Note: a relationship with *inflation* over years is not a tradable
  *forward equity-return* signal at monthly frequency.

## Why the overlay chart deceives — the methodological traps

- **Contemporaneity ≠ prediction.** M2 growth and equities both respond to the
  business cycle and to monetary policy. In recessions the Fed eases, M2 growth
  jumps, and stocks recover off the bottom — so a *contemporaneous* overlay looks
  causal. We impose a one-month execution lag (M2 is itself released weeks late)
  and test whether *lagged* M2 predicts *forward* returns. It barely does.

- **Overlapping-window t-stat inflation.** Regressing the forward 12-month return
  on M2 uses overlapping windows, which manufactures serial correlation and
  inflates the naive OLS t-stat. The honest correction is a **Newey-West (HAC)**
  standard error with a truncation lag at least as long as the window. In our
  data the 12-month OLS t is ~−2.0 ("significant!") but the HAC t is ~−0.6.

- **Nonstationarity & regime overlap.** Most of the eye-catching co-movement
  comes from two episodes (the GFC and COVID), where money and markets both moved
  violently. A handful of regimes is not 40 years of independent evidence.

- **Wrong sign for the bullish claim.** In our sample, high-money-growth months
  are followed by *lower* average next-month returns, not higher — the opposite
  of the folk claim, and itself statistically weak.

## Academic literature on money and stock returns

- **Rozeff, M. S. (1974).** "Money and Stock Prices: Market Efficiency and the
  Lag in Effect of Monetary Policy." *Journal of Financial Economics*, 1(3),
  245–302. An early, careful finding: by the time money-supply data are
  published, the market has already discounted them — consistent with our null.

- **Thorbecke, W. (1997).** "On Stock Market Returns and Monetary Policy."
  *Journal of Finance*, 52(2), 635–654. Monetary-policy shocks affect stock
  returns, but the channel is the *unexpected* policy surprise (e.g., fed-funds
  futures), not the published, lagged M2 aggregate.

- **Sullivan, R., Timmermann, A. & White, H. (1999).** "Data-Snooping, Technical
  Trading Rule Performance, and the Bootstrap." *Journal of Finance*, 54(5),
  1647–1691. The reference treatment of how mining many macro overlays inflates
  apparent predictive power.

## Method lineage

- **Newey, W. K. & West, K. D. (1987).** "A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix."
  *Econometrica*, 55(3), 703–708. The HAC estimator implemented in
  `strategy.nw_regression` (Bartlett kernel). The truncation lag is set to the
  forward-window length for overlapping returns.

- **Quantile sort.** Bucket months by lagged M2 growth; compare the mean forward
  return of the top vs bottom bucket with a Welch t-test — the model-free
  companion to the regression.

- **Tradable rule with costs.** Long ^GSPC when lagged M2 YoY exceeds its
  trailing median, else cash; one-way costs charged on NAV at every flip; net vs
  passive buy-and-hold (price-only, no borrow because the rule is long/flat).

## Data sources

- **M2 YoY growth.** Hardcoded monthly anchors in `data.py`, interpolated to a
  monthly grid. Source contour: FRED series **M2SL** (M2 money stock, seasonally
  adjusted), year-over-year % change. As-of 2026-06-16.
- **^GSPC (S&P 500 price index).** Monthly closes via yfinance, cache-only by
  default at `_cache/gspc_monthly.parquet`. **Price only** (no dividends), so the
  return series understates total return — labeled on the Signal axis. yfinance
  ^GSPC monthly history starts in 1985, giving us a 1985–2025 overlap with the
  M2 series (~492 months).

## Related desk studies

- **[Study 120 — Excess-CAPE-Yield](../../120-excess-cape-yield/)**: a macro
  valuation overlay tested with the same HAC discipline.
- **[Study 158 — Super-Bowl](../../158-super-bowl/)**: the canonical
  spurious-overlay teardown whose structure this study mirrors.
