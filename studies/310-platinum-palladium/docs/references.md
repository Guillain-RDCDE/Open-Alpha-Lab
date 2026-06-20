# References & literature map — Study 310 (Platinum-Palladium)

## The claim under test

The platinum/palladium "autocatalyst pair" folk trade: track the ratio of platinum price
to palladium price (or the PL=F / PA=F futures ratio). When the ratio is at a historical
extreme — palladium "cheap" or "expensive" relative to platinum — rotate into the cheaper
metal and short the richer one, betting the ratio reverts to its long-run level. The
steelman is structural: both metals are platinum-group metals (PGMs) used as the active
ingredient in automotive catalytic converters, and the two are partially *substitutable*
in catalyst chemistry. So the relative price "should" be anchored by the cross-elasticity
of substitution — when palladium gets too rich, automakers re-engineer toward platinum,
pulling the ratio back. The hypothesis requires that the ratio is (a) stationary or
mean-reverting, and (b) that deviations at extremes carry a reliable directional drift.

## Why the steelman is fragile — the real question

- **Substitution is slow and one-directional in practice.** Petrol (gasoline) engines use
  palladium-rich catalysts; diesel engines use platinum-rich ones. The 2015 "Dieselgate"
  scandal and tightening petrol-emission standards drove a structural demand shift *toward*
  palladium that lasted years. Substituting platinum back into petrol catalysts requires
  re-certification and is gradual — so the "restoring force" the trade relies on operates
  on a multi-year, not multi-month, timescale (see Johnson Matthey PGM Market Reports).
- **The regime inverted.** For most of the 2000s–2010s platinum traded *above* palladium
  (ratio > 1.5). From 2018 to 2022 palladium overtook platinum and the ratio fell below
  1.0 — at the extreme, palladium cost more than three times platinum. A reversion trade
  anchored to the historical mean was run over by a structural regime change. Our full-
  sample ratio range (0.31–3.70) is the fingerprint of that non-stationarity-in-practice.
- **Pairs trading requires cointegration.** The theoretical basis for tradable reversion
  between two assets is cointegration (Engle & Granger 1987): a linear combination of the
  two log-prices is stationary. Our Engle-Granger test (p = 0.70) fails to find it. A
  full-sample ADF on the ratio *does* reject a unit root (p = 0.011), but that is the
  signature of a series that wandered far and partly came back over 16 years — with a
  475-day half-life it is not a tradable restoring force.

## Literature on commodity pairs / mean-reversion

- **Engle, R.F. & Granger, C.W.J. (1987).** *Co-integration and Error Correction:
  Representation, Estimation, and Testing.* Econometrica, 55(2), 251–276. — The
  foundational paper for cointegration-based pairs trading; we implement the two-step test
  on log(PL) vs log(PA).
- **Gatev, E., Goetzmann, W.N. & Rouwenhorst, K.G. (2006).** *Pairs Trading: Performance
  of a Relative-Value Arbitrage Rule.* Review of Financial Studies, 19(3), 797–827. —
  Shows pairs strategies earn where cointegration is verified; the ancestor of the z-score
  approach we implement.
- **Avellaneda, M. & Lee, J.-H. (2010).** *Statistical Arbitrage in the U.S. Equities
  Market.* Quantitative Finance, 10(7), 761–782. — The OU / AR(1) half-life framework for
  mean-reversion speed; our 475-day estimate uses this method.
- **Daskalaki, C. & Skiadopoulos, G. (2011).** *Should Investors Include Commodities in
  Their Portfolios After All? New Evidence.* Journal of Banking & Finance, 35(10),
  2606–2626. — Documents that precious metals often move together but the *ratio* is not a
  stable reversion target.
- **Johnson Matthey, *PGM Market Report* (annual/biannual).** The industry reference on
  platinum-group-metal supply, demand, and automotive substitution — the source of the
  regime-change narrative that breaks the naive ratio trade.

## Statistical methods

- **ADF (Augmented Dickey-Fuller) test.** Said & Dickey (1984), *Testing for Unit Roots in
  Autoregressive-Moving Average Models of Unknown Order* (Biometrika). H0: unit root;
  rejection (p < 0.05) is evidence of stationarity. Wrapped in
  [`strategy.adf_pvalue`](../platinum_palladium/strategy.py).
- **Engle-Granger two-step cointegration.** Engle & Granger (1987), as above.
  [`strategy.engle_granger_pvalue`](../platinum_palladium/strategy.py) wraps
  `statsmodels.tsa.stattools.coint`.
- **Ornstein-Uhlenbeck half-life.** AR(1) OLS on Δy vs lagged y; half-life
  = −log(2)/log(1+β). [`strategy.half_life`](../platinum_palladium/strategy.py).
- **Newey-West HAC t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica).
  [`strategy.summarize`](../platinum_palladium/strategy.py).
- **Circular block bootstrap.** Politis & Romano (1992), *A circular block-resampling
  procedure for stationary data.* — block CI on the mean per-trade return, preserving
  short-run dependence. [`strategy.block_bootstrap_ci`](../platinum_palladium/strategy.py).

## Related desk studies

- **[Study 113 — Gold-Silver-Ratio](../../113-gold-silver-ratio/)**: the closest sibling —
  the same z-score ratio-reversion machinery on the gold/silver pair. PL/PA differs in that
  its ratio is dominated by a one-off automotive substitution regime change.
- **[Study 305 — Gold-Oil-Ratio](../../305-gold-oil-ratio/)** and
  **[Study 306 — Crack-Spread](../../306-crack-spread/)**: other cross-commodity ratio /
  spread signals in the Carry-curves-commodities family.
- **[Study 85 — Dr-Copper]**: a macro-grounded cross-asset ratio (copper/gold).

## Data sources

- **PL=F (platinum continuous front-month futures)** and **PA=F (palladium continuous
  front-month futures)** daily OHLCV via `yfinance`, auto-adjusted closes,
  2010-01-04 to 2026-05-29. The continuous front-month series carries roll effects, but
  the PL/PA *ratio* largely cancels common roll/level artefacts. Each headline run carries
  a content fingerprint (see `docs/results.md`). The reproducible core and the test-suite
  run entirely on a deterministic synthetic OU pair — the network is touched only on an
  explicit cache miss.
