# References & literature map — Study 113 (Gold-Silver-Ratio)

## The claim under test

The gold/silver ratio folk trade: track the ratio of gold price to silver price (or
equivalently the GLD/SLV ETF price ratio). When the ratio is at a historical extreme
(silver is "cheap" or "expensive" relative to gold), rotate into the cheaper metal and
out of the richer one, betting the ratio reverts to its historical mean. The claim is
sometimes sharpened to: buy silver when the ratio exceeds 80 (gold expensive relative to
silver), sell silver and buy gold when the ratio falls below 40, and hold until the ratio
normalises. The hypothesis requires that the ratio is (a) stationary or mean-reverting,
and (b) that deviations at historical extremes have a reliable directional drift.

## Why the steelman is fragile — the real question

- **Pairs trading requires cointegration.** The theoretical basis for mean-reversion
  between two assets is cointegration (Engle & Granger 1987): a linear combination of
  the two log-prices is stationary even though each price is individually I(1). Gatev,
  Goetzmann & Rouwenhorst (2006), *Pairs Trading: Performance of a Relative-Value
  Arbitrage Rule* (Review of Financial Studies), document positive returns from pairs
  strategies in equities, but only where pairs are genuinely cointegrated.
- **The gold/silver ratio in practice.** Both metals are industrial and monetary
  commodities with largely different demand drivers (gold: monetary/jewellery; silver:
  industrial/solar). Over short windows they can drift far apart and stay there. Our ADF
  test (p = 0.22) and Engle-Granger cointegration test (p = 0.74) on 2010–2026 data
  both fail to find evidence of the required stationarity.
- **Anchoring to round numbers.** The practitioner thresholds of 80 (sell gold, buy
  silver) and 40 (sell silver, buy gold) are not derived from any statistical model —
  they are round-number anchors. There is no a priori reason the ratio should revert to
  a level that happens to be memorable.

## Literature on commodity pairs / mean-reversion

- **Engle, R.F. & Granger, C.W.J. (1987).** *Co-integration and Error Correction:
  Representation, Estimation, and Testing.* Econometrica, 55(2), 251–276. — The
  foundational paper for cointegration-based pairs trading. Our study implements the
  Engle-Granger two-step test on GLD/SLV.
- **Gatev, E., Goetzmann, W.N. & Rouwenhorst, K.G. (2006).** *Pairs Trading: Performance
  of a Relative-Value Arbitrage Rule.* Review of Financial Studies, 19(3), 797–827. —
  Shows pairs strategies work in equities where cointegration is verified; the methodology
  is the ancestor of the z-score approach we implement.
- **Novy-Marx, R. (2009).** *Hot and Cold Markets.* Real Estate Economics. (Also his
  unpublished commodity spreads work.) — Discusses conditions under which commodity
  spread strategies work; emphasises that mean-reversion requires a structural link.
- **Daskalaki, C. & Skiadopoulos, G. (2011).** *Should Investors Include Commodities in
  Their Portfolios After All? New Evidence.* Journal of Banking & Finance, 35(10). —
  Examines commodities as a portfolio ingredient; documents that precious metals often
  move together but the *ratio* is not always a stable reversion target.

## Statistical methods

- **ADF (Augmented Dickey-Fuller) test.** Said & Dickey (1984), *Testing for Unit
  Roots in Autoregressive-Moving Average Models of Unknown Order* (Biometrika) — tests
  whether the ratio series has a unit root. A failure to reject (p > 0.05) means the
  ratio looks like a random walk, undermining the reversion premise.
- **Ornstein-Uhlenbeck half-life.** Avellaneda & Lee (2010), *Statistical Arbitrage in
  the U.S. Equities Market* (Quantitative Finance) — the AR(1) regression approach to
  estimating mean-reversion half-life from time-series data. Our estimate of 301 trading
  days (~15 months) is extremely slow.
- **Engle-Granger two-step cointegration.** Engle & Granger (1987), as above.
  [`strategy.engle_granger_pvalue`](../gold_silver_ratio/strategy.py) wraps
  `statsmodels.tsa.stattools.coint`.
- **Newey-West HAC t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.summarize`](../gold_silver_ratio/strategy.py).

## Related desk studies

- **[Study 05 — Pairs-Trading family]**: direct ancestor of the z-score spread approach.
- **[Study 23 — Pairs family]**: further pairs-trading results on equities; the same
  cointegration caveat applies.
- **[Study 85 — Dr-Copper](../../85-dr-copper/)**: another cross-asset ratio predictor —
  the copper/gold ratio as an economic signal — tested with the same structure.
- **[Study 68 — All-Weather](../../68-all-weather/)**: commodity allocation/timing, the
  broader context for whether any cross-commodity signal beats buy-and-hold.
- **[Study 16 — Storm-Shy](../../16-storm-shy/)**: precious metals as a defensive tilt —
  the buy-and-hold framing of gold/silver without the ratio overlay.

## Data sources

- **GLD (SPDR Gold Shares ETF)** and **SLV (iShares Silver Trust ETF)** daily OHLCV
  via `yfinance`, adjusted closes, 2010-01-04 to 2026-06-12. GLD holds approximately
  1/10 troy oz of gold; SLV holds approximately 0.91 troy oz of silver. The GLD/SLV
  price ratio is proportional to the spot gold/silver ratio (scaled by ~10×) and inherits
  the same mean-reversion structure. Yahoo Finance adjusted prices account for splits and
  dividends; each headline run carries a content fingerprint (see `docs/results.md`).
