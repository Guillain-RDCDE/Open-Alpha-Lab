# References & literature map — Study 204 (Talmud-Portfolio)

## The claim under test

- **The original text.** Babylonian Talmud, *Bava Metzia* 42a (3rd–6th century CE):
  *"A man should always divide his money into three parts: a third in land, a third in
  business, and a third in reserve."* Attributed to Rabbi Isaac Bar Aha. The "reserve"
  is conventionally interpreted as liquid wealth (gold or cash); rendered in modern ETFs
  this is 1/3 VNQ (real estate), 1/3 SPY (equities), 1/3 BND (bonds/cash).
- **Modern rendering.** The Talmud portfolio has attracted attention in the personal-finance
  and quantitative-finance communities as a "2,000-year-old diversification rule" that
  anticipates the three-bucket (growth / income / reserve) framework. We test whether it
  stands up against the standard 60/40 benchmark on Sharpe ratio, CAGR, and max drawdown.

## Why the steelman is plausible — the real effect it leans on

- **Diversification across asset classes.** Markowitz (1952), *Portfolio Selection*
  (Journal of Finance) — the mean-variance framework formalises what the Talmud
  anticipates intuitively: mixing assets with imperfect correlations reduces portfolio
  variance without a proportional cut in expected return. The Talmud's three legs span
  "real" assets (land/REITs), "business" risk (equities), and "reserve" (cash/bonds).
- **Equal-weight as a robust allocator.** DeMiguel, Garlappi & Uppal (2009), *Optimal
  versus Naive Diversification* (Review of Financial Studies) — 1/N equal-weight beats
  sample mean-variance out of sample because estimation error swamps the optimiser's
  theoretical advantage. The Talmud's 1/3–1/3–1/3 is exactly this rule applied to three
  broad asset classes. See also Study 171 (Naive-1-Over-N) for the direct test on sectors.
- **Real estate as a diversifier.** Case & Shiller (1989), *The Efficiency of the Market
  for Single-Family Homes* (AER), and Hoesli & MacGregor (2000), *Property Investment*,
  document the low correlation of real estate with equities and bonds over long horizons.
  However, publicly-traded REITs (VNQ) show much higher equity correlation than physical
  property — a critical nuance that drives this study's result.

## Why the REIT leg fails as a hedge

- **REITs and equity correlation.** Clayton & MacKinnon (2003), *The Relative Importance
  of Stock, Bond and Real Estate Factors in Explaining REIT Returns* (Journal of Real
  Estate Finance and Economics), show that REITs became increasingly correlated with
  broad equities through the 1990s–2000s. During crises, REIT liquidity risk makes them
  behave like leveraged equities, not diversifiers. Our data confirms this: VNQ fell
  -69% in the GFC (vs -55% for SPY) and -42% in the COVID crash (vs -34% for SPY).
- **The leverage problem in REITs.** Pagliari, Scherer & Monopoli (2005), *Public versus
  Private Real Estate Equities* (Real Estate Economics) — REITs use leverage and share
  equity market liquidity, making them poor proxies for "land" in the Talmud's sense.
  Physical property — the original intent — would behave very differently.
- **2022 as the decisive counter-example.** In the 2022 rate-shock environment bonds
  fell -14.5% (ending their traditional hedging role at low rates) while REITs fell
  -31.9% — the Talmud blend was left with all three legs losing simultaneously. The
  "reserve" (BND) that the Talmud designated for safety failed exactly when needed.

## Comparison to related allocation frameworks

- **Permanent Portfolio.** Harry Browne (1981), *Inflation-Proofing Your Investments*,
  and see Study 144 (Permanent-Portfolio) on this desk — the 25/25/25/25 four-regime
  portfolio (stocks/bonds/gold/cash). The gold leg provides genuine inflation and
  crisis hedging that the REIT leg in the Talmud portfolio does not; PP has a better
  Sharpe history than the Talmud blend.
- **Risk parity.** Bridgewater's All Weather portfolio and Qian (2005), *Risk Parity
  Portfolios* (PanAgora white paper) — weighting assets by inverse volatility rather
  than equal nominal amounts. Risk parity would assign far less weight to the volatile
  VNQ and SPY than the Talmud does and far more to BND. See Study 68 (All-Weather).
- **The 60/40 benchmark.** Blanchett (2014), *The True Cost of Active Management*
  (Morningstar), confirms that simple 60/40 portfolios are hard to beat out of sample.
  Our 19-year test shows the Talmud trails 60/40 by -1.2 pp/yr (HAC t = -1.26), not
  statistically significant but consistently negative.

## Method lineage (the desk's shared engine)

- **Annual-return HAC t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica)
  applied to the annual return differential — the same inference convention as Study 144.
- **Max drawdown and worst-year metrics.** Used throughout the desk's allocation studies
  as the primary risk measure for defensive-allocation claims; see METHODOLOGY.md.
- **Total-return prices.** `yfinance auto_adjust=True` folds in dividends and splits,
  producing a total-return series for each ETF — essential for bonds (BND's coupon
  income is a material fraction of its return) and REITs (VNQ pays large dividends).

## Data sources used here

- **Yahoo Finance daily prices** (via `yfinance`), adjusted close (total return) for VNQ,
  SPY, and BND. Joint window: 2007-04-10 (BND inception) to 2026-06-11. The shared
  cross-asset panel (`_cache/cross_asset_etfs.parquet`) supplies VNQ and SPY; BND is
  fetched into the per-study cache. Both caches are git-ignored; the offline
  reproducible core and tests run on the deterministic `data.synthetic_three_asset`
  generator, never the network.

## Related desk studies

- **[Study 144 — Permanent-Portfolio](../../144-permanent-portfolio/)**: the 25/25/25/25
  four-regime allocation with gold; better Sharpe than the Talmud blend and more honest
  about the gold-as-crisis-hedge claim.
- **[Study 171 — Naive-1-Over-N](../../171-naive-1-over-n/)**: tests whether 1/N
  equal-weight beats Markowitz optimisers across sector ETFs — the same intuition
  as the Talmud's 1/3 rule but on a richer universe.
- **[Study 68 — All-Weather](../../68-all-weather/)**: the risk-parity version of
  diversified allocation, the natural competitor to both 60/40 and the Talmud blend.
- **[Study 97 — Balancing-Act](../../97-balancing-act/)**: annual rebalancing vs
  buy-and-hold across asset classes — the direct test of the rebalancing mechanism
  this study also relies on.
