# References & literature map — Study 207 (REITs-Diversifier)

## The claims under test

This study dissects three overlapping claims made by REIT advocates:

1. **Diversification claim.** REITs have a low correlation with stocks and bonds, so a REIT
   sleeve in a 60/40 improves the portfolio Sharpe ratio and reduces drawdowns. The empirical
   test: does 60/20/20 SPY/VNQ/TLT beat 60/40 on risk-adjusted terms?
2. **Inflation-hedge claim.** REITs hold real assets; rents adjust upward with inflation; ergo
   REITs should outperform in high-CPI regimes. The empirical test: does VNQ's annual return
   rank above SPY in high-inflation years?
3. **Crisis-resilience claim.** REITs have stable rental income uncorrelated with the business
   cycle, so they hold up better than equities in crashes. The empirical test: does VNQ
   outperform SPY during SPY drawdowns deeper than −15%?

All three claims are steelmanned (the *strongest* version) and tested honestly — the results
expose a consistent pattern of equity-beta amplification rather than diversification.

## The academic backdrop — why REITs *should* diversify (the bull case)

- **Real estate as an asset class.** Ibbotson, R.G. & Siegel, L.B. (1984), *Real Estate
  Returns: A Comparison with Other Investments*, AREUEA Journal — the foundational paper
  documenting real estate's low correlation with financial assets in private (appraisal-based)
  data. The catch: publicly traded REITs are far more correlated with equities than private
  real estate because they trade on exchanges and absorb market-wide sentiment shocks in real
  time.
- **REIT diversification claims.** Glascock, J.L. & Davidson, W.N. (1995), *Further Examination
  of the Relationship Between REIT Returns and Inflation*, Journal of Real Estate Finance and
  Economics — documents that the REIT-equity correlation in earlier decades (pre-1990s) was
  genuinely low, lending early credence to the diversification story.
- **The securitisation shift.** Hoesli, M. & Serrano, C. (2010), *Are Securitized Real Estate
  Returns More Predictable Than Stock Returns?*, Journal of Real Estate Finance and Economics
  — argues that post-1990 REIT returns have increasingly aligned with small-cap equity returns
  as the sector matured and became institutionally dominated.

## Why the story breaks down in practice

- **REIT-equity convergence.** Clayton, J. & MacKinnon, G. (2003), *The Relative Importance of
  Stock, Bond and Real Estate Factors in Explaining REIT Returns*, Journal of Real Estate Finance
  and Economics — shows the equity-market factor dominates REIT return variation by the 2000s,
  displacing the earlier real-estate-specific factor. Our full-sample VNQ–SPY correlation of
  0.745 is consistent with this finding.
- **Interest-rate sensitivity of REITs.** Bredin, D., O'Reilly, G. & Stevenson, S. (2007),
  *Monetary Shocks and REIT Returns*, Journal of Real Estate Finance and Economics — documents
  that REITs are highly sensitive to interest rate changes, creating a dual headwind in rising-
  rate / high-inflation environments (both the discount rate and cap rates widen). This explains
  VNQ's −31.9% in 2022 even as inflation was high.
- **Correlation spikes in crises.** Longin, F. & Solnik, B. (2001), *Extreme Correlation of
  International Equity Markets*, Journal of Finance — establishes the general phenomenon that
  cross-asset correlations spike in bear markets. Publicly traded REITs are not exempt: our
  rolling correlation analysis shows the 63-day VNQ–SPY correlation reached 0.95 during the
  GFC, precisely when diversification was most needed.
- **The inflation-REIT claim debunked.** Simpson, M.W., Ramchander, S. & Webb, J.R. (2007),
  *The Asymmetric Response of Equity REIT Returns to Inflation*, Journal of Real Estate Finance
  and Economics — finds that equity REIT returns actually respond negatively to *unexpected*
  inflation (because of the interest-rate transmission) even if they respond positively to
  *expected* inflation in the short run. Our four high-CPI years (2021–2022, 2007, 2005)
  average −2.7% for VNQ, confirming the net effect is negative.

## The desk's shared tooling and method lineage

- **HAC / Newey-West t-stat on annual differences.** Newey, W.K. & West, K.D. (1987),
  *A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent
  Covariance Matrix*, Econometrica — annual granularity matches the rebalance cadence and
  avoids within-year autocorrelation. Implemented in `strategy.hac_tstat_annual`.
- **Block bootstrap Sharpe CI.** Politis, D.N. & Romano, J.P. (1994), *The Stationary
  Bootstrap*, Journal of the American Statistical Association — circular block resampling
  preserves volatility clustering in the joint resample of the two arms. Implemented in
  `strategy.bootstrap_sharpe_diff`.
- **Sharpe annualisation and inference.** Lo, A.W. (2002), *The Statistics of Sharpe Ratios*,
  Financial Analysts Journal — the i.i.d. delta-method SE and the autocorrelation-robust
  adjustment.

## Related desk studies

- **[Study 144 — Permanent-Portfolio](../../144-permanent-portfolio/)**: the 25/25/25/25
  SPY/TLT/GLD/SHY blend — another allocation study that tests regime diversification, using
  the same crisis-episode and regime-analysis machinery.
- **[Study 171 — Naive-1/N](../../171-naive-1-over-n/)**: equal-weight across eleven SPDR
  sectors — tests whether simplicity beats Markowitz in the sector space, the same
  "diversification does it actually help?" question applied to equities.
- **[Study 68 — All-Weather](../../68-all-weather/)** and **[Study 97 — Balancing-Act](../../97-balancing-act/)**: risk-parity
  approaches that, like the PP, claim to produce regime-robust diversification.
- **[Study 69 — Safe-Haven](../../69-safe-haven/)**: tests whether gold (GLD) genuinely
  hedges equity crashes — the one non-REIT asset in the panel that our crisis analysis shows
  to be a real diversifier (positive in 4 of 5 crisis episodes vs VNQ's 2 of 5).

## Data sources used here

- **VNQ (Vanguard Real Estate ETF)**, incepted 2004-09-29 — the liquid, institutionally held
  proxy for publicly traded US REITs. Source: `_cache/cross_asset_etfs.parquet` (yfinance
  `auto_adjust=True`, total-return). The joint window with GLD starts 2004-11-18.
- **SPY, TLT, IEF, SHY, GLD** — the comparison universe from the same shared cache.
- **Shiller monthly dataset** (`_cache/shiller_sp500.parquet`) — CPI column for inflation
  regime classification. We compute year-over-year inflation from the monthly series and
  classify each calendar year into terciles (low / medium / high CPI environment). The
  classification uses 1871–2026 tercile thresholds so the sample is not forward-looking.
