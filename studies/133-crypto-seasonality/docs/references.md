# References & literature map — Study 133 (Crypto-Seasonality)

## The claim under test

- **The folk calendar.** A staple of crypto social media: *"October is 'Uptober' —
  Bitcoin almost always rallies. September is 'Rektember' — it's reliably the worst
  month. Q1 is when altcoins explode. Trade accordingly."* The claim circulates widely
  on Twitter/X, Reddit, and in crypto media every year, backed by year-by-year return
  tables that cherry-pick the evidence. We steelman it as: *specific calendar months
  have expected BTC returns that differ reliably from the all-month baseline, and this
  difference is large enough to support a tradable seasonal rule.* Those are exactly the
  two hypotheses we measure.

## Seasonality in asset returns — the general literature

- **Fama (1970)** — *Efficient Capital Markets: A Review of Theory and Empirical Work*
  (Journal of Finance). The baseline: in an efficient market, predictable calendar
  effects should be arbitraged away. Monthly seasonality in an actively traded,
  globally accessible asset with low barriers to entry faces this prior.

- **French (1980)** — *Stock Returns and the Weekend Effect* (Journal of Financial
  Economics). One of the earliest well-documented seasonal anomalies — the Monday
  effect in equity returns. Established the template for testing calendar effects and
  the importance of multiple-comparison corrections when searching across days/months.

- **Keim (1983)** — *Size-Related Anomalies and Stock Return Seasonality* (Journal of
  Financial Economics). The 'January effect' in small-cap stocks — a classic calendar
  anomaly that has subsequently weakened (McLean & Pontiff 2016). Shows that seasonal
  signals identified in-sample often decay out-of-sample, especially once publicised.

- **Sullivan, Timmermann & White (2001)** — *Dangers of Data Mining: The Case of
  Calendar Effects in Stock Returns* (Journal of Econometrics). Formal treatment of the
  multiple-comparison problem in calendar anomaly research. Testing many calendar
  patterns on the same data inflates apparent significance; the study uses the White
  Reality Check to correct for this. Directly relevant here: we test 12 months and must
  correct for 12 hypotheses.

## Cryptocurrency-specific seasonality

- **Baur & Dimpfl (2021)** — *The Volatility of Bitcoin and Its Role as a Medium of
  Exchange and a Store of Value* (Empirical Economics). Documents BTC's time-varying
  volatility and stylised facts including high auto-correlation in volatility and
  fat-tailed returns — the distributional properties that make monthly-return inference
  hard (large std dev per month, roughly 15–25%, makes t-stats low-powered).

- **Kurihara & Fukushima (2017)** — *The Market Efficiency of Bitcoin: A Weekly
  Seasonal Adjustment Approach* (International Journal of Financial Research). Finds
  weak day-of-week effects in BTC returns but concludes the market is increasingly
  efficient. Sets the expectation that calendar effects in crypto are fragile.

- **Caporale & Plastun (2019)** — *The Day of the Week Effect in the Crypto Currency
  Market* (Finance Research Letters). Tests seven-day-of-week seasonality in four
  major crypto assets; finds some day effects but notes the results are data-period
  sensitive. Exactly the instability concern that motivated this study.

## Multiple comparisons and the inference bar

- **Miller (1981)** — *Simultaneous Statistical Inference*, 2nd edition (Springer).
  The foundational reference for the Bonferroni correction. When testing $m$
  hypotheses simultaneously, the family-wise error rate at $\alpha$ requires each
  individual test to use threshold $\alpha/m$. For 12 months at $\alpha = 0.05$,
  the per-test threshold becomes $\approx 0.00417$, equivalent to $|t| \approx 3.1$
  at the degrees of freedom available (n≈12 per month, approximately t₁₁).

- **Harvey, Liu & Zhu (2016)** — *... and the Cross-Section of Expected Returns*
  (Review of Financial Studies). A systematic review of hundreds of published factors
  and their implied false discovery rates. Argues that a naive |t| > 2 bar is
  insufficient given the volume of academic factor testing; proposes t > 3 as a
  more appropriate threshold. Our Bonferroni correction is in the same spirit.

## Methodology and inference tools used here

- **Newey & West (1987)** — *A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix* (Econometrica). The HAC long-run
  variance estimator underlying the t-stats in `strategy.month_stats` and
  `strategy.summarize`. Monthly BTC returns exhibit some autocorrelation and strong
  heteroskedasticity, making HAC inference appropriate over the naive OLS standard error.

- **Lo (2002)** — *The Statistics of Sharpe Ratios* (Financial Analysts Journal). The
  formula for the delta-method SE of an estimated Sharpe ratio, used in
  `quantlab.analytics.sharpe_with_se` for the buy-and-hold and strategy comparison.

## Related desk studies

- **[Study 83 — Half-Life](../../83-half-life/)**: tests whether Bitcoin halving events
  predict outsized post-halving returns. Same structural power problem: n=3–4 events.
  The verdict is WEAK/NONE for the same reasons — small effective sample, fat-tailed
  returns, high volatility.

- **[Study 84 — Moon-Math](../../84-moon-math/)**: lunar-cycle effects on BTC prices.
  Another calendar claim with an even smaller effective n per cycle; found to be NONE.

- **[Study 117 — Pi-Cycle-Top](../../117-pi-cycle-top/)**: a technical indicator for
  BTC cycle tops using moving average crosses. Related: calendar lore often aligns with
  cycle theories, but both face the same low-power problem on a 10-year history.

- **[Study 55 — Summer-Lull](../../55-summer-lull/)**: equity monthly seasonality
  ("sell in May and go away") — a directly analogous calendar claim for equities that
  is also WEAK/MIRAGE. The multiple-comparison problem is identical.

- **[Study 48 — Groundhog](../../48-groundhog/)**: a calendar/event effect on equities.
  Another case where the effect looks compelling in the folklore but faces the
  small-n / multiple-comparison problem in formal testing.
