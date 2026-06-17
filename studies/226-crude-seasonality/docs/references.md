# References & literature map — Study 226 (Crude-Seasonality)

## The claim and its sources

- **Hamilton, J. D. (1983).** *Oil and the Macroeconomy since World War II.* Journal of Political
  Economy 91(2), 228–248 — the foundational study linking oil-price shocks to economic cycles,
  from which seasonal intuitions were later drawn.
- **Milonas, N. T., & Henker, T. (2001).** *Price Spread and Convenience Yield Behaviour in the
  Petroleum Complex of Futures Markets.* Applied Financial Economics 11(1), 23–36 — documents
  seasonal patterns in crude futures and the convenience yield around refinery seasonal demand.
- **Kilian, L. (2009).** *Not All Oil Price Shocks Are Alike: Disentangling Demand and Supply
  Shocks in the Crude Oil Market.* American Economic Review 99(3), 1053–1069 — supply vs demand
  decomposition; seasonal demand shocks from driving-season gasoline demand are real but modest.
- **Gorton, G., & Rouwenhorst, K. G. (2006).** *Facts and Fantasies about Commodity Futures.*
  Financial Analysts Journal 62(2), 47–68 — seasonal variation in commodity returns is a known
  feature; the backwardation/contango cycle is seasonal for crude and products.

## On multiple testing and seasonality data-mining

- **Sullivan, R., Timmermann, A., & White, H. (2001).** *Dangers of Data Mining: The Case of
  Calendar Effects in Stock Returns.* Journal of Econometrics 105(1), 249–286 — a direct warning
  that calendar effects found in exploratory screens rarely survive post-publication; Bonferroni
  correction is the minimum due diligence.
- **McLean, R. D., & Pontiff, J. (2016).** *Does Academic Research Destroy Stock Return
  Predictability?* Journal of Finance 71(1) — post-publication decay applies equally to seasonal
  patterns as to factor premia.

## Data

- **Yahoo! Finance** — WTI crude front future (CL=F), energy equities ETF (XLE) and the 13-week
  T-bill (^IRX, the cash leg), 2000–2026, **daily** closes resampled to month-end. CL=F begins in
  September 2000 (309 months total). XLE begins January 1999. T-bill is used as the cash leg for
  the timer and for excess-Sharpe computation. The study-local cache lives at
  `_cache/crude_seasonality.parquet` (gitignored).

*Companion studies: [49 Black-Gold](../../49-black-gold/) (oil→equities cross-asset); the monthly
calendar-effect family includes [82 Witching-Hour](../../82-witching-hour/),
[89 Turn-of-the-Month](../../89-turn-of-the-month/), and [95 Holiday-Cheer](../../95-holiday-cheer/).*
