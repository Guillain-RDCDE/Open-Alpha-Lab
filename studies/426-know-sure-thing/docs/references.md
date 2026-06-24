# References — Study 426 (Know Sure Thing / KST)

## The claim's source

- **Pring, Martin J.** *Technical Analysis Explained* (1985; 4th/5th eds.) and **"Summed Rate
  of Change (KST)"**, *Technical Analysis of Stocks & Commodities* (1992). Pring introduced the
  Know Sure Thing as a weighted sum of four smoothed rate-of-change series and named it the
  "Know Sure Thing" deliberately — the marketing is in the title. The folk rule: buy on a
  KST/signal-line bullish crossover, sell/flat on a bearish crossover.
- **pring.com** and the **StockCharts ChartSchool** entry on KST — the canonical daily
  parameters used here: ROC (10, 15, 20, 30), SMA (10, 10, 10, 15), signal SMA 9.
- **Investopedia — "Know Sure Thing (KST) Indicator"** — the popular how-to that states the
  crossover rule in the form we test.

## Trend-following / momentum — the steelman context

- **Faber, Mebane T.** "A Quantitative Approach to Tactical Asset Allocation," *Journal of Wealth
  Management* (2007). The 200-day SMA long/flat rule — our key simpler benchmark and the reason
  "beats a moving average" is the right myth-check.
- **Moskowitz, T., Ooi, Y. H., & Pedersen, L. H.** "Time Series Momentum," *Journal of Financial
  Economics* (2012). Establishes that trend persistence — the only thing a KST crossover can
  harvest — is real but asset- and horizon-dependent, and weakest on broad equity indices.
- **Hurst, B., Ooi, Y. H., & Pedersen, L. H.** "A Century of Evidence on Trend-Following
  Investing," AQR (2017). Trend works best where trends persist (commodities, FX, rates), least on
  large mean-reverting equity indices — context for the real-tape miss.

## Shared method citations

- **Newey, W. K., & West, K. D.** "A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix," *Econometrica* (1987). The HAC *t*-stat on the
  book's excess daily returns — the desk's inference bar.
- **Sharpe, W. F.** "The Sharpe Ratio," *Journal of Portfolio Management* (1994). Excess-of-cash,
  net-of-cost, risk-adjusted return — the metric every arm is raced on.
- **Sullivan, R., Timmermann, A., & White, H.** "Data-Snooping, Technical Trading Rule
  Performance, and the Bootstrap," *Journal of Finance* (1999). Why a single technical rule that
  looks good in-sample needs a placebo/permutation null and a comparison to the simplest
  alternative — the spirit of our sign-flip permutation and benchmark race.
- **White, H.** "A Reality Check for Data Snooping," *Econometrica* (2000). The selection problem
  behind picking a flattering parameter set; motivates the out-of-sample caveat in Beat 7.

## Related desk studies

- [`../105-coppock-curve`](../105-coppock-curve) — another long-term momentum oscillator turned
  into a market-timing rule; the closest cousin in construction.
- [`../110-faber-timing`](../110-faber-timing) — the 200-day SMA tactical rule used here as the
  decisive simpler benchmark.
- [`../180-trix`](../180-trix) and [`../185-chande-momentum`](../185-chande-momentum) — other
  smoothed-momentum oscillators raced on the same protocol.
- [`../104-bollinger-reversion`](../104-bollinger-reversion) and [`../178-cci`](../178-cci) —
  technical-indicator teardowns sharing the synthetic-control + honest-baseline idiom.
- [`../106-supertrend`](../106-supertrend) — a trend-filter timing rule with the same
  beta-vs-alpha caveat (a standalone *t* > 2 is the risk premium, not skill).
