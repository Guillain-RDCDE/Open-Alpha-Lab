# References & literature map — Study 195 (Monthly-OpEx)

## The claim under test

Options expiration occurs on the third Friday of every calendar month (since 1973 for equity
options; SPY options since 1993).  Traders and researchers have claimed that the expiry
day and surrounding week carry: (1) elevated volume from hedging, rolling, and delta-unwinding
flows; (2) elevated intraday volatility ("gamma pinning" effects); and (3) a systematic
directional bias (the "expiry drift" or "pin risk" effect).  This study asks whether the
*monthly* cycle — not just the quarterly triple-witching studied in Study 82 — carries an
incremental measurable effect.

## Theoretical foundations

- **Options expiration mechanics.** Black & Scholes (1973), *The Pricing of Options and
  Corporate Liabilities* (Journal of Political Economy) — the delta-hedging framework implies
  that dealers who are net short gamma must continuously rebalance as expiry approaches,
  increasing turnover mechanically.  The magnitude is largest near the money and near expiry —
  the "charm" and "gamma" Greeks drive end-of-cycle hedging.
- **Gamma pinning / expiration pinning.** Ni, Pearson & Poteshman (2005), *Stock Price
  Clustering on Option Expiration Dates* (Journal of Financial Economics) — document that
  stock prices are attracted to option strike prices on expiration Fridays, consistent with
  dealer delta-hedging.  The effect is most pronounced for heavily traded strikes.
- **Volume and volatility around expiration.** Stoll & Whaley (1987), *Program Trading and
  Expiration Day Effects* (Financial Analysts Journal) — document elevated volume and
  intraday volatility on quarterly triple-witching days (the "Friday effect" literature).
  This study extends their quarterly finding to the monthly cycle.

## Evidence on the quarterly triple-witching (Study 82 baseline)

- **Witching-Hour (Study 82, this desk)** — tests the quarterly triple-witching (Mar/Jun/Sep/Dec)
  on SPY 1993–2026 and finds a real volume uplift (~24% on the day, t = +5.72) and a
  negative return on the expiry day (−14 bps, t = −2.34), but no tradable weekly drift and a
  Tradability stamp of MIRAGE.  The present study's aggregate signals are consistent with
  this finding — and the incremental monthly decomposition shows that the monthly-only
  months add nothing beyond noise.
- **Bhattacharya (1987)**, *Price Changes of Related Securities: The Case of Call Options and
  Stocks* (Journal of Financial and Quantitative Analysis) — early documentation that
  options hedging flows affect underlying asset prices around expiration.
- **Chamberlain, Cheung & Kwan (1993)**, *The Hedging Effectiveness of Stock Index Futures:
  Evidence for the S&P 500 Index Options* — related evidence on futures/options expiration
  dynamics for the S&P 500.

## 0DTE options and the post-2010 structural break

- **0DTE (zero days to expiration) options.** CBOE introduced weekly options broadly from
  2005; the market exploded post-2019 to the point where 0DTE SPX options now represent
  more than 40–45% of daily SPX options volume (CBOE data, 2023).  The pre/post-2010 split
  in this study shows the volume uplift is a post-2010 phenomenon (t = +4.00 post vs −0.36
  pre), consistent with the growing hedging footprint of weeklies and monthlies.
- **Gârleanu, Pedersen & Poteshman (2009)**, *Demand-Based Option Pricing* (Review of
  Financial Studies) — shows that end-user demand imbalances in the options market affect
  underlying prices via dealer hedging; the 0DTE era amplifies this channel.
- **Muravyev & Pearson (2020)**, *Options Trading Costs Are Lower Than You Think* (Review of
  Financial Studies) — on the declining cost of options trading and the growing retail options
  participation; relevant context for the post-2010 volume surge.

## Multiple comparisons and inference bar

- **Bonferroni correction.** With six sub-tests (OpEx day vol, OpEx week vol, monthly-only
  vol, return day, return week, monthly-only return), a family-wise error rate of 5% requires
  each test to clear |t| ≥ 2.64 for a two-sided test.  The only sub-test clearing the
  unadjusted bar (t ≥ 2) is the volume-day aggregate (t = +4.44) and the OpEx-day return
  (t = −2.42); after the incremental decomposition, both are attributable to the quarterly
  sub-sample.
- **Harvey, Liu & Zhu (2016)**, *… and the Cross-Section of Expected Returns* (Review of
  Financial Studies) — argue for a t-stat threshold of 3.0 or higher for new factor discovery
  given the multiple testing in the empirical finance literature.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.compare_opex_vs_baseline`](../monthly_opex/strategy.py).
- **Year-demeaned log volume.** Removes the strong secular trend in ETF volume
  (SPY volume declined as the fund matured); delivers clean within-year comparisons.
- **Pre/post-2010 structural break.** Motivated by the CBOE weekly options expansion
  (2005 introduction, broad adoption 2010+) and the 0DTE explosion (post-2019).

## Related desk studies

- **[Study 82 — Witching-Hour](../../82-witching-hour/)**: the quarterly triple-witching
  (Mar/Jun/Sep/Dec) on SPY — the parent effect this study tries to extend monthly.  Signal
  MIXED (vol REAL, return WEAK), Tradability MIRAGE.
- **[Study 163 — Friday-13th](../../163-friday-13th/)**: another calendar anomaly test on the
  same daily SPY tape — zero-effect verdict, same null infrastructure.
- **[Study 48 — Groundhog](../../48-groundhog/)** and **[Study 136 — Mark-Twain](../../136-mark-twain/)**:
  other calendar studies in the same family, all finding NONE/WEAK effects.
