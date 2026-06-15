# References & literature map — Study 202 (Fifty-Two-Week-Low)

## The claim under test

The 52-week-low contrarian strategy is the mirror of the well-known 52-week-high
momentum anomaly.  The folk version: *"Stocks near their 52-week low are beaten down,
oversold, and priced for bad news that's already in the price. Buy them as a
contrarian 'bargain bet' and wait for the mean reversion."*  The formal version
(George & Hwang 2004, see below) actually argues the *opposite* — that stocks near
their 52-week high outperform, not those near their low — but the contrarian reading
persists in retail forums and is routinely paired with the value-investing narrative.
We steelman it as: *the cross-sectional rank by proximity to 52-week low forecasts
positive forward returns, with near-low stocks outperforming near-high stocks.*

## The primary empirical literature

- **George, T.J. & Hwang, C.-Y. (2004)**, *The 52-week high and momentum investing*,
  Journal of Finance, 59(5), 2145–2176.  The canonical paper: proximity to the
  52-week *high* (not low) predicts future returns.  Anchoring to the high creates
  underreaction to good news — stocks near the high are less undervalued than
  investors believe.  This paper is about the *high*, not the low; the contrarian
  reading of the low is the orphaned flip-side.

- **Jegadeesh, N. & Titman, S. (1993)**, *Returns to buying winners and selling losers:
  implications for stock market efficiency*, Journal of Finance, 48(1), 65–91.
  The foundational momentum paper: past 3–12 month winners outperform losers by
  ~1%/month.  Stocks near their 52-week low have by definition been losers over the
  past year — the Jegadeesh-Titman result predicts they should *underperform*, which
  is exactly what we find.

- **DeBondt, W.F.M. & Thaler, R.H. (1985)**, *Does the stock market overreact?*,
  Journal of Finance, 40(3), 793–805.  The contrarian reversal paper: past 3-5 year
  losers outperform over the *following* 3-5 years.  This is long-horizon reversion,
  not short-horizon — and even here it depends on the full loser universe, not just
  proximity to the 52-week low.  Our study tests 1-to-65 day horizons.

- **Novy-Marx, R. (2012)**, *Is momentum really momentum?*, Journal of Financial
  Economics, 103(3), 429–453.  Decomposes cross-sectional momentum into a
  near-term reversal component and a longer-term continuation component; the
  52-week-low proximity is essentially a proxy for the continuation component
  (recent losers), not the reversal component.

- **Liu, M., Liu, Q. & Ma, T. (2011)**, *The 52-week high momentum strategy in
  international stock markets*, Journal of International Money and Finance, 30(1),
  180–204.  Documents the George-Hwang 52-week-high anomaly across 20 international
  markets.  Finds that the *high*-proximity sort is a more consistent predictor than
  intermediate-horizon returns, consistent with anchoring psychology.

## Why the contrarian story is theoretically coherent (but wrong here)

- **Baker, M. & Wurgler, J. (2006)**, *Investor sentiment and the cross-section of
  stock returns*, Journal of Finance, 61(4), 1645–1680.  Sentiment effects make
  hard-to-arbitrage, beaten-down stocks more sensitive to investor mood cycles;
  in principle contrarian bets could pay off when sentiment normalises.  But this
  mechanism operates over multi-year horizons, not the weekly-to-quarterly windows
  we test.

- **Shleifer, A. & Vishny, R.W. (1997)**, *The limits of arbitrage*, Journal of
  Finance, 52(1), 35–55.  Even if beaten-down stocks are genuinely undervalued,
  professional arbitrageurs may be unwilling to bet against the trend (career risk,
  margin calls) — leaving the mispricing in place for much longer than the 1-65 day
  windows tested here.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.summarize`](../fifty_two_week_low/strategy.py) uses the inline Bartlett-kernel
  implementation, the same as Studies 72 and 127.

- **Cross-sectional quintile sort.** Fama & French (1992), *The cross-section of expected
  stock returns* (Journal of Finance) — the standard portfolio-sort methodology applied
  to the proximity signal, equal-weight within quintile, mean forward return as the test
  statistic.

- **Survivorship bias naming convention.** The desk requires explicit naming whenever a
  study's universe is restricted to surviving firms (see METHODOLOGY.md).  Here the bias
  is doubly noted: our basket excludes delisted names, and the basket skews toward
  mega-cap survivors, which inflates all quintile returns.

## Related desk studies

- **[Study 50 — Fifty-Two-Week-High](../../50-fifty-two-week-high/)**: the momentum
  mirror of this study — stocks near their 52-week *high* do outperform.  The George
  & Hwang finding is the true anomaly; this study tests its contrarian inverse.

- **[Study 72 — Loaded-Dice](../../72-loaded-dice/)**: the 5-minute SMA crossover —
  another "simple rank-and-ride" rule that turns out to be a fair coin.

- **[Study 127 — Williams-R](../../127-williams-r/)**: Williams %R is the *single-stock*
  version of the short-term oversold/overbought call.  Also fails to produce a
  real contrarian edge.

- **[Study 106 — Supertrend](../../106-supertrend/)**: a trend-following indicator that
  shares the same fundamental failure mode — by the time the signal fires, the
  information is already in the price.
