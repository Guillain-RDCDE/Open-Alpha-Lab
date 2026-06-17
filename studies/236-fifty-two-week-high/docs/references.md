# References & literature map — Study 236 (Fifty-Two-Week-High)

## The claim under test

The 52-week-high momentum strategy is the direct application of George & Hwang (2004):
*rank stocks by proximity to their 52-week high (close / 252d rolling high); buy the
quintile nearest their highs and short the quintile farthest away.*  The argument is
anchoring-based: investors are reluctant to bid stocks above a salient reference price
(the 52-week high), causing underreaction to good news for near-high stocks and
underreaction to continued bad news for far-from-high stocks.  The spread (near-high
minus far-from-high) should be positive.  We steelman it as: *the cross-sectional rank
by proximity to 52-week high forecasts positive forward returns, with near-high stocks
outperforming far-from-high stocks.*

## The primary empirical literature

- **George, T.J. & Hwang, C.-Y. (2004)**, *The 52-week high and momentum investing*,
  Journal of Finance, 59(5), 2145–2176.  The canonical paper: stocks near their
  52-week high earn significantly higher returns over the next 6-12 months than
  stocks far from their highs.  The signal subsumes Jegadeesh-Titman 6-month momentum
  on US stocks, 1963-2001, NYSE/AMEX/Nasdaq universe.  Key detail: the original
  universe is broad (thousands of stocks) and the holding period is 6-12 months —
  neither of which matches our large-cap, 1-13 week test.

- **Jegadeesh, N. & Titman, S. (1993)**, *Returns to buying winners and selling losers:
  implications for stock market efficiency*, Journal of Finance, 48(1), 65–91.
  The foundational cross-sectional momentum paper.  Past 3-12 month winners outperform
  losers by ~1%/month.  The 52-week-high proximity is correlated with past 12-month
  returns (stocks near their high have typically had high past returns).

- **Liu, M., Liu, Q. & Ma, T. (2011)**, *The 52-week high momentum strategy in
  international stock markets*, Journal of International Money and Finance, 30(1),
  180–204.  Documents the George-Hwang effect across 20 international markets with
  broadly similar results, suggesting it is not US-specific.

- **Marshall, B.R. & Cahan, R.M. (2005)**, *Is the 52-week high momentum strategy
  profitable outside the US?*, Applied Financial Economics, 15(18), 1259–1267.
  Finds the strategy profitable in Australia and New Zealand, but notes sensitivity
  to market conditions and universe construction.

- **Novy-Marx, R. (2012)**, *Is momentum really momentum?*, Journal of Financial
  Economics, 103(3), 429–453.  Shows that intermediate-horizon (7-12 month) return
  momentum drives most of the classic momentum premium; the most recent 1-month return
  reverses.  The 52-week-high proximity is a proxy for 12-month momentum.

## Why the anomaly might not replicate on our sample

- **Hou, K., Xue, C. & Zhang, L. (2020)**, *Replicating anomalies*, Review of
  Financial Studies, 33(5), 2019–2133.  Comprehensive replication study finding that
  many published factor anomalies fail to replicate out-of-sample, particularly when
  tested on smaller, more liquid, and more recent samples.

- **McLean, R.D. & Pontiff, J. (2016)**, *Does academic publication erode stock return
  predictability?*, Journal of Finance, 71(1), 5–32.  Anomalies decay after
  publication as arbitrageurs trade them away; by 2013-2026 the George-Hwang effect
  may have been largely arbitraged in large-cap names.

- **Survivorship bias and universe composition.** Our 20-name basket consists of current
  S&P 500 blue chips.  Within this elite group, a stock "far from its 52-week high" is
  temporarily soft but fundamentally sound — the opposite of a traditional momentum short
  leg.  Baker & Wurgler (2006) note that hard-to-arbitrage, low-quality stocks drive most
  contrarian/momentum effects.  Restricting to top-quality survivors changes the regime.

## Behavioural finance foundations

- **Tversky, A. & Kahneman, D. (1974)**, *Judgment under uncertainty: heuristics and
  biases*, Science, 185(4157), 1124–1131.  Anchoring heuristic: the 52-week high is a
  salient anchor that analysts and investors use when evaluating whether a stock is
  "cheap" or "expensive" relative to its recent history.

- **Shiller, R.J. (2000)**, *Irrational Exuberance*, Princeton University Press.
  Documents the role of reference prices and historical comparisons in equity valuation
  decisions, providing the behavioural backdrop for the anchoring mechanism.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.summarize`](../fifty_two_week_high/strategy.py) uses the inline Bartlett-kernel
  implementation, consistent with Studies 202 and 127.

- **Cross-sectional quintile sort.** Fama & French (1992), *The cross-section of expected
  stock returns* (Journal of Finance) — the standard portfolio-sort methodology applied
  to the proximity signal, equal-weight within quintile, mean forward return as the test
  statistic.

- **Survivorship bias naming convention.** The desk requires explicit naming whenever a
  study's universe is restricted to surviving firms (see METHODOLOGY.md).

## Related desk studies

- **[Study 202 — Fifty-Two-Week-Low](../../202-fifty-two-week-low/)**: the contrarian
  mirror of this study — stocks near their 52-week *low*.  Also fails on this sample,
  consistent with mild momentum (low-proximity names trail high-proximity names).

- **[Study 107 — Faber-Timing](../../107-faber-timing/)**: tactical allocation based on
  the 10-month SMA — a related trend/momentum signal at a longer horizon.

- **[Study 103 — Turtle](../../103-turtle/)**: systematic trend following (Donchian
  channel breakout) — a related breakout rule that uses the 52-week *range* for entries.

- **[Study 106 — Supertrend](../../106-supertrend/)**: ATR-based trend indicator that
  shares the same "ride the winner" logic as 52-week-high momentum.
