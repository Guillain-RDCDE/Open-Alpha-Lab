# References & literature map -- Study 329 (One-Month-Reversal)

## The canonical claim

- **Jegadeesh, N. (1990).** *Evidence of Predictable Behavior of Security Returns.*
  Journal of Finance, 45(3), 881-898.
  The founding short-horizon paper: monthly stock returns exhibit strong **negative**
  first-order serial correlation in the cross-section. A strategy that each month buys the
  prior-month losers and sells the prior-month winners earned ~2%/month gross in 1934-1987
  -- the canonical one-month reversal we replicate here.

- **Lehmann, B. N. (1990).** *Fads, Martingales, and Market Efficiency.*
  Quarterly Journal of Economics, 105(1), 1-28.
  An independent contemporaneous documentation at the **weekly** horizon: contrarian
  portfolios formed on one-week returns are profitable, which Lehmann attributes to either
  short-lived overreaction or to the rent earned for supplying liquidity to it.

## Why the steelman is almost right -- microstructure and the bid-ask bounce

- **Lo, A. W. & MacKinlay, A. C. (1990).** *When Are Contrarian Profits Due to Stock
  Market Overreaction?* Review of Financial Studies, 3(2), 175-205.
  Decomposes short-horizon contrarian profits and shows a large share comes from
  *cross-autocovariances* (lead-lag effects) and from spurious negative own-autocorrelation
  induced by non-synchronous trading and the bid-ask bounce -- not from genuine
  overreaction. This is exactly why our ``skip=1`` (one-month-gap) variant collapses the
  effect to *t* = -0.21: removing the contaminated formation/holding adjacency removes most
  of the "reversal."

- **Roll, R. (1984).** *A Simple Implicit Measure of the Effective Bid-Ask Spread in an
  Efficient Market.* Journal of Finance, 39(4), 1127-1139.
  The mechanism in one equation: transaction prices bouncing between bid and ask induce a
  mechanical negative serial correlation in measured returns. A stock that closed on the
  bid prints a low return and "reverts" upward when the next print lands on the ask --
  manufacturing a reversal with no information content.

- **Conrad, J., Gultekin, M. N. & Kaul, G. (1997).** *Profitability of Short-Term
  Contrarian Strategies: Implications for Market Efficiency.* Journal of Business, 70(3).
  Finds short-term contrarian profits are concentrated in the smallest, least-liquid,
  highest-spread stocks and largely vanish once realistic trading costs and the bid-ask
  bounce are accounted for -- the tradability MIRAGE we measure.

## Costs, capacity and post-publication decay

- **Avramov, D., Chordia, T. & Goyal, A. (2006).** *Liquidity and Autocorrelations in
  Individual Stock Returns.* Journal of Finance, 61(5), 2365-2394.
  Shows the one-month reversal is overwhelmingly a function of stock illiquidity: it is
  strong in illiquid names (where it cannot be traded cheaply) and weak in liquid names
  (where it could). The reversal is compensation for providing liquidity, not free alpha.

- **Khandani, A. E. & Lo, A. W. (2007).** *What Happened to the Quants in August 2007?*
  Journal of Investment Management, 5(4), 5-54.
  Documents that the returns to the classic mean-reversion strategy decayed sharply over
  1995-2007 as statistical-arbitrage capital crowded in -- consistent with our sub-period
  finding (1990-2002 *t* = +2.67, 2015-2026 *t* = -0.10).

- **McLean, R. D. & Pontiff, J. (2016).** *Does Academic Research Destroy Stock Return
  Predictability?* Journal of Finance, 71(1), 5-32.
  The general result: anomaly returns fall ~58% post-publication. The one-month reversal,
  published in 1990, is a textbook example of the fade.

## The survivorship-bias dimension

- **Shumway, T. (1997).** *The Delisting Bias in CRSP Data.* Journal of Finance, 52(1),
  327-340.
  Delisted firms earn extreme negative returns but are often missing from vendor data. Our
  universe (current S&P 500 backwards) excludes them, so the loser-leg return is biased
  upward; all real-tape results are upper bounds, named on the Signal axis.

## Related desk studies (the reversal spectrum)

- **[Study 33 -- Slingshot](../../33-slingshot/)**: the *daily*-rebalanced, dollar-neutral
  fade-against-peers book on the same names. Same anomaly family, much higher frequency and
  a different construction; 329 is the **monthly quintile** specification of Jegadeesh's
  Table I, so the two bracket the reversal horizon (daily vs monthly).
- **[Study 196 -- Long-Term-Reversal](../../196-long-term-reversal/)**: De Bondt-Thaler at
  **36-60 months**. The *opposite end* of the return-autocorrelation spectrum -- 329 is the
  1-month horizon, 196 the multi-year horizon. Same cross-sectional engine, opposite sign.
- **[Study 32 -- Rip-Tide](../../32-rip-tide/)**: short-term contrarian on liquid
  *futures*, not a stock cross-section -- found to be statistically nothing on deep markets.
- **[Study 251 -- Crypto-Reversal](../../251-crypto-reversal/)**: the same reversal-vs-
  momentum question in crypto.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), Econometrica --
  [`strategy.summarize`](../one_month_reversal/strategy.py).
- **Block-bootstrap Sharpe CI.** Politis & Romano (1994), *The Stationary Bootstrap* (JASA)
  -- [`quantlab.stats.sharpe_ci_bootstrap`](../../../quantlab/stats.py).
- **Survivorship notation.** Shumway (1997) -- named on the Signal axis; the panel loader
  goes through the opt-in ``quantlab.universe`` guard.
