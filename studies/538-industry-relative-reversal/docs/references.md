# References & literature map -- Study 538 (Industry-Relative-Reversal)

## The canonical claim

- **Hameed, A. & Mian, G. M. (2015).** *Industries and Stock Return Reversals.*
  Journal of Financial and Quantitative Analysis, 50(1-2), 89-117.
  The headline paper: the short-horizon (one-month) return reversal is concentrated in the
  **within-industry** component of returns. Decomposing a stock's prior-month return into an
  industry component and an industry-relative (idiosyncratic) component, the authors show it
  is the industry-relative part that reverses; the industry part does not. An industry-
  relative reversal strategy is substantially stronger than the raw Jegadeesh (1990)
  reversal -- exactly the head-to-head we replicate (RAW vs IRR).

- **Da, Z., Liu, Q. & Schaumburg, E. (2014).** *A Closer Look at the Short-Term Return
  Reversal.* Management Science, 60(3), 658-674.
  Decomposes the one-month reversal into a component driven by across-industry (sector)
  cash-flow news, which does *not* reverse, and a residual (industry-relative) component
  that does. Removing the non-reversing sector/news component roughly **doubles** the
  reversal's information ratio and makes it more robust to microstructure -- the mechanism
  behind our finding that the IRR spread survives a one-month gap while the raw one dies.

## The reversal it refines

- **Jegadeesh, N. (1990).** *Evidence of Predictable Behavior of Security Returns.*
  Journal of Finance, 45(3), 881-898.
  The founding short-horizon reversal: monthly stock returns have strong negative first-
  order serial correlation; buy prior-month losers, sell prior-month winners. This is the
  RAW signal in our head-to-head and the whole subject of [Study 329](../../329-one-month-reversal/).

- **Lehmann, B. N. (1990).** *Fads, Martingales, and Market Efficiency.*
  Quarterly Journal of Economics, 105(1), 1-28.
  The weekly-horizon contrarian companion to Jegadeesh; the same overreaction-vs-liquidity-
  provision debate the industry decomposition tries to settle.

## Why raw reversal is contaminated -- microstructure

- **Lo, A. W. & MacKinlay, A. C. (1990).** *When Are Contrarian Profits Due to Stock Market
  Overreaction?* Review of Financial Studies, 3(2), 175-205.
  Shows much of short-horizon contrarian profit comes from cross-autocovariances (lead-lag)
  and from spurious negative own-autocorrelation due to non-synchronous trading and the
  bid-ask bounce. The industry adjustment is partly an attempt to net out the common
  (industry lead-lag) part, which is why the IRR spread behaves so differently from the raw
  one under our skip=1 gap test.

- **Roll, R. (1984).** *A Simple Implicit Measure of the Effective Bid-Ask Spread.*
  Journal of Finance, 39(4), 1127-1139.
  Transaction prices bouncing between bid and ask induce mechanical negative serial
  correlation -- the artefact that kills Study 329's raw reversal at skip=1 but leaves the
  industry-relative version standing.

## Costs, capacity, decay

- **Avramov, D., Chordia, T. & Goyal, A. (2006).** *Liquidity and Autocorrelations in
  Individual Stock Returns.* Journal of Finance, 61(5), 2365-2394.
  The one-month reversal is overwhelmingly a function of illiquidity -- strong where it
  cannot be traded cheaply. Our ~77% monthly turnover and ~3 bps break-even are the cost
  face of the same fact.

- **McLean, R. D. & Pontiff, J. (2016).** *Does Academic Research Destroy Stock Return
  Predictability?* Journal of Finance, 71(1), 5-32.
  Anomaly returns fall ~58% post-publication. A modern, large-cap, cost-charged replication
  of even a refined reversal landing WEAK/Mirage is the expected outcome.

## The survivorship-bias dimension

- **Shumway, T. (1997).** *The Delisting Bias in CRSP Data.* Journal of Finance, 52(1),
  327-340.
  Delisted firms earn extreme negative returns but are missing from vendor data. Our basket
  (current S&P 500 backwards) excludes them, biasing loser-leg returns upward; all real-tape
  results are upper bounds, named on the Signal axis.

## Related desk studies (the reversal spectrum)

- **[Study 329 -- One-Month-Reversal](../../329-one-month-reversal/)**: the RAW Jegadeesh
  (1990) reversal this study refines. 329 finds the raw monthly reversal is real only as a
  1990s bid-ask-bounce rent; 538 asks whether netting out the industry component leaves a
  cleaner, tradable residual. (It leaves a cleaner *signal* -- IRR beats RAW and survives the
  gap -- but still no tradable edge on a survivor basket.)
- **[Study 196 -- Long-Term-Reversal](../../196-long-term-reversal/)**: De Bondt-Thaler at
  36-60 months -- the opposite end of the return-autocorrelation spectrum.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), Econometrica --
  [`strategy.summarize`](../industry_relative_reversal/strategy.py).
- **Placebo / label-permutation null.** A randomisation test on the industry map --
  [`strategy.placebo_irr_tstats`](../industry_relative_reversal/strategy.py): the right way
  to ask whether the "industry" in "industry-relative" is doing real work.
- **Survivorship notation.** Shumway (1997) -- named on the Signal axis; the basket is a
  fixed, explicitly-survivor list.
