# Sources & literature map — Study 995 (Whose Sharpe Is It?)

## Currency in a portfolio

- **Perold, A. F. & Schulman, E. C. (1988), "The Free Lunch in Currency Hedging: Implications
  for Investment Policy and Performance Standards", *Financial Analysts Journal* 44(3), 45-50.**
  The classic argument that currency exposure is uncompensated risk, so hedging it is free — the
  position this study's variance channel supports and its drift channel complicates.
- **Froot, K. A. (1993), "Currency Hedging over Long Horizons", NBER Working Paper 4355.** The
  counter-argument: over long horizons real exchange rates mean-revert, so the hedge's benefit
  shrinks with the holding period.
- **Campbell, J. Y., Serfaty-de Medeiros, K. & Viceira, L. M. (2010), "Global Currency Hedging",
  *Journal of Finance* 65(1), 87-121.** Where the variance-minimising hedge ratio comes from,
  and why it is not 100%: some currencies co-move with equities and therefore hedge them.
- **Black, F. (1989), "Universal Hedging: Optimizing Currency Risk and Reward in International
  Equity Portfolios", *Financial Analysts Journal* 45(4), 16-22.** The universal hedging ratio,
  and the first serious argument that the right answer is strictly between 0 and 1.

## The measurement problem

- **Sharpe, W. F. (1994), "The Sharpe Ratio", *Journal of Portfolio Management* 21(1), 49-58.**
  Sharpe's own restatement, which is explicit that the risk-free rate must be the investor's own
  — the point section 3 is built on and which practice routinely ignores.
- **Lo, A. W. (2002), "The Statistics of Sharpe Ratios", *Financial Analysts Journal* 58(4),
  36-52.** The sampling distribution, and why Sharpe differences of this size need care.
- **Solnik, B. (1974), "Why Not Diversify Internationally Rather Than Domestically?", *Financial
  Analysts Journal* 30(4), 48-54.** The founding paper of the whole question.

## Interest parity, which the rate channel rests on

- **Fama, E. F. (1984), "Forward and Spot Exchange Rates", *Journal of Monetary Economics* 14(3),
  319-338.** The forward premium puzzle — the reason the implied-rate approach in
  `implied_foreign_rate` is an approximation rather than an identity.
- **Du, W., Tepper, A. & Verdelhan, A. (2018), "Deviations from Covered Interest Rate Parity",
  *Journal of Finance* 73(3), 915-957.** Post-2008 CIP violations, and how large they got.

## Neighbours on this desk

**370-currency-hedging-costs**, **481-international-diversification**,
**744-dollar-strength-and-returns**, **970-annualisation-factors**.
