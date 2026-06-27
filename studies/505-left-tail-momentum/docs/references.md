# References & literature map -- Study 505 (Left-Tail-Momentum)

## The primary claim under test

- **Atilgan, Y., Bali, T. G., Demirtas, K. O. & Gunaydin, A. D. (2020).** "Left-Tail Momentum:
  Underreaction to Bad News, Costly Arbitrage and Equity Returns." *Journal of Financial
  Economics*, 135(3), 725--753. The founding paper. Stocks with the worst left-tail risk -- the
  lowest 1%/5% Value-at-Risk or Expected Shortfall over the trailing year -- **continue** to
  earn low returns over the following month. Investors under-react to extreme negative returns
  (bad news), so the left tail has *momentum*: crashed stocks keep crashing. The effect is
  strongest where arbitrage is costly (illiquid, high idiosyncratic-vol names). A long-low-risk
  / short-high-risk (long-safe / short-crashed) book earns a positive risk-adjusted spread on a
  broad US universe (1962--2014).

## The tension -- left-tail momentum vs. short-run reversal

- **Bali, T. G., Cakici, N. & Whitelaw, R. F. (2011).** "Maxing Out: Stocks as Lotteries and the
  Cross-Section of Expected Returns." *Journal of Financial Economics*, 99(2), 427--446. The MAX
  (lottery) effect: stocks with the highest *recent single-day gains* underperform. The mirror of
  the left tail -- the *right* tail predicts low returns. The desk tested this directly in
  [Study 53 -- Jackpot](../../53-jackpot/) and [Study 365 -- Lottery-MAX](../../365-lottery-max-effect/),
  both `None x Mirage` on large-cap survivors.
- **Jegadeesh, N. (1990).** "Evidence of Predictable Behavior of Security Returns." *Journal of
  Finance*, 45(3), 881--898. Short-run (one-month) **reversal**: last month's losers beat last
  month's winners next month. This is the force that competes with -- and on a survivor basket
  *dominates* -- left-tail momentum: recently-crashed survivors bounce.
- **Da, Z., Liu, Q. & Schaumburg, E. (2014).** "A Closer Look at the Short-Term Return Reversal."
  *Management Science*, 60(3), 658--674. Decomposes reversal; the residual reversal is strongest
  in exactly the high-volatility names that also carry the worst left tails -- which is why the
  two effects collide.

## Why the effect should exist -- the behavioural backbone

- **Barberis, N., Shleifer, A. & Vishny, R. (1998).** "A Model of Investor Sentiment." *Journal
  of Financial Economics*, 49(3), 307--343. Under-reaction to news as a driver of momentum --
  the mechanism ABDG invoke for the left tail specifically.
- **Kahneman, D. & Tversky, A. (1979).** "Prospect Theory: An Analysis of Decision under Risk."
  *Econometrica*, 47(2), 263--291. Loss aversion and probability weighting of extreme outcomes;
  the behavioural foundation for why extreme negative returns are mispriced.

## Tail-risk pricing more broadly

- **Kelly, B. & Jiang, H. (2014).** "Tail Risk and Asset Prices." *Review of Financial Studies*,
  27(10), 2841--2871. A common tail-risk factor: stocks more exposed to aggregate tail risk earn
  higher average returns -- a *positive* risk premium, the opposite sign to ABDG's behavioural
  under-reaction, underscoring how delicate the left-tail/return relation is.
- **Bali, T. G., Cakici, N. & Whitelaw, R. F. (2014).** "Hybrid Tail Risk and Expected Stock
  Returns: When Does the Tail Wag the Dog?" *Review of Asset Pricing Studies*, 4(2), 206--246.
  Systematic vs idiosyncratic tail risk carry different signs -- the same delicacy.

## Survivorship bias and universe construction

- **Shumway, T. (1997).** "The Delisting Bias in CRSP Data." *Journal of Finance*, 52(1),
  327--340. Delistings are correlated with terrible prior performance. Removing them biases
  factor returns. For a *left-tail* sort this bias is acute: the worst-tail names are the most
  likely to delist, so a survivor basket systematically drops the very stocks the short leg
  needs -- and a survivor basket's "crashed" names are disproportionately *rebounds*.
- **Linnainmaa, J. T. & Roberts, M. R. (2018).** "The History of the Cross-Section of Stock
  Returns." *Review of Financial Studies*, 31(7), 2606--2649. Many cross-sectional anomalies are
  weaker or vanish out-of-sample and pre-sample; data-snooping and survivorship matter.

## Method lineage (the desk's shared engine)

- **Newey, W. K. & West, K. D. (1987).** "A Simple, Positive Semi-Definite, Heteroskedasticity
  and Autocorrelation Consistent Covariance Matrix." *Econometrica*, 55(3), 703--708. The HAC
  long-run-variance t-stat reported alongside the plain one-sample t in
  [`strategy.summary`](../left_tail_momentum/strategy.py).

## Related desk studies

- **[Study 53 -- Jackpot](../../53-jackpot/)**: the MAX / lottery effect (right-tail) -- the
  mirror of this study; `None x Mirage`.
- **[Study 365 -- Lottery-MAX-Effect](../../365-lottery-max-effect/)**: a second pass on the
  lottery tail; `None x Mirage`.
- **[Study 332 -- Downside-Beta](../../332-downside-beta/)**: co-crash beta pricing -- another
  left-tail risk lens; `Mixed x Mirage`.
- **[Study 330 -- Low-Volatility-Anomaly](../../330-low-volatility-anomaly/)**: calm-beats-wild
  risk-adjusted; same engine family (rolling sort, HAC inference, costs + borrow).
- **[Study 238 -- Betting-Against-Beta](../../238-betting-against-beta/)**: the BAB factor --
  the closest cross-sectional-risk cousin of this study.
