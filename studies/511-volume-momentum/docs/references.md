# References & literature map -- Study 511 (Volume-Momentum, Lee-Swaminathan)

## The primary claim under test

- **Lee, C. M. C. & Swaminathan, B. (2000).** "Price Momentum and Trading Volume." *Journal of
  Finance*, 55(5), 2017-2069. The founding paper. Trading volume predicts both the **magnitude**
  and the **persistence** of price momentum. Double-sorting stocks on past return (winners /
  losers) THEN on past **trading volume** (turnover) reveals a "momentum life cycle":
  intermediate-horizon momentum is strongest among **high-volume winners** and **low-volume
  losers**; and -- their most distinctive prediction -- **high-volume past performers reverse
  faster**, so volume forecasts *when* the drift turns into long-horizon reversal. On the broad
  CRSP universe (1965-1995) the high-volume momentum spread is large and the volume-conditioned
  reversal is pronounced.

## The momentum factor it conditions

- **Jegadeesh, N. & Titman, S. (1993).** "Returns to Buying Winners and Selling Losers:
  Implications for Stock Market Efficiency." *Journal of Finance*, 48(1), 65-91. The canonical
  3-12 month relative-strength momentum that volume conditions. The base effect this desk finds
  flat on a large-cap survivor basket (see [507](../../507-cross-sectional-momentum/)).
- **Jegadeesh, N. & Titman, S. (2001).** "Profitability of Momentum Strategies: An Evaluation of
  Alternative Explanations." *Journal of Finance*, 56(2), 699-720. Confirms momentum persists then
  reverses at long horizons -- the reversal that Lee-Swaminathan claim volume *times*.

## Trading volume as a state variable

- **Datar, V. T., Naik, N. Y. & Radcliffe, R. (1998).** "Liquidity and Stock Returns: An
  Alternative Test." *Journal of Financial & Markets*, 1(2), 203-219. Turnover as a priced
  liquidity characteristic -- the same conditioner this desk tests directly in
  [141 Turnover-Anomaly](../../141-turnover-anomaly/).
- **Gervais, S., Kaniel, R. & Mingelgrin, D. H. (2001).** "The High-Volume Return Premium."
  *Journal of Finance*, 56(3), 877-919. Extreme-volume days predict subsequent returns -- volume
  as an attention/visibility shock, the channel Lee-Swaminathan exploit.
- **Conrad, J. S., Hameed, A. & Niden, C. (1994).** "Volume and Autocovariances in Short-Horizon
  Individual Security Returns." *Journal of Finance*, 49(4), 1305-1329. High-volume securities
  show stronger return reversals -- an early thread of the volume-conditioned reversal idea.

## Why under-reaction would track volume

- **Hong, H. & Stein, J. C. (1999).** "A Unified Theory of Underreaction, Momentum Trading, and
  Overreaction in Asset Markets." *Journal of Finance*, 54(6), 2143-2184. Gradual information
  diffusion generates momentum then overreaction-driven reversal; trading volume proxies how far
  along that cycle a stock sits -- the theoretical backbone of the life-cycle hypothesis.
- **Barberis, N., Shleifer, A. & Vishny, R. (1998).** "A Model of Investor Sentiment." *Journal of
  Financial Economics*, 49(3), 307-343. Under-reaction then over-reaction -- the behavioural arc
  the momentum life cycle traces.

## Subsequent evidence, replication, and attenuation

- **Hou, K., Xue, C. & Zhang, L. (2020).** "Replicating Anomalies." *Review of Financial Studies*,
  33(5), 2019-2133. Volume-conditioned and other momentum-refinement anomalies replicate with much
  weaker magnitudes on broad, recent samples; large-cap-only universes are the harshest test.
- **McLean, R. D. & Pontiff, J. (2016).** "Does Academic Research Destroy Stock Return
  Predictability?" *Journal of Finance*, 71(1), 5-32. ~32% post-publication attenuation; the
  Lee-Swaminathan effect (published 2000) is deep in the post-publication window.
- **Chordia, T., Subrahmanyam, A. & Anshuman, V. R. (2001).** "Trading Activity and Expected Stock
  Returns." *Journal of Financial Economics*, 59(1), 3-32. The volume-return relation is fragile
  and sign-unstable out of sample -- consistent with the flat, sign-inverted result on this tape.

## Survivorship bias and universe construction

- **Shumway, T. (1997).** "The Delisting Bias in CRSP Data." *Journal of Finance*, 52(1), 327-340.
  Delistings correlate with poor performance; a *quiet* (low-volume) slide into delisting is
  exactly a LOW-volume loser -- Lee-Swaminathan's strongest short leg -- so survivorship bias
  inflates the volume-conditioned slice the most. Named on the SIGNAL axis here.

## Method lineage (the desk's shared engine)

- **Newey, W. K. & West, K. D. (1987).** "A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix." *Econometrica*, 55(3), 703-708. The HAC *t*-stat
  in [`strategy.hac_tstat`](../volume_momentum/strategy.py).
- **Politis, D. N. & Romano, J. P. (1994).** Permutation / resampling inference -- the seed-robust
  label-shuffle placebo p-value that fails to certify the HIGH-volume WML mean.

## Related desk studies (the dedup map)

- **[Study 507 -- Cross-Sectional-Momentum](../../507-cross-sectional-momentum/)** -- the plain
  12-1 winners-minus-losers on the same basket family (the baseline conditioned here).
- **[Study 510 -- Frog-In-The-Pan](../../510-frog-in-the-pan/)** -- conditions momentum on
  information *continuity*; Study 511 conditions on *trading volume / turnover* -- an orthogonal
  state variable, same double-sort machinery.
- **[Study 509 -- Intermediate-Momentum](../../509-intermediate-momentum/)** -- conditions
  momentum on the *timing* of past returns; Study 511 on their *trading volume*.
- **[Study 141 -- Turnover-Anomaly](../../141-turnover-anomaly/)** -- turnover as a *standalone*
  return predictor; Study 511 uses turnover only to *condition* momentum, not as the signal itself.
- **[Study 24 -- Stampede](../../24-stampede/)** -- cross-sectional momentum on the full S&P 500.
- House methodology: [`METHODOLOGY.md`](../../../METHODOLOGY.md) -- the |t| >= 2 inference bar, the
  excess-of-cash Sharpe rule, one execution lag, costs one-way × NAV with shorts paying borrow.
