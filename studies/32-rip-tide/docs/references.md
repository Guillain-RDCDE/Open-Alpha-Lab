# Sources & literature map — Study 32 (Rip-Tide)

## The claim's source

- **Z. Kakushadze & J. A. Serur (2018), *151 Trading Strategies*, §10.3 — "Contrarian (mean-reversion)
  futures."** The book's catalogue entry for fading short-horizon moves on liquid futures, the mirror of
  the §10.4 trend-following entry that Study 31 (Trade-Winds) tested. SSRN `3247865` · arXiv
  [1912.04492](https://arxiv.org/abs/1912.04492). *(Copyrighted; not redistributed in this repo.)*

## Why short-term reversion is real in stocks but not in deep futures

- **Jegadeesh, N. (1990), "Evidence of Predictable Behavior of Security Returns," *Journal of Finance*
  45(3).** The foundational short-horizon (monthly/weekly) reversal result in individual equities.
- **Lehmann, B. (1990), "Fads, Martingales, and Market Efficiency," *QJE* 105(1).** Weekly contrarian
  profits in stocks — and the early argument that they are compensation for liquidity provision.
- **Lo, A. & MacKinlay, A. C. (1990), "When Are Contrarian Profits Due to Stock Market Overreaction?,"
  *Review of Financial Studies* 3(2).** Decomposes contrarian profit into own-autocorrelation,
  cross-autocorrelation and a cross-sectional variance term — much of it is *not* overreaction.
- **Avramov, Chordia & Goyal (2006), "Liquidity and Autocorrelations in Individual Stock Returns,"
  *Journal of Finance* 61(5).** Short-term reversal concentrates in *illiquid* stocks and largely
  vanishes once you account for trading frictions — the direct reason it does not survive on the
  deepest futures, the finding of this study.
- **Nagel, S. (2012), "Evaporating Liquidity," *Review of Financial Studies* 25(7).** Returns to
  short-term reversal strategies are a proxy for the *expected return on liquidity provision*; they
  spike in crises and are otherwise thin — consistent with a break-even cost below 1 bp on liquid
  instruments.

## The contrast study

- **Study 31 — Trade-Winds (§10.4 time-series momentum)**, [`../../31-trade-winds/`](../../31-trade-winds/).
  Same 18-futures tape, same equal-risk vol-targeted machinery, opposite signal sign. Trend is `REAL`
  (fragile standalone, real crisis alpha); reversion is `NONE`. The two studies bracket the
  autocorrelation question on one universe.

## The shared method

- **Newey, W. & West, K. (1987)** — HAC standard errors. **Lo, A. (2002), "The Statistics of Sharpe
  Ratios," *FAJ* 58(4)** — autocorrelation-aware Sharpe inference. **White, H. (2000), "A Reality Check
  for Data Snooping," *Econometrica* 68(5)** — the data-snooping correction. Implemented in the shared
  [`quantlab/`](../../../quantlab/) engine; see [`METHODOLOGY.md`](../../../METHODOLOGY.md).
