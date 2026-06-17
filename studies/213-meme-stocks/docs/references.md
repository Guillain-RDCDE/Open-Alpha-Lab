# References & literature map — Study 213 (Meme Stocks)

## The claim under test

The **meme-stock mania** of January 2021 (and its sequels): a coordinated
retail crowd on r/WallStreetBets (WSB) drove enormous short squeezes in a
handful of stocks — principally **GameStop (GME)** but also AMC, Koss (KOSS),
BlackBerry (BB), Bed Bath & Beyond (BBBY), and Nokia (NOK). The popular claim
is that a savvy retail investor could have *ridden* the mania for outsized
gains. The counter-claim, tested here, is that most retail participants who
bought the widely-publicised squeeze were exit liquidity for earlier holders.

## The WSB / meme-stock episode — primary accounts and data

- Anand, Amber & Pathak, Jiajun (2022). *"Anatomy of a meme stock short
  squeeze."* Available at SSRN: <https://ssrn.com/abstract=3782195>. Documents
  the short-interest dynamics and the role of retail order flow in GME Jan-2021.
- Hasso, Tim, Müller, Daniel, Pelster, Matthias & Warkulat, Sonja (2022).
  *"Who participated in the GameStop frenzy? Evidence from brokerage accounts."*
  Finance Research Letters, 45, 102140.
  Retail buyers entered *after* the peak squeeze day; most held losing
  positions post-squeeze.
- Eaton, Gregory W., Green, T. Clifton, Roseman, Brian S. & Wu, Yanbin (2022).
  *"Retail Trader Sophistication and Stock Market Quality: Evidence from
  Brokerage Outages."* Journal of Financial Economics, 146(2), 502-528.
  Retail flow in meme stocks showed persistent buying-into-momentum, consistent
  with exit-liquidity dynamics.

## Why meme-stock mania is plausible as a "signal"

- **Short-squeeze mechanics are real.** When short interest is very high
  (GME was ~140% of float short in Jan-2021), a forced buy-in cascade can
  compress the float and spike price non-linearly. This is documented in
  classic squeeze literature (D'Avolio, 2002; *The Market for Borrowing Stock*,
  JFE).
- **Retail coordination via social media can move prices.** Cookson, J.A.,
  Engelberg, J.E. & Mullins, W. (2023). *"Echo Chambers."* Review of Financial
  Studies, 36(2), 450–500. Social-media sentiment correlates with subsequent
  trading in targeted stocks.

## Why it fails as a replicable strategy

- **Timing is everything and unknowable ex ante.** The profitable window in GME
  was a few days in January 2021. A strategy that buys *after* the mania peaked
  (our Strategy B: entry 2021-01-28) lost 45% vs +93% for SPY through 2025.
- **Basket construction introduces survivorship bias.** We know in *hindsight*
  which names became meme stocks. A practitioner in, say, August 2020 had no
  basis for identifying GME as the coming squeeze candidate. The momentum
  signals in Strategy C that entered GME and KOSS in Aug-Sep 2020 are
  look-ahead-contaminated.
- **Single-name concentration.** One name (GME) drove nearly all of the
  basket's positive performance in Strategy A; four of six names lost 40–92%.
  Equal-weighting an ex-post identified basket is not an investable recipe.
- **Extreme volatility eviscerates risk-adjusted returns.** The basket's
  100%+/yr annualised vol vs 17% for SPY means the Sharpe ratio (0.57 vs 0.89)
  is deeply inferior despite a higher raw CAGR in Strategy A.

## Survivorship and delisting

- **Bed Bath & Beyond (BBBY)** filed for Chapter 11 bankruptcy 23 April 2023
  and was delisted. A hold-to-death backtest absorbs the near-total loss; any
  study that stops the BBBY position at the last traded price before the
  bankruptcy filing inflates the basket return.
- **AMC** conducted a reverse split (AMC:APE conversion, Aug-Sep 2023) — the
  adjusted price tape (yfinance `auto_adjust=True`) correctly accounts for this.

## Method lineage

- **HAC t-statistic:** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*,
  Econometrica 55(3), 703-708. Applied here to a sample of n=6 trade returns —
  not enough observations for inference; noted explicitly in the results.
- **Synthetic panel control:** a deterministic GBM panel where a planted
  ``premium`` parameter adds a genuine basket return edge (harness must detect
  it) or is set to zero (harness must not detect an edge). Calibrated to
  meme-stock-realistic volatility (150% annualised).

## Data sources

- **Meme basket prices:** `yfinance` (auto-adjusted, total-return), daily closes
  for GME, AMC, BB, BBBY, KOSS, NOK, cached to `_cache/` under the study root.
- **Benchmark:** SPY (SPDR S&P 500 ETF Trust), total-return (`auto_adjust=True`).

## Related desk studies

- [Study 91 — Death-Cross](../91-death-cross/) — another momentum-timing rule
  judged against a genuine positive/negative control protocol.
- [Study 103 — Turtle](../../103-turtle/) — systematic trend-following, a
  disciplined version of the momentum-entry idea.
- [Study 210 — Crypto-Trend](../../210-crypto-trend/) — the same mania-and-crash
  dynamics in a different asset class with similar look-ahead contamination risks.
