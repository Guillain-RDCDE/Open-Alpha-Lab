# References & literature map -- Study 293 (MVRV-Ratio)

## The folk claim ("MVRV times tops and bottoms")

- **Mahmudov, M. & Puell, D. (2018).** *MVRV -- An On-Chain Indicator for
  Bitcoin.* The widely cited practitioner formulation that defined MVRV =
  market cap / realized cap as a *contrarian* valuation gauge: MVRV well above
  ~3.5 marks a euphoric, over-valued top (sell); MVRV below ~1.0 marks
  capitulation / under-valuation (buy). Popularised the "MVRV-Z-Score" cycle
  bands on crypto Twitter and dashboards. No peer review; the bands are
  calibrated to the handful of cycle turns Bitcoin happened to have.

- **Coinmetrics -- "Realized Capitalization" (2018).** Nic Carter & Antoine Le
  Calvez introduced *realized cap* (every coin valued at the price it last moved
  on-chain), the denominator of MVRV. The realized cap is a slow, backward-
  looking moving average of past prices, which is precisely why MVRV and price
  are mechanically linked -- a point this study leans on.

## What the evidence actually supports

- **In-sample band fitting.** MVRV's "sell above 3.5 / buy below 1.0"
  thresholds are chosen *after* seeing the 2013, 2017 and 2021 tops. With only
  ~four full Bitcoin cycles in existence, any two thresholds can be drawn
  through the extremes; the over-heated band fires only 4 times in our 2014-2026
  monthly sample (n = 4), which is anecdote, not a testable edge.

- **Reverse / mechanical causality.** Because realized cap lags price, a spike
  in MVRV is largely *caused by* a price spike, not a leading cause of the next
  move. The contemporaneous co-movement is a shared trend, the textbook
  spurious-regression trap when two slowly-evolving series are compared.

- **Liu, Y. & Tsyvinski, A. (2021).** *Risks and Returns of Cryptocurrency.*
  Review of Financial Studies, 34(6), 2689--2727. The most-cited academic study
  of crypto return predictability: BTC returns are driven by momentum and
  investor-attention proxies, not by on-chain valuation ratios. No robust
  out-of-sample edge for level-based valuation gauges.

- **Bianchi, D., Babiak, M. & Dickerson, A. (2022).** *Trading volume and
  liquidity provision in cryptocurrency markets.* Documents how thin, reflexive
  crypto microstructure makes single-asset "valuation" timing signals fragile
  and regime-dependent.

## Methodological cautions

- **Granger, C. W. J. & Newbold, P. (1974).** *Spurious regressions in
  econometrics.* Two trending (I(1)) series show high R^2 and significant slopes
  in levels even with no relationship -- which is why this study works in
  *stretch* (log MVRV vs its mid-band) and *forward returns*, not levels.

- **Newey, W. K. & West, K. D. (1987).** HAC standard errors, used for every
  t-stat here.

- **Single-survivor bias.** BTC is the one cryptocurrency that survived and
  1000x'd; MVRV is derived from its own price path; the contrarian bands are
  fitted to its ~four cycles. Any long-biased timing rule benefits
  mechanically. Named on the Signal axis.

## Related desk studies

- **[Study 292 -- Bitcoin-Hashrate](../../292-bitcoin-hashrate/)**: the sibling
  on-chain folk indicator ("price follows hashrate") -- same hardcoded-series +
  BTC-tape pattern, same None/Mirage outcome via the buy-and-hold benchmark.
- **[Study 174 -- Bitcoin-Rainbow](../../174-bitcoin-rainbow/)**: another
  curve-fit Bitcoin valuation band.
- **[Study 158 -- Super-Bowl](../../158-super-bowl/)**: the folklore-pattern
  template -- a hardcoded series/event table pinned against real returns, with
  the tiny-n reckoning that sinks it.
- **[Study 223 -- Same-Month-Seasonality](../../223-same-month-seasonality/)**:
  the data-driven reference pattern (synthetic panel + cached real series) this
  study mirrors structurally.
