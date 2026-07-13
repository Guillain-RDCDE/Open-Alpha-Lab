# References & literature map -- Study 764 (SOPR)

## The folk claim ("SOPR times capitulation and greed")

- **Shirakashi, R. (2019).** *SOPR -- Spent Output Profit Ratio.* The original
  practitioner formulation (introduced via the "unspent" / Bitcoin-magazine
  write-up and the Renato Shirakashi Twitter posts) that defined SOPR = value of
  spent outputs at the moment they move / value at the moment they were created
  -- i.e. the aggregate realised profit multiple of the coins changing hands.
  SOPR **> 1** = coins move in profit; SOPR **< 1** = coins move at a loss;
  SOPR = 1 is the profit/loss boundary.

- **Glassnode -- "Spent Output Profit Ratio (SOPR)" & "Adjusted SOPR (aSOPR)"
  (2019-2020).** The dashboard/education pieces that popularised SOPR as a
  market-timing gauge. The widely repeated chart-lore: **in bull markets SOPR
  bounces off 1 as support** (holders refuse to sell at a loss), **in bear
  markets it caps at 1 as resistance**; a decisive break of 1 is read as a
  regime change. aSOPR excludes outputs younger than 1 hour to filter noise.
  This is the framing this study steelmans and the source of the labelled proxy
  series in [`sopr/data.py`](../sopr/data.py).

- **Edwards, D. (Glassnode Academy) -- "SOPR" indicator guide.** The
  practitioner "how to trade it" material: long/accumulate when SOPR recovers
  above 1, de-risk when it breaks below 1 -- the literal ">1 / <1" rule this
  study backtests.

## What the evidence actually supports

- **In-sample threshold lore.** The "1 is support in bulls / resistance in
  bears" reading is stated *after* the fact -- it is a description of which
  regime you were already in, not a leading rule. On our 2014-2026 monthly
  sample the ">1 / <1" rule is out of the market 38% of the time and *trails*
  buy-and-hold at every threshold.

- **Reverse / mechanical causality.** SOPR is computed from realised
  profit/loss against *past* acquisition prices, so a month where price fell
  mechanically pushes SOPR below 1 (coins now sold at a loss). SOPR is therefore
  close to a re-labelling of recent price direction -- which is why it dies in a
  price-momentum horse race (HAC *t* drops from +1.32 to +0.18).

- **Liu, Y. & Tsyvinski, A. (2021).** *Risks and Returns of Cryptocurrency.*
  Review of Financial Studies, 34(6), 2689--2727. The most-cited academic study
  of crypto return predictability: BTC returns load on momentum and
  investor-attention proxies, not on on-chain profit/valuation ratios. No robust
  out-of-sample edge for level-based on-chain gauges.

- **Bianchi, D., Babiak, M. & Dickerson, A. (2022).** *Trading volume and
  liquidity provision in cryptocurrency markets.* Documents how thin, reflexive
  crypto microstructure makes single-asset on-chain timing signals fragile and
  regime-dependent.

## Methodological cautions

- **Granger, C. W. J. & Newbold, P. (1974).** *Spurious regressions in
  econometrics.* Two trending / mechanically-linked series show high R^2 and
  significant slopes in levels with no genuine relationship -- which is why this
  study works in *stretch* (log SOPR vs 1) against *forward returns*, not levels.

- **Newey, W. K. & West, K. D. (1987).** HAC standard errors, used for every
  t-stat here.

- **Placebo / permutation testing.** The time-shuffle placebo (permute SOPR,
  re-run the rule) is the White (2000)-style check that the rule's result is not
  what a random schedule with the same in-market share would produce. Here the
  real rule beats random but still loses to buy-and-hold.

- **Single-survivor bias.** BTC is the one cryptocurrency that survived and
  ~150x'd; SOPR is derived from its own spending; the regime thresholds are
  fitted to its ~four cycles. Any long-biased rule benefits mechanically. Named
  on the Signal axis.

## Related desk studies

- **[Study 293 -- MVRV-Ratio](../../293-mvrv-ratio/)**: the sibling on-chain
  profit/valuation gauge (market cap / realized cap) -- same curated-series +
  BTC-tape pattern, same None/Mirage outcome via the buy-and-hold benchmark.
- **[Study 292 -- Bitcoin-Hashrate](../../292-bitcoin-hashrate/)**: another
  on-chain folk indicator ("price follows hashrate") with the same structure and
  verdict.
- **[Study 117 -- Pi-Cycle-Top](../../117-pi-cycle-top/)**: a Bitcoin
  cycle-timing indicator judged on tiny-n cycle turns, the same reckoning that
  sinks SOPR's "sell band."
