# References & literature map — Study 766 (Memecoin-Season)

## The claim under test

- **The folklore.** "Memecoin season" (a cousin of "altseason") — the recurring crypto-Twitter/X
  thesis that in euphoric bull phases the dog-coins (Dogecoin, Shiba Inu) rocket 10-1000× while
  Bitcoin merely doubles, and that a nimble momentum rotation ("just buy what's pumping") harvests
  it. Popularised every cycle since the 2021 mania; tracked by retail dashboards such as
  BlockchainCenter's *Altcoin Season Index* (https://www.blockchaincenter.net/altcoin-season-index/).
  The steelman: memecoins are genuinely higher-beta than BTC and *do* scream past it in windows —
  the leap of faith is that a **mechanical rotation** converts that raw volatility into kept money.
- **The economic logic, steelmanned.** If weekly returns exhibited persistence (momentum), then
  ranking on trailing returns and holding the leader would harvest the trend. This is the
  cross-asset analogue of the equity momentum literature applied to the highest-octane corner of
  crypto. We test the most literal, mechanical version and let it fail on its own terms.
- **What we are NOT testing.** A dominance-timed long/short (that is 222-altseason-rotation), a
  broad cross-sectional coin-panel factor (632-crypto-xs-momentum), an Elon-tweet event study
  (291-doge-tweets), or calendar seasonality (133-crypto-seasonality). This study is the narrowest
  version: a **winner-take-all weekly momentum rotation over exactly {BTC, DOGE, SHIB}**, framed as
  the retail "memecoin season" trade, torn down on survivorship and the volatility tax.

## Method & prior literature

- **Momentum, the source idea.** Jegadeesh & Titman (1993), "Returns to Buying Winners and Selling
  Losers", *Journal of Finance* 48(1) — the canonical cross-sectional momentum result our rotation
  imitates. Moskowitz, Ooi & Pedersen (2012), "Time Series Momentum", *JFE* 104(2) — the
  own-asset-trend variant closest to "ride whatever is pumping".
- **Crypto momentum specifically.** Liu, Tsyvinski & Wu (2022), "Common Risk Factors in
  Cryptocurrency", *Journal of Finance* 77(2) — documents a size and (weekly) momentum factor in
  crypto; our sibling 632-crypto-xs-momentum reproduces it and finds it is a **bull-market-only**
  factor. Consistent with this study: memecoin weekly returns carry no exploitable persistence
  once you leave the mania.
- **The volatility tax (why a positive-mean strategy loses).** The geometric-vs-arithmetic gap
  $g \approx \mu - \tfrac12\sigma^2$ (Fernholz & Shay (1982), "Stochastic Portfolio Theory and
  Stock Market Equilibrium", *Journal of Finance* 37(2)). At the rotation's 162%/yr volatility the
  drag term dominates the weekly edge — the central mechanism of this study's Mirage verdict.
- **Momentum crashes.** Daniel & Moskowitz (2016), "Momentum Crashes", *JFE* 122(2); Barroso &
  Santa-Clara (2015), "Momentum Has Its Moments", *JFE* 116(1) — momentum's negative skew and
  crash risk, amplified in memecoins; the desk's 508-momentum-crashes reproduces the equity case.
- **Survivorship bias.** Brown, Goetzmann, Ibbotson & Ross (1992), "Survivorship Bias in
  Performance Studies", *Review of Financial Studies* 5(4) — why restricting the universe to
  ex-post survivors (DOGE, SHIB) inflates every backtested return. Named on the Signal axis here.
- **Data-snooping / placebo lineage.** White (2000), "A Reality Check for Data Snooping",
  *Econometrica* 68(5), and Politis & Romano (1994)'s stationary bootstrap — the spirit behind the
  4,000-seed random-rotation placebo that asks whether the momentum signal beats a coin flip.

## Data sources

- **BTC-USD, DOGE-USD, SHIB-USD** daily closes — yfinance (no key), cached under `_cache/` as one
  CSV per asset, resampled to Friday-close weekly bars. Common window 2021-04-16 → 2026-06-26.
  Price-only == total-return for crypto (no dividends). **Named data quirk:** SHIB's sub-1e-10
  launch price stores as literal `0.0` in yfinance for ~8 months, so the tradable three-asset
  universe only begins April 2021 — a short, mania-dominated sample, stated in the open.
- All headline numbers are pinned in [`docs/results.md`](results.md) and reproduced by
  [`examples/verify.py`](../examples/verify.py) (fingerprint `b75635d0b392`).

## Related desk studies (the dedup map — what this study is NOT)

- [222-altseason-rotation](../222-altseason-rotation/) — a **long-alt / short-BTC** rotation timed
  by a **BTC-dominance** signal over a broad alt basket (ETH/XRP/ADA/SOL/BNB/DOGE); MIRAGE. A
  different signal (dominance regression, not price momentum), a different universe, and long/short.
  This study is long-only price-momentum over the concrete memecoin trio.
- [632-crypto-xs-momentum](../632-crypto-xs-momentum/) — the **Liu-Tsyvinski-Wu cross-sectional
  momentum factor** as a ~44-coin weekly **quintile long/short panel** (Real but Fragile,
  bull-only). This study is the retail **winner-take-all** version on just three assets, framed as
  "memecoin season", and lands NONE/MIRAGE — a narrower, blunter instrument that fails harder.
- [291-doge-tweets](../291-doge-tweets/) — an **event study** of Elon Musk tweets on DOGE abnormal
  returns (Real signal, un-tradable at a one-day lag); no rotation, no SHIB, a news-driven jump.
- [251-crypto-reversal](../251-crypto-reversal/) — short-horizon **reversal** (opposite-sign to
  momentum); [133-crypto-seasonality](../133-crypto-seasonality/) — calendar-month "Uptober"
  effects; [134-bitcoin-dominance](../134-bitcoin-dominance/) — the dominance **signal** alone;
  [210-crypto-trend](../210-crypto-trend/) — a 200-day price-SMA trend rule on BTC only. None
  test a winner-take-all momentum rotation across BTC and the two surviving memecoins.

None of the siblings test the literal retail "memecoin season" trade — a weekly top-momentum
rotation over {BTC, DOGE, SHIB} — as its own object, torn down on survivorship and the volatility
tax. That is this study's axis.
