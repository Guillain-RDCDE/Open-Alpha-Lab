# References & literature map -- Study 222 (Altseason-Rotation)

## The claim under test

- **The "alt-season rotation" recipe.** A widely-shared belief in crypto communities: when
  Bitcoin's dominance (its share of total crypto market capitalisation) falls, capital
  rotates from BTC into altcoins, triggering "alt season" -- a period of outsized alt
  outperformance. The actionable version: use falling BTC dominance over the past N days
  as a trigger to go long an equal-weighted alt basket and short BTC; exit when dominance
  recovers. We test whether this timing rule produces risk-adjusted returns that exceed
  buy-and-hold benchmarks and clear the |t| >= 2 inference bar. The claim is a fixture of
  crypto Twitter/Reddit, CoinGecko dominance charts, and crypto newsletter commentary.
  See also: Study 134 (Bitcoin-Dominance) which tests the underlying regression signal;
  Study 222 tests the explicit strategy implementation.

## Why the steelman is almost coherent

- **Portfolio rotation theory.** Corbet, Meegan, Larkin, Lucey & Yarovaya (2018),
  *Exploring the Dynamic Relationships Between Cryptocurrencies*, Economics Letters --
  document time-varying correlation structures consistent with sequential rotation from BTC
  into alts during risk-on episodes.
- **Sentiment cascade.** Ante (2020), *A place for cryptocurrencies in the portfolios of
  institutional investors* (Hamburg Business School, doctoral thesis) -- BTC acts as a
  "gateway" asset; dominance declines coincide with retail-driven speculative phases that
  lift smaller coins disproportionately. If this is a repeatable mechanism, a dominance-
  timed rotation strategy should capture the spread.
- **Momentum and regime timing.** Jegadeesh & Titman (1993), *Returns to Buying Winners and
  Selling Losers: Implications for Stock Market Efficiency*, Journal of Finance -- the
  general momentum literature supports the idea that assets with recent relative strength
  continue to outperform in the short run. If alts have been gaining vs BTC (dominance
  falling), momentum predicts continued alt outperformance. However, crypto momentum is
  documented to be very short-lived (days, not weeks), undermining the weekly/monthly
  rotation horizon.
- **Altcoin beta structure.** Bouri, Molnar, Azzi, Roubaud & Hagfors (2017), *On the hedge
  and safe haven properties of Bitcoin*, Finance Research Letters -- alts exhibit high
  conditional beta to BTC in up-markets; the differential return is what "alt season"
  trading targets. Empirically, the beta is time-varying and regime-dependent.

## Why the strategy fails in practice

- **Short history: one cycle.** The overlapping panel covers 2020-04-10 to 2026-06-16
  (~6 years), dominated by the 2021 bull run and 2022 crash. With only one crypto cycle,
  there is insufficient power to distinguish a genuine timing rule from a pattern specific
  to that regime. The 2021 alt season drives most of the gross positive returns.
- **Survivorship bias.** The alt basket uses the six largest alts as of 2024 (ETH, XRP,
  ADA, SOL, BNB, DOGE). This ex-post survival selection inflates historical alt returns;
  alts that failed during this period (LUNA, FTT, UST, GALA, etc.) are not included.
  A survivorship-free test would require a commercial historical database (Kaiko, Messari).
- **Timing does not beat passive.** The rotation strategy's net Sharpe (0.45 at 40 bps)
  is below BTC buy-and-hold (0.53) and equal-weight alt buy-and-hold (0.51). This is the
  decisive failure: even in the best historical scenario, turning off the rotation signal
  and just holding the assets would have done better. Liu & Tsyvinski (2021),
  *Risks and Returns of Cryptocurrency*, Review of Financial Studies -- document that
  simple crypto momentum and carry are weak predictors of future crypto returns once
  proper multiple-testing adjustments are applied.
- **Execution costs.** Altcoin perpetuals on major exchanges have fees of 0.04-0.1% per
  side plus bid-ask spreads of 0.05-0.5%. For mid/small-cap alts the costs are higher.
  Our 40 bps round-trip is already optimistic for the full basket. At 80 bps the strategy
  Sharpe degrades to 0.27 -- well below any reasonable hurdle.
- **Dominance proxy limitations.** BTC dominance is proxied by price x fixed supply, not
  live circulating supply. The proxy tracks CoinGecko's reported dominance directionally
  but differs in level; it is a valid signal carrier but not a perfect one. CoinGecko,
  *Global Charts -- BTC Dominance*, https://www.coingecko.com/en/global-charts.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, Econometrica --
  implemented in `strategy.performance_summary`.
- **Long/short PnL simulation.** Position logic: 0/1 (flat vs rotation on), 1-day lag,
  cost deducted on each position change. Dollar-neutral: long alt EW, short BTC, equal
  notional. Standard approach used by the desk for strategy-level PnL studies.
- **Reproducibility stamp.** Content fingerprint on the raw panel (SHA-1 of close matrix);
  as-of date stamped in `docs/results.md`.

## Data sources used here

- **Yahoo Finance daily bars** (via `yfinance`): BTC-USD, ETH-USD, XRP-USD, ADA-USD,
  SOL-USD, BNB-USD, DOGE-USD. Daily OHLCV from each asset's listing date through
  2026-06-16. Effective panel start: 2020-04-10 (SOL listing date). Forward-filled up
  to 3 days for weekend/holiday gaps; rows with any remaining NaN dropped.
  See: https://finance.yahoo.com/ for data terms.

## Related desk studies

- **[Study 134 -- Bitcoin-Dominance](../../134-bitcoin-dominance/)**: the regression
  companion -- does falling BTC dominance *predict* alt-minus-BTC spread returns?
  Same conclusion: direction right, signal weak (t = -1.88), does not clear bar.
  Study 222 takes that weak signal and asks: even if you built a strategy around it,
  would it outperform buy-and-hold? The answer is no.
- **[Study 117 -- Pi-Cycle-Top](../../117-pi-cycle-top/)**: another crypto timing signal
  with sparse data. The shared theme: crypto market regimes are non-stationary and
  dominated by a handful of bull/bear cycles, making timing strategies structurally weak.
- **[Study 210 -- Crypto-Trend](../../210-crypto-trend/)**: crypto trend-following via
  momentum signals on BTC and ETH. A related study in the crypto family; trend-following
  on crypto assets shows more persistence than rotation signals.
- **[Study 209 -- ETH-BTC-Ratio](../../209-eth-btc-ratio/)**: the ETH/BTC relative-
  value ratio as a signal -- another rotation-family study in the crypto space.
