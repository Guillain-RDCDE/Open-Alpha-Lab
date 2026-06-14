# References & literature map — Study 134 (Bitcoin-Dominance)

## The claim under test

- **The "alt-season" narrative.** A widely-shared belief in crypto communities: when
  Bitcoin's dominance (its share of total crypto market capitalization) falls, capital
  rotates from BTC into altcoins, triggering "alt season" — a period of outsized alt
  outperformance. The actionable form is: *falling BTC dominance over the past N days
  forecasts positive alt-minus-BTC returns over the next week*. We test whether this
  lagged relationship exists and clears the |t| >= 2 inference bar. The claim is a staple
  of crypto Twitter/Reddit, CoinGecko dominance charts, and crypto newsletter commentary.

## Why the steelman is almost coherent

- **Portfolio rotation theory.** A rational basis exists: as BTC matures and its marginal
  buyer becomes a macro/institutional investor rather than a crypto-native speculator, risk
  appetite shocks first move BTC then flow into smaller-cap, higher-beta alts. Corbet,
  Meegan, Larkin, Lucey & Yarovaya (2018), *Exploring the Dynamic Relationships Between
  Cryptocurrencies*, (Economics Letters) document time-varying correlation structures across
  crypto assets consistent with sequential rotation.
- **BTC dominance as a sentiment indicator.** Ante (2020), *A place for cryptocurrencies in
  the portfolios of institutional investors* (doctoral thesis, Hamburg Business School),
  documents that BTC acts as a "gateway" asset; dominance declines often coincide with
  retail-driven speculative phases that lift smaller coins disproportionately.
- **Altcoin beta to BTC.** Bouri, Molnar, Azzi, Roubaud & Hagfors (2017), *On the hedge and
  safe haven properties of Bitcoin: Is it really more than a diversifier?* (Finance Research
  Letters) — alts exhibit high conditional beta to BTC, especially in up-markets; but the
  *differential* return (spread) is what "alt season" trading targets.
- **Crypto market microstructure.** Makarov & Schoar (2020), *Trading and Arbitrage in
  Cryptocurrency Markets* (Journal of Financial Economics), document that crypto markets are
  highly fragmented and that price discovery is dominated by BTC; alt prices lag BTC in
  information diffusion, which *could* create a short-lived predictable spread — but the
  effect is measured in minutes, not days/weeks as the alt-season narrative requires.

## Why the signal is weak in practice

- **Short-history problem.** SOL-USD (the most recent major alt) begins in 2020-04-10 on
  Yahoo Finance, capping the overlapping panel at ~6 years. The sample spans only one full
  crypto cycle (2020 bull, 2022 bear, 2023-24 recovery), insufficient for robust inference
  across regimes.
- **Survivorship bias in the basket.** We use the six largest alts as of 2024 (ETH, XRP,
  ADA, SOL, BNB, DOGE). This is an ex-post survival selection: alts that failed (LUNA, FTT,
  etc.) are not included. Survivorship inflates historical alt returns and may exaggerate the
  dominance-spread relationship. A genuine study would require a survivorship-free historical
  market-cap database (e.g., from Messari or Kaiko), which is not publicly available for free.
- **Dominance proxy limitations.** We use price x fixed supply as a market-cap proxy because
  live circulating supply history for all assets is not available via Yahoo Finance. The proxy
  tracks CoinGecko's reported BTC dominance directionally but differs in level. BTC supply
  changes with mining (block rewards halved in 2020, 2024); alt supplies change with token
  emissions, burns, and unlocks. The supply mismatch is largest for inflationary alts (ADA,
  DOGE) where circulating supply grew materially. See: CoinGecko, *Global Charts — BTC
  Dominance*, https://www.coingecko.com/en/global-charts.
- **Regime dependency.** The alt-season pattern (if any) is most visible in 2020-21 retail-
  driven markets and essentially absent in 2022 (bear market correlation collapse) and 2023
  (BTC-led recovery). A single-regime predictor with a 6-year sample cannot be distinguished
  from a stylised fact of one bull market. See: Chohan (2022), *A History of Gamestop and
  the Rise of Retail Investors*, working paper — for the broader context of retail-driven
  speculative rotations across asset classes.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  implemented in [`strategy.summarize`](../bitcoin_dominance/strategy.py) and
  [`quantlab.analytics.mean_tstat_hac`](../../../quantlab/analytics.py).
- **OLS with HAC SE.** The slope of alt-minus-BTC spread on the dominance change, with
  Newey-West standard error on the score (x * residual) — implemented in
  [`strategy.regression_tstat`](../bitcoin_dominance/strategy.py).
- **Permutation control.** Signal column shuffled, spread column fixed — the standard
  permutation null that breaks timing while preserving marginal distributions.
- **Reproducibility stamp.** Content fingerprint on the raw panel (SHA-1 of close matrix);
  as-of date stamped in [`docs/results.md`](results.md).

## Data sources used here

- **Yahoo Finance daily bars** (via `yfinance`): BTC-USD, ETH-USD, XRP-USD, ADA-USD,
  SOL-USD, BNB-USD, DOGE-USD. Daily OHLCV from each asset's Yahoo listing date through
  2026-06-14. Effective panel start: 2020-04-10 (SOL listing date). The panel is
  forward-filled up to 3 days for weekend/holiday gaps (crypto trades 7 days a week but
  Yahoo timestamps can have gaps); rows with any remaining NaN dropped (pre-listing dates).
  See: https://finance.yahoo.com/ for data terms.

## Related desk studies

- **[Study 83 — Half-Life](../../83-half-life/)**: another small-n crypto study (Bitcoin
  halving events, n=3-4). Same power warning: when events are sparse or history is short,
  HAC t-stats cannot clear the inference bar even when the direction is right.
- **[Study 117 — Pi-Cycle-Top](../../117-pi-cycle-top/)**: Bitcoin cycle-top indicator
  with n=3 signals — the same "the machine works but the sample is too small" finding.
- **[Study 84 — Moon-Math](../../84-moon-math/)**: another crypto calendar/pattern study.
  The shared theme: crypto asset returns are highly non-stationary and regime-dependent,
  making pattern-based forecasts structurally unreliable without a very long history.
- **[Study 63 — Free-Fall](../../63-free-fall/)**: cross-asset correlation dynamics —
  the broader family of "does X predict Y rotation?" studies and their consistent finding
  that coincident correlations do not survive as lagged predictors.
