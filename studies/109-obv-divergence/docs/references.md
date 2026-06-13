# References & literature map — Study 109 (OBV-Divergence)

## The claim under test

- **Granville (1963), the original source.** Joseph Granville, *A Strategy of Daily Stock Market
  Timing for Maximum Profit* (Prentice-Hall, 1963): *"Volume is the steam that makes the choo-choo
  go — and I say that volume precedes price."* Granville's On-Balance Volume (OBV) accumulates
  signed daily volume (add when close is up, subtract when down) as a proxy for buying/selling
  pressure. The hypothesis is that a rising OBV signals accumulation by informed investors
  *before* the price fully reflects their conviction — making OBV a leading rather than confirming
  indicator. We test two operationalizations: (A) OBV above/below its own SMA as a directional
  price signal; (B) OBV-vs-price divergence as a reversal predictor.

## Why the steelman has some empirical grounding

- **Informed-trading and price discovery.** Glosten & Milgrom (1985), *Bid, Ask, and Transaction
  Prices in a Specialist Market with Heterogeneously Informed Traders* (Journal of Financial
  Economics) — informed traders act on private information *through* volume; volume imbalances
  can reflect information flow, which is the theoretical kernel OBV appeals to.
- **Volume as a confirming indicator.** Blume, Easley & O'Hara (1994), *Market Statistics and
  Technical Analysis: The Role of Volume* (Journal of Finance) — volume carries information about
  the *quality* of a price signal, not its direction independently. High volume at highs confirms
  the trend; this is subtly different from volume *preceding* price, and our test distinguishes
  them.
- **Some empirical support in specific contexts.** Yin & Yang (2021), *On the Market Timing of
  Hedging* (Journal of Financial Markets), and several cryptocurrency studies document OBV-related
  volume patterns with short-term predictive value in specific, less-efficient settings. On deep
  and liquid US equities the evidence is much weaker.

## Why it likely fails on liquid US daily data

- **Price discovery is fast in modern markets.** With high-frequency algorithmic market-makers and
  massive HFT volume, any volume-based information advantage available to retail OBV users is
  almost certainly already incorporated in the price within seconds, not days.
- **Daily bar OBV is noisy.** OBV accumulates volume over the *entire* session without regard to
  *intrabar* direction; a day with 90% of volume in the last hour is treated identically to one
  with uniform volume distribution. Intraday volume-weighted measures (e.g. VWAP direction,
  order-flow imbalance) are theoretically better — and even those typically predict at minutes,
  not days.
- **The unconditional drift.** In a rising market the buy-and-hold 5-day drift (+22.88 bps,
  t = +3.65 on SPY 2010–2026) swamps any timing signal. A bearish OBV divergence signal fights
  this drift and thereby underperforms not only a coin but an agnostic hold.
- **Data-snooping pressure.** OBV is one of thousands of volume-derived technical indicators;
  any ex-post study finding it profitable is subject to multiple-comparison bias. This study
  corrects for 12 tests simultaneously; under Bonferroni, the threshold is |t| ≥ 3.0 — not
  a single test clears even the uncorrected |t| ≥ 2.0.

## Prior academic evidence on OBV

- **Granville self-assessment vs replication.** Colby & Meyers (1988), *The Encyclopedia of
  Technical Market Indicators* (McGraw-Hill), tested OBV on index data and found mixed at best
  results net of costs.
- **Negative reviews in systematic tests.** Park & Irwin (2007), *What Do We Know About the
  Profitability of Technical Analysis?* (Journal of Economic Surveys) — the meta-review of 95
  studies covering MA rules, oscillators, and volume indicators concludes that once data-snooping
  bias and transaction costs are properly accounted for, volume-based rules do not reliably earn
  excess returns on liquid markets.
- **Divergence specifically.** Brown & Jennings (1989), *On Technical Analysis* (Review of
  Financial Studies), provide a rational-expectations model where technical signals can carry
  real information — but only in markets with significant adverse selection and slow price
  adjustment. US large-cap equities in 2010–2026 do not fit that description.

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.summarize`](../obv_divergence/strategy.py).
- **Random-direction control.** The same honest benchmark used throughout Open-Alpha-Lab:
  on identical signal dates, replace the signal direction with a fair coin. The claimed
  predictor must beat the coin, not just beat zero — see [`METHODOLOGY.md`](../../../METHODOLOGY.md).
- **Multiple-comparison awareness.** Bonferroni and Romano-Wolf corrections for the 2 signals
  × 6 instruments = 12 tests; the uncorrected inference bar of |t| ≥ 2 is already too loose
  in a multi-test setting.
- **Forward-return discipline.** Signal formed at close *t*; position enters at open *t+1*;
  return measured to close *t+H*. No look-ahead.

## Data sources

- **Yahoo! Finance daily bars** (via `yfinance`), 2010-01-04 to 2026-06-12, six liquid names
  (SPY, QQQ, IWM, AAPL, MSFT, NVDA). Auto-adjusted for splits and dividends. Each run is
  pinned with the as-of date and a content fingerprint per instrument — see
  [`docs/results.md`](results.md).

## Related desk studies

- **[Study 21 — Fools-Gold](../../21-fools-gold/)**: the daily SMA(50/200) golden cross —
  a price-only trend indicator, same asset class, same verdict: no edge over a coin.
- **[Study 72 — Loaded-Dice](../../72-loaded-dice/)**: the 5-minute SMA(5/10) crossover scalp —
  the pure price-momentum family this study compares against. Coin. Same conclusion.
- **[Study 85 — Dr-Copper](../../85-dr-copper/)**: cross-asset ratio as price predictor —
  the desk's methodology for testing whether one series (here volume/OBV) leads another.
- **[Study 86 — Tail-Radar](../../86-tail-radar/)**: VIX / VVIX as vol-index signals — the
  closest desk analogue to "a derived indicator predicts equity direction."
