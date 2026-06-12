# References & literature map — Study 82 (Witching-Hour)

## The claim under test

- **The folk hypothesis.** Every third Friday of March, June, September, and December,
  equity index options, individual stock options, and stock index futures all expire on
  the same day — *triple* witching (before 2002, when single-stock futures were
  introduced, technically quadruple on some contracts).  The claim has two parts: (1) a
  *mechanical* elevated volume and volatility effect driven by delta-hedging, roll
  activity, and institutional rebalancing, and (2) a *directional* return effect that
  can be traded.  The steelman: "the mechanical flows are so large and predictable that
  they systematically move price, and you can position around them."

## The mechanical effect — established in the academic literature

- **Stoll & Whaley (1987)**, *Program Trading and Expiration-Day Effects* (Financial
  Analysts Journal) — the original witching-hour study: documents elevated volume and
  price reversals at expiry close.  The effect was dramatic in the 1980s when cash-index
  arbitrage dominated expiry activity.
- **Stoll & Whaley (1991)**, *Expiration-Day Effects: What Has Changed?* (Financial
  Analysts Journal) — update documenting how regulatory changes (moving settlement to
  the open, not the close) damped the 1980s end-of-day spike.  The volume effect
  persisted; the return spike attenuated.
- **Corredor, Lechón & Santamaría (2001)**, *Option Expiration Effects in Small Markets:
  The Spanish Stock Exchange* (Journal of Futures Markets) — documents the same expiry
  volume and return-reversal pattern in a smaller market, supporting the hypothesis that
  the effect is driven by mechanical hedging rather than US-market-specific structure.

## Why volume is real but range is not

- **Hedging and roll volume without price impact.** The large witching-day volume is
  driven by *predictable* institutional activity (delta-hedging, closing out positions,
  rolling futures), not by information asymmetry.  Kyle (1985), *Continuous Auctions and
  Insider Trading* (Econometrica), predicts that informed-order flow moves price; a
  surge in *liquidity-motivated* roll trades need not.  This is consistent with our
  finding: volume +24% on witching day, range −1% (not elevated).
- **The 'witching hour' was the final 45 minutes.** The original Stoll-Whaley (1987)
  effect was a sharp spike at the daily close (the expiry settlement print).  After the
  NYSE moved settlement to the *opening* print in 1987–1988, the end-of-day price spike
  disappeared; what remained was elevated all-day volume, not a price-dislocating close.

## The return effect

- **Expiry-day return anomaly — mixed evidence.** Alkebäck & Hagelin (2004), *Expiration
  Day Effects of Index Futures and Options: Evidence from a Market with a Long Settlement
  Period* (Applied Financial Economics) — documents a negative return on expiry day in
  the Swedish market, consistent with our finding of −14 bps vs +5 bps baseline.  The
  effect is attributed to delta-unwinding pressure and is short-lived.
- **Pinning and gravitational pull.** Ni, Pearson & Poteshman (2005), *Stock Price
  Clustering on Option Expiration Dates* (Journal of Financial Economics) — documents
  that stock prices cluster near option strike prices at expiry, consistent with
  market-maker pinning activity.  For an index ETF like SPY, cross-sectional pinning
  forces partially net out.

## Method lineage

- **HAC / Newey-West t-stat.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica) —
  [`strategy.compare_witching_vs_baseline`](../witching_hour/strategy.py).
- **Year-demeaned volume.** The log-volume trend in SPY (declining over the ETF's
  history as AUM grows relative to daily turnover) requires within-year demeaning for an
  unbiased seasonal comparison.  The technique follows standard panel-data practice
  (within-group demeaning); see Wooldridge, *Econometric Analysis of Panel Data* (2010).
- **Calendar-only signals.** The witching date is a pure function of the exchange
  calendar — known before the month begins — so no execution lag is required.  This
  follows the same discipline as Study 42 (Last-Call, turn-of-month) and Study 48
  (Groundhog, January effect).

## Related desk studies

- **[Study 42 — Last-Call](../../42-last-call/)**: the turn-of-month premium —
  the same "mechanical calendar drives returns" family, same daily-bar testing protocol.
- **[Study 48 — Groundhog](../../48-groundhog/)**: the January effect —
  another calendar anomaly tested with the same HAC inference engine.
- **[Study 13 — Crimson-Hour](../../13-crimson-hour/)**: intraday time-of-day
  return patterns — the sister study to the witching *hour* (intraday) effect.
- **[Study 67 — Fed-Drift](../../67-fed-drift/)**: the FOMC drift — another event-window
  study where a predictable institutional event (Fed announcement) is associated with
  systematic return behaviour.

## Data sources

- **Yahoo! Finance daily bars** (via `yfinance`), adjusted for splits and dividends
  (auto-adjust=True), SPY from 1993-01-29 (ETF inception).  As-of 2026-06-12, n=8,400
  trading days, 132 witching events in the tape.  Content fingerprint pinned in
  [`docs/results.md`](results.md).
