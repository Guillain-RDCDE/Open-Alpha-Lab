# References & literature map — Study 93 (Round-Numbers)

## The claim under test

"Prices are magnetised by round numbers." The folk version, sold at full strength, has two
limbs: **(a)** prices *cluster* at and near round levels (whole dollars, 00-levels, round
index milestones), and **(b)** that magnetism is *tradable* — an approach to a round number
stalls or reverses, so you can **fade the approach and collect the bounce**. The desk tests
the two limbs separately, because the first is well-documented and the second is a far taller
order.

- Popular framing, e.g. Investopedia, *"Round Number Effect"* and the trading-floor lore of
  "support/resistance at round numbers": <https://www.investopedia.com/terms/r/round-number-effect.asp>
- A staple of technical-analysis commentary at every Dow/S&P milestone (the "Dow 40,000"
  headlines).

## Why the steelman is almost coherent — the clustering limb is real

- **Osborne (1962), *Periodic Structure in the Brownian Motion of Stock Prices*, Operations
  Research** — the original observation that transaction prices cluster on preferred
  fractions, not uniformly.
- **Harris (1991), *Stock Price Clustering and Discreteness*, Review of Financial Studies** —
  the definitive treatment: prices cluster on round increments, more so for higher-priced and
  more-volatile stocks, consistent with a negotiation/attraction hypothesis. This is the
  literature that makes limb (a) the expected, *real* result.
- **Niederhoffer (1965), *Clustering of Stock Prices*, Operations Research** — early
  confirmation that round prices are over-represented.
- **Ikenberry & Weston (2008), *Clustering in US Stock Prices after Decimalisation*,
  European Financial Management** — clustering survives the 2001 move to decimal pricing.

## Why limb (b) — "fade it for a bounce" — is likely to fail

- **Donaldson & Kim (1993), *Price Barriers in the Dow Jones Industrial Average*, JFQA** —
  finds support/resistance *patterns* at 100-point Dow multiples, but the evidence that one
  could *trade* them profitably (net of the obvious frictions) is far weaker; the effect is a
  description of where prices pause, not a forecast of where they go next.
- A reversal you can fade has to clear two bars our harness imposes: an
  **autocorrelation-robust *t* over 2** on the realised fade returns, **and** beating a
  **random-level control** that fades arbitrary thresholds at the same frequency. Clustering
  in the *unconditional* price distribution implies neither — a ruler can be lumpy without the
  future being predictable.
- Any tradable version pays the spread twice per round-trip; a sub-spread "edge" is not an edge.

## Method lineage

- **Pearson chi-square goodness-of-fit** for the uniformity of the distance-to-nearest-round
  distribution (`scipy.stats.chisquare`) — the standard discreteness test (Harris 1991 uses
  the analogous frequency comparison on price fractions).
- **Newey–West HAC standard errors** for the mean of an autocorrelated trade-return series:
  Newey & West (1987), *A Simple, Positive Semi-Definite, Heteroskedasticity and
  Autocorrelation Consistent Covariance Matrix*, Econometrica.
- **Random-level control** — a phase-shifted grid of the same spacing isolates whether the
  *round* levels (vs arbitrary levels at the same frequency) carry information. This is the
  desk's standard "beats a coin?" control (cf. the matched random-timing coin in
  [Study 91 — Death-Cross](../../91-death-cross/) and the random-direction control in
  [Study 87 — Center-Line](../../87-center-line/)).

## Data sources used

- **^GSPC** (S&P 500 spot) and **SPY** (S&P 500 ETF), and the single names **AAPL** and
  **MSFT**, daily, via `quantlab.data` (Yahoo Finance), cached to parquet under `_cache/`.
  Adjustment mode is a **decision**: single names are pulled **`raw`** (as-traded nominal
  prices — the object the clustering literature studies), while the spot index and the ETF
  carry no split history of consequence so **`split_only`** is their traded level. The fade
  return is **demeaned by each tape's own drift**, so the tradability test measures reversal,
  not the market's trend.

## Related desk studies

- [Study 91 — Death-Cross](../../91-death-cross/) — the matched-control "beats a coin?" pattern.
- [Study 87 — Center-Line](../../87-center-line/) — the random-direction control pattern.
- [Study 72 — Loaded-Dice](../../72-loaded-dice/) — another "real-looking but unforecastable" teardown.
