# References & literature map — Study 609 (VIX Weekend Arithmetic)

## The claim under test

- **The folklore.** Vol-desk lore and retail commentary hold that the VIX has a "guaranteed"
  day-of-week pattern hiding in plain sight — a weekend seesaw baked into the index's own
  formula. The retellings disagree on the sign (the version we were handed says "drifts up into
  Friday, drops on Monday"; the arithmetic says the opposite). We test the pattern, its
  direction, its magnitude against the day-count model, and whether anything tradable survives.
- **The formula that creates it.** CBOE, *Volatility Index Methodology: Cboe Volatility Index*
  (the VIX white paper) — <https://www.cboe.com/micro/vix/vixwhite.pdf>. The VIX squares to the
  expected S&P-500 variance over the next **30 calendar days**, annualized on a
  calendar-time (minutes-to-expiration) clock. Because variance is *realized* mostly during
  trading hours, a 30-calendar-day window quoted on Friday (20 trading + 10 weekend days)
  carries less expected variance per calendar day than the same window quoted on Monday
  (22 trading + 8 weekend days) — the quoted index must dip into the weekend and pop after it,
  to the extent option markets discount weekend variance.
- **Whaley, R. (2009), "Understanding the VIX",** *Journal of Portfolio Management* 35(3) —
  the standard plain-language description of what the index is and is not.

## Trading time vs calendar time (why weekends carry less variance)

- **Fama, E. (1965), "The Behavior of Stock-Market Prices",** *Journal of Business* 38 — first
  systematic evidence that variance accrues in *trading* time more than calendar time.
- **French, K. (1980), "Stock Returns and the Weekend Effect",** *Journal of Financial
  Economics* 8 — the original equity weekend-effect paper (and the sibling claim tested in
  [90-weekend](../../90-weekend/)).
- **French, K. & Roll, R. (1986), "Stock Return Variances: The Arrival of Information and the
  Reaction of Traders",** *Journal of Financial Economics* 17 — the classic measurement:
  variance over the Friday-close→Monday-close span is barely larger than over a single trading
  day, i.e. a weekend day realizes only a small fraction of a trading day's variance. Our
  fitted *f* ≈ 0.6 says option markets discount weekends far **less** than realized variance
  would justify.
- **Jones, C. & Shemesh, J. (2018), "Option Mispricing around Nontrading Periods",** *Journal
  of Finance* 73(2) — option prices do not fully adjust for the lower variance of nontrading
  periods (options are relatively overpriced before weekends/holidays). Directly consistent
  with the tape's partial weekend discount (*f* ≈ 0.6 rather than the realized-variance ~0.1).
- **Muravyev, D. & Ni, X. (2020), "Why Do Option Returns Change Sign from Day to Night?",**
  *Journal of Financial Economics* 136 — the day/night decomposition of option returns; the
  same trading-time-vs-calendar-time tension at daily frequency.

## Why the seesaw cannot be harvested

- **VIX futures price the forward, not the spot.** The final settlement of a VIX future is a
  30-day *forward* implied-vol print; between settlements the future is an expectation of the
  index at expiry, so a deterministic weekend dip in the *spot* index is fully anticipated and
  arbitraged out of the futures curve. See CBOE VX contract specifications —
  <https://www.cboe.com/tradable_products/vix/vix_futures/specifications/>.
- **The ETP inherits the futures plus roll drag.** [375-vxx-roll-decay](../../375-vxx-roll-decay/)
  measures the contango bleed of a long short-term VIX-futures ETP (the desk's VIXY tape, the
  continuous VXX-equivalent): ≈ −15 %/yr of decay is the *baseline* a weekend-hold strategy
  must beat before the first basis point of the "pop" arrives — and the pop never arrives in
  the ETP at all (our third axis: over-weekend mean −0.32 %, Welch *t* = −0.92).

## Method lineage (the desk's shared engine)

- **Day-count model + implied weekend fraction.** [`data.model_weekday_changes`](../vix_weekend_arithmetic/data.py)
  (predicted Δln VIX by weekday given the weekend variance fraction *f*) and
  [`strategy.implied_weekend_fraction`](../vix_weekend_arithmetic/strategy.py) (deterministic
  least-squares inversion of the five weekday means).
- **HAC inference.** [`strategy.mon_fri_contrast`](../vix_weekend_arithmetic/strategy.py) —
  Welch *t* on the Monday/Friday groups + a Newey-West(10) *t* on the Monday-minus-Friday
  contrast from a weekday-dummy regression (Newey & West 1987, *Econometrica* 55).
- **Label-shuffle placebo.** [`strategy.placebo_spread`](../vix_weekend_arithmetic/strategy.py)
  — 20,000 seeded reshuffles of the Monday/Friday tags (Fisher randomization logic).
- **Deterministic synthetic control.** [`data.synthetic_tape`](../vix_weekend_arithmetic/data.py)
  — log-AR(1) true vol quoted through the day-count arithmetic with a planted *f*; the null
  (*f* = 1) must not manufacture significance, and the estimator must recover a planted *f*.

## Data sources used here

- **yfinance ^VIX** daily closes, 1990-01-02 → 2026-06-30 (9,191 rows), cached under
  `_cache/vix_close.csv`; **yfinance VIXY** auto-adjusted daily closes, 2011-01-04 → 2026-06-30
  (3,894 rows), cached under `_cache/vixy_close.csv`. All headline numbers are pinned in
  [`docs/results.md`](results.md) and reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies (and how this one differs)

- [90-weekend](../../90-weekend/) — the weekend effect in **equity returns** (French 1980);
  this study is about the **VIX index's own day-count formula**, not equity risk premia.
- [375-vxx-roll-decay](../../375-vxx-roll-decay/) — the **futures-carry** bleed of long VIX
  ETPs; here that bleed is the *baseline* the third-axis harvest test must overcome.
- [605-vix-settlement-day](../../605-vix-settlement-day/) — the monthly **settlement-auction**
  footprint in ^VIX; a different, event-driven mechanism on the same index.
- [111-vix-term-structure](../../111-vix-term-structure/) — the contango/backwardation state of
  the futures curve; the forward-pricing reason the weekend seesaw cannot leak into tradables.
