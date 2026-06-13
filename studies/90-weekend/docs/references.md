# References & literature map — Study 90 (Weekend / day-of-week effect)

## The claim under test

The "weekend effect" (a.k.a. the "Monday effect") and its companion "turnaround Tuesday":
the strong, sold-at-full-strength version is that **Mondays are systematically negative** —
the market drifts down over the weekend and opens weak — and that **Tuesday rebounds**, so
a simple calendar rule ("avoid Monday, buy Tuesday") **beats buy-and-hold**.

- Popular framing, e.g. Investopedia, *"Weekend Effect"* and *"Monday Effect"*:
  <https://www.investopedia.com/terms/w/weekendeffect.asp>
- "Turnaround Tuesday" is a recurring fixture of trading-desk and financial-media folklore.

## Why the steelman is almost coherent

- **The weekend effect is one of the best-documented seasonal anomalies in the early
  literature.** Kenneth French, *Stock Returns and the Weekend Effect*, Journal of Financial
  Economics 8 (1980), 55–69, found significantly **negative average Monday returns** on the
  S&P composite over 1953–1977 — the canonical citation. Gibbons & Hess (1981, *Journal of
  Business*) confirmed it across the S&P and individual stocks.
- A plausible micro-structure story exists: settlement timing, the clustering of bad
  corporate news after Friday's close, and dealer inventory effects could in principle push
  weekend-spanning returns down.

## Why it is likely to fail *as stated* ("beats buy-and-hold today")

- **The effect famously decayed — and in places reversed — after it was published.** Once an
  anomaly is in print, it tends to be arbitraged away. Studies through the 1990s–2000s
  (e.g. Kamara 1997; Mehdian & Perry, *The Reversal of the Monday Effect*, Journal of
  Business Finance & Accounting, 2001; Olson, Mossman & Chou, 2015) document the weakening
  or disappearance of the Monday effect in modern U.S. data.
- **Calendar-of-the-week rules are a selection minefield.** With five weekdays you are
  running five tests; the "best" weekday is significant by construction unless you correct
  for the multiplicity. A single contrast at *t* just above 2 is not the same as an effect
  that survives a data-snooping correction.
- **Even a real per-day tilt rarely survives as a tradable rule.** Sitting in cash four days
  out of five (the "buy Tuesday" rule) throws away the equity risk premium on the other days
  — the lost beta dwarfs any weekday tilt, before a single basis point of cost.

## Method lineage

- **Newey–West HAC standard errors** for the mean of an autocorrelated return series, and for
  the difference of means across weekday groups and sub-periods: Newey & West (1987), *A
  Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent
  Covariance Matrix*, Econometrica 55, 703–708. Daily index returns are mildly autocorrelated,
  so a plain *t* overstates significance.
- **Test of the *difference* across a pre-registered split** (pre-2000 vs post-2000) rather
  than two separately-reported sub-period means — the desk rule that a "decayed since…" claim
  must carry a test of the change, on a justified, not snooped, split.

## Data sources used

- **SPY**, daily, **total-return adjusted** (dividends folded in) via `quantlab.data`
  (Yahoo Finance), cached to parquet under `_cache/`. Total return is the fair series for a
  per-weekday mean read and for the buy-and-hold benchmark a part-time-in-cash rule competes
  against. Cash is assumed to earn **0%** — a stated, conservative choice. Note the SPY tape
  starts **1993-01-29** (the ETF's inception), so it is a *post-publication* sample of
  French's 1980 effect — by construction it tests the effect in the era when it is reported
  to have faded, not the 1953–1977 window where it was discovered. A longer price-only
  `^GSPC` series could extend the sample but would *not* be total return (no dividends) and
  is not used for the headline numbers.

## Related desk studies

- [Study 91 — Death-Cross](../../91-death-cross/) — the same engine and "calendar/known-signal,
  no-lag" and matched-control discipline applied to a moving-average timer.
- [Study 87 — Center-Line](../../87-center-line/) — the "beats a coin?" control pattern.
