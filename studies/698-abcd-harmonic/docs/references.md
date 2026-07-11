# References & literature map — Study 698 (ABCD-Harmonic)

## The claim under test

- **The AB=CD harmonic.** The simplest member of the harmonic-pattern family taught
  in retail technical-analysis circles: identify three alternating swing pivots A, B,
  C such that the retracement leg BC pulls back **61.8%** of the impulse leg AB, then
  project a fourth point **D** such that the extension leg CD is **equal in length to
  AB** ("AB=CD"). Chartists expect price to *reverse* the instant it reaches D — the
  pattern is said to be "complete" and the market has, in their language, exhausted
  the move. Scott Carney popularized the modern harmonic-pattern taxonomy (Gartley,
  Bat, Butterfly, Crab — all built around this same AB=CD skeleton with an added X
  point and extra Fibonacci confluence) in *Harmonic Trading, Volumes 1-3*
  (2004-2010); H.M. Gartley's *Profits in the Stock Market* (1935) is the pattern's
  much older, pre-Fibonacci ancestor (Gartley's original book did not even specify
  the 0.618 ratio — that was bolted on later by Larry Pesavento).
- **The mechanism claimed.** No causal story beyond "traders watch these ratios and
  place orders there, so the pattern becomes self-fulfilling" — the same
  order-clustering logic examined (and found wanting for Fibonacci specifically) by
  sibling study [77-golden-mean](../../77-golden-mean/).

## What we measure, and the honesty rails

- **Confirmed pivots only, no look-ahead.** A percentage-threshold zigzag records a
  swing pivot only at its *confirmation* bar — the session where price has already
  reversed far enough to lock it in — never at the (earlier) extreme itself. The
  projected D level is fully computable, and the D-touch scan starts, the moment C
  confirms: a real trader could have placed this exact order in real time.
- **The placebo is the whole test.** Any level-based rule inherits the base rate of
  "swings that have already retraced tend to keep moving" — a generic mean-reversion
  / momentum artefact that has nothing to do with 0.618 or "AB=CD" specifically. The
  placebo arm reruns the *identical* pivot-detection and touch-scanning pipeline with
  each candidate's retrace/extension targets replaced by a deterministic, seeded,
  off-Fibonacci draw. Only a Fibonacci arm that *beats* this placebo is evidence that
  these particular ratios carry information the placebo's arbitrary ratios don't.
- **HAC / Welch statistics.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica)
  for the within-arm mean test (ABCD events on the same tape can cluster in time);
  Welch's *t* for the arm-vs-arm comparison. Wilson (1927) interval on the hit rate.
- **One documented execution convention**, identical in both arms: enter the fade at
  the touch bar's own close (the touch is observed intrabar, via that bar's high-low
  range, then executed at the same session's close) — the same convention sibling
  [77-golden-mean](../../77-golden-mean/) uses for its level touches.
- **Costs one-way × NAV per leg**, both legs charged (2× per round trip); a
  faithful-engine synthetic control (tunable mean-reversion knob) proves the
  detection + inference pipeline is unbiased on a null and recovers a planted effect
  — never cited in support of the real-tape stamp.

## Why the evidence is weak even before our test

- Carney's own books present harmonic patterns as requiring a **confluence zone**
  (multiple overlapping Fibonacci projections from several legs, plus the "PRZ" —
  potential reversal zone) rather than a bare two-leg AB=CD — this study
  deliberately tests the *simplest, most literal* reading of the claim ("BC retraces
  0.618, CD=AB") because that is the version stated in the brief and the version
  every retail charting tutorial teaches first; a looser, multi-confluence definition
  would only add more researcher degrees of freedom, not less.
- Academic tests of harmonic/Gartley-style patterns are scarce and largely negative:
  Boasiako, Iyke & Krige (2023) and older technical-pattern surveys in the Sullivan,
  Timmermann & White (1999) tradition — *Data-Snooping, Technical Trading Rule
  Performance, and the Bootstrap* (Journal of Finance) — find that once a proper
  data-snooping / selection correction is applied, most rule-based chart-pattern
  "edges" published without a randomized control do not survive. This study supplies
  exactly that missing control.

## Data sources

- **Yahoo! Finance daily bars** (via `yfinance`), total-return adjusted, six liquid
  instruments (SPY, QQQ, AAPL, MSFT, TSLA, NVDA) — the identical basket used by
  [77-golden-mean](../../77-golden-mean/). Daily history stretches back to
  2001 for four of the six names (TSLA from its 2010-06-29 IPO). Every headline
  number is pinned with a content fingerprint and as-of date; see
  [`docs/results.md`](results.md), reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- **[468-gartley-harmonic](../../468-gartley-harmonic/)** — the four-leg **XABCD**
  Gartley pattern (an X point, AB retracing XA by 0.618, BC retracing AB, CD
  projecting off Fibonacci extensions of both AB *and* XA, plus a confluence PRZ).
  This study has no X point and only the bare two-leg AB=CD skeleton — the Gartley
  is the *elaborated* version this study deliberately does not test.
- **[699-butterfly-harmonic](../../699-butterfly-harmonic/)** — the Butterfly
  variant (XA extended 1.27-1.618, not equal-legged) — a different, more extreme
  extension ratio on the CD leg, again anchored on an X point this study omits.
- **[700-bat-harmonic](../../700-bat-harmonic/)** — the Bat variant (a shallower
  0.382-0.50 B retracement and a deeper 1.618-2.618 CD extension) — again XABCD,
  again a different ratio set.
- **[77-golden-mean](../../77-golden-mean/)** — tests plain Fibonacci
  *retracement* levels (38.2/50/61.8%) and round numbers as static
  support/resistance on a single swing, with the same placebo-control design this
  study borrows. It does **not** test a multi-pivot harmonic pattern or a projected
  (not merely retraced) target — this study's D point, built from three confirmed
  pivots, is a distinct object. Both studies independently reach the same verdict
  on the same six tapes: Fibonacci ratios show no specificity over a randomized
  control.

None of the siblings test the plain **AB=CD** two-leg reversal projection — this
study's own, narrowest reading of the harmonic-pattern claim.
