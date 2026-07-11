# References & literature map — Study 702 (Shark-Harmonic)

## The claim under test

- **The Shark harmonic ("5-0" pattern).** A five-point **X-A-B-C-D** structure in the
  "harmonic pattern" family, but the outlier of the group: **neither leg is a retracement.**
  B **extends** the XA leg **1.13-1.618x**, overshooting *past* point X; C **extends** the AB
  leg a further **1.618-2.24x**, overshooting *past* point A; and the completion point D is a
  **PRICE ZONE at 0.886-1.13x the original XA leg** — the "5-0" band, so named because
  practitioners describe D as sitting near a 50%-of-BC / 0%-of-XA confluence. Every sibling
  pattern's B is a *retracement* of XA (Gartley 0.618, Bat 0.382-0.50, Butterfly 0.786, Crab
  0.382-0.618, Cypher 0.382-0.618); the Shark's B is the only one built from an *extension*.
  Believers read the B/C overshoot as exhaustion and treat the 0.886-1.13 zone as an
  unusually precise "potential reversal zone" (PRZ) — a buy when XA rose (expecting price to
  resume up after exhausting the overshoot), sell when XA fell.
- **The source.** The Shark is generally credited to **Scott M. Carney**, introduced in
  *Harmonic Trading, Volume 3* (2015) as a deliberately "non-standard" completion pattern —
  Carney's own framing is that it does *not* fit the classical Gartley/Bat/Butterfly/Crab
  retracement-based mold, which is exactly the property this study's structural bands encode
  (extension ratios throughout, no fixed AB retracement band at all).
- **The mechanism claimed.** As with every harmonic pattern, no causal story beyond "enough
  traders watch these ratios and place orders at the PRZ, so the confluence becomes
  self-fulfilling" — the same order-clustering logic examined (and found wanting for
  Fibonacci specifically) by siblings [77-golden-mean](../../77-golden-mean/) and
  [468-gartley-harmonic](../../468-gartley-harmonic/).

## What we measure, and the honesty rails

- **Confirmed pivots only, no look-ahead.** A percentage-threshold zigzag records a swing
  pivot only at its *confirmation* bar — the session where price has already reversed far
  enough to lock it in — never at the (earlier) extreme itself. The projected D-zone is fully
  computable, and the zone-touch scan starts, the moment C confirms: a real trader could have
  placed this exact order in real time.
- **A price ZONE, not a point-plus-tolerance.** Every other pattern in this study's family
  (Gartley, Bat, Butterfly, Crab, Cypher) tests D as a single target with a small tolerance
  band; the Shark's own "5-0" literature describes D as a genuine *range* (0.886-1.13 of XA).
  We encode that difference explicitly: a touch is any forward bar whose high-low range
  overlaps `[X + 0.886·(A−X), X + 1.13·(A−X)]`.
- **A random-day base rate, not just a placebo ratio grid.** The pooled and per-ticker Signal
  tests compare the Shark D-zone fade against a base rate built from random entry days on the
  *same* ticker, matched to the *same* empirical mix of bullish/bearish setups (pooled over 20
  seeds) — the specific confound named in the study brief: any directional rule on an
  upward-drifting tape inherits some of the market's own drift, and the base rate neutralizes
  it.
- **Bonferroni correction, stated up front.** The basket is tested seven ways (one pooled
  headline + six per-ticker splits, all at the 5-day horizon). Reporting the best of seven
  without correcting for having looked seven times is exactly the data-snooping trap this desk
  exists to catch; the critical |*t*| is raised from the naive 2.0 to the family-wise-corrected
  threshold (see `docs/results.md`) before any test is allowed to "survive."
- **A second, independent placebo — the third-axis myth-check.** Beyond the base rate, the
  identical pivot-detection and zone-scanning pipeline reruns with each candidate's D-zone
  *re-centered* on a deterministic, seeded location clear of the Shark's own 0.886-1.13 band —
  "does ANY nearby completion zone work, not specifically Carney's 0.886-1.13."
- **HAC / Welch statistics.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica), for the
  within-arm mean test (Shark events on the same tape can cluster in time); Welch's *t* for
  the base-rate and placebo arm-vs-arm comparisons. Wilson (1927) interval on the hit rate.
- **One documented execution convention**, identical across arms: enter the fade at the touch
  bar's own close (the touch is observed intrabar, via that bar's high-low range, then
  executed at the same session's close).
- **Costs one-way x NAV per leg**, both legs charged (2x per round trip); a faithful-engine
  synthetic control (tunable mean-reversion knob) proves the detection + inference pipeline is
  unbiased on a null and recovers a planted effect — never cited in support of the real-tape
  stamp.

## Why the evidence is weak even before our test

- The Shark is Carney's own admitted "non-standard" pattern — introduced *after*, and
  explicitly outside, the classical Gartley grid that founds the rest of the zoo — which is
  itself a tell that the taxonomy keeps growing new members whenever an existing ratio fails
  to explain a chart in hindsight (a textbook multiple-comparisons trap: with five-plus
  patterns and multiple ratio bands each, *some* XABCD-shaped structure will always be nearby).
- Academic tests of harmonic/Gartley-family patterns are scarce and largely negative: the
  broader technical-pattern literature in the Sullivan, Timmermann & White (1999) tradition —
  *Data-Snooping, Technical Trading Rule Performance, and the Bootstrap* (Journal of Finance) —
  finds that once a proper data-snooping / multiple-comparison correction is applied, most
  rule-based chart-pattern "edges" published without a randomized control and without a
  Bonferroni-style penalty do not survive. This study supplies both missing controls
  explicitly.
- The Shark's own "5-0" comparative claim — that a *zone* target is a genuine reversal PRZ
  rather than an arbitrary nearby band — is tested directly here via the third-axis placebo,
  which samples zones from the same neighbourhood of extension ratios the pattern's own
  literature would treat as "close but not the real thing."

## Data sources

- **Yahoo! Finance daily bars** (via `yfinance`), total-return adjusted, six liquid instruments
  (SPY, QQQ, AAPL, MSFT, TSLA, NVDA) — the identical basket used by
  [699-butterfly-harmonic](../../699-butterfly-harmonic/),
  [700-bat-harmonic](../../700-bat-harmonic/), [701-crab-harmonic](../../701-crab-harmonic/)
  and [703-cypher-harmonic](../../703-cypher-harmonic/). Daily history stretches back to 2001
  for five of the six names (TSLA from its 2010-06-29 IPO). Every headline number is pinned
  with a content fingerprint and as-of date; see [`docs/results.md`](results.md), reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- **[468-gartley-harmonic](../../468-gartley-harmonic/)** — the classic Gartley: B *retraces*
  XA by **0.618**, and D retraces back to **0.786 of XA — it never leaves the X-A range**.
  Every leg is a retracement. The Shark shares no retracement band with it at all: B and C are
  both *extensions* that overshoot past the structure's own origin, and D is a zone measured
  well outside a simple retracement reading.
- **[698-abcd-harmonic](../../698-abcd-harmonic/)** — the bare two-leg **AB=CD** skeleton (no
  X point at all, no confluence, just B retraces A by 0.618 and D projects an equal-length leg
  from C). This study's five-point X-A-B-C-D structure and its extension-based 1.13-1.618 /
  1.618-2.24 / 0.886-1.13-of-XA grid is a strict superset of what 698 tests; 698's negative
  finding (Signal `NONE`) does not by itself resolve whether the *extra* X-point confluence
  that defines the harmonic zoo adds anything — this study is the direct test of that question
  for the Shark variant.
- **[699-butterfly-harmonic](../../699-butterfly-harmonic/)** — the Butterfly variant: B
  *retraces* XA by a fixed 0.786 (not an extension), and D extends the *original* XA leg only
  **1.27-1.618x** past X — a much *smaller* overshoot window than the Shark's B leg
  (1.13-1.618x, measured off a different leg entirely) and D stays a single-target range, not a
  zone measured relative to XA the way the Shark's is.
- **[700-bat-harmonic](../../700-bat-harmonic/)** — the Bat variant: B *retraces* XA by
  0.382-0.50, C *retraces* AB by 0.382-0.886 (never overshoots), and D retraces back to
  **0.886 of XA — inside the X-A range, never past it.** Both legs are retracements; the
  Shark's B and C are both overshoots past the structure's own prior pivots. A superficially
  similar D-zone anchor (both land near the 0.886-1.13 neighborhood of XA) built from an
  entirely different pivot geometry.
- **[701-crab-harmonic](../../701-crab-harmonic/)** — the Crab variant: B *retraces* XA by
  0.382-0.618, C *retraces* AB by 0.382-0.886 (never overshoots), and D extends the original
  XA leg by a single, tight **1.618x, past point X** — the most extreme *single-point* target
  in the zoo. The Shark shares the "D projects past/near the structure's own geometry" flavor
  but replaces both retracement legs with extensions and replaces Crab's point target with a
  genuine zone.
- **[703-cypher-harmonic](../../703-cypher-harmonic/)** — the Cypher variant: B *retraces* XA
  by 0.382-0.618, C *overshoots* the original A swing by 1.13-1.414x (extension, like part of
  the Shark's own C leg), but **D retraces the freshly-extended XC leg (not XA) by a single,
  exact 78.6%** — a different reference leg entirely (XC vs the Shark's XA) and a point target
  rather than a zone. The two studies share "C overshoots past A" but diverge completely on
  what D is measured against and how tightly it is specified.
- **[77-golden-mean](../../77-golden-mean/)** — plain Fibonacci *retracement* levels
  (38.2/50/61.8%) and round numbers as static support/resistance on a single swing, with the
  same placebo-control design this study borrows. It does **not** test a multi-pivot harmonic
  pattern or a projected target built from four confirmed pivots — this study's D-zone is a
  distinct object. All of these studies independently converge on the same verdict shape:
  Fibonacci ratio labels add no detectable specificity over a randomized control on this
  basket.

None of the siblings test the Shark's own defining signature — an **extension-only pivot
structure (no retracement leg at all) completing into a genuine price ZONE at 0.886-1.13 of
the original XA leg** — against both a drift-matched base rate and a Bonferroni correction, nor
Carney's own "non-standard, 5-0" framing of the pattern; that is this study's own, narrowest
reading of the claim.
