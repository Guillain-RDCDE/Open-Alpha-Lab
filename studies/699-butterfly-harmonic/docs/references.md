# References & literature map — Study 699 (Butterfly-Harmonic)

## The claim under test

- **The Butterfly harmonic.** A five-point **X-A-B-C-D** structure in the "harmonic
  pattern" family: B retraces XA by **0.786**, C retraces AB somewhere in
  **0.382-0.886**, and **D extends the ORIGINAL XA leg by 1.27-1.618x, past point
  X** — the defining feature that separates the Butterfly from every other member
  of the zoo. Gartley's D (0.786 of XA), Bat's D (0.886 of XA) and Crab's D (1.618
  of XA, but off a different B retracement) all keep D somewhere between X and A or
  just past A; the Butterfly's whole identity is that D overshoots the *origin*
  point X. Believers read the overshoot as a climactic, exhaustion move and treat
  D as a "potential reversal zone" (PRZ) — a high-probability buy (bearish
  Butterfly: sell).
- **The source.** Bryce Gilmore's *Geometry of Markets* first described extension
  structures of this kind; **Scott M. Carney** codified and named "Butterfly" in
  *The Harmonic Trader* (1998) and fixed the exact 0.786 / 1.27-1.618 grid in
  *Harmonic Trading, Volumes 1-3* (2004-2010) — the same taxonomy that produced
  Gartley, Bat and Crab (see the dedup map below). H. M. Gartley's original 1935
  five-point figure predates any of this and did not specify the ratios at all —
  Larry Pesavento (1997) and Carney bolted the Fibonacci grid on afterward.
- **The mechanism claimed.** No causal story beyond "traders watch these ratios and
  place orders at the PRZ, so the confluence becomes self-fulfilling" — the same
  order-clustering logic examined (and found wanting for Fibonacci specifically) by
  siblings [77-golden-mean](../../77-golden-mean/) and
  [468-gartley-harmonic](../../468-gartley-harmonic/).

## What we measure, and the honesty rails

- **Confirmed pivots only, no look-ahead.** A percentage-threshold zigzag records a
  swing pivot only at its *confirmation* bar — the session where price has already
  reversed far enough to lock it in — never at the (earlier) extreme itself. The
  projected D level is fully computable, and the D-touch scan starts, the moment C
  confirms: a real trader could have placed this exact order in real time.
- **A random-day base rate, not just a placebo ratio grid.** The pooled and
  per-ticker Signal tests compare the Butterfly D-touch fade against a base rate
  built from random entry days on the *same* ticker, matched to the *same*
  empirical mix of bullish/bearish setups (pooled over 20 seeds) — this is the
  specific confound named in the study brief: any directional rule on an
  upward-drifting tape inherits some of the market's own drift, and the base rate
  neutralizes it.
- **Bonferroni correction, stated up front.** The basket is tested seven ways (one
  pooled headline + six per-ticker splits, all at the 5-day horizon). Reporting the
  best of seven without correcting for having looked seven times is exactly the
  data-snooping trap this desk exists to catch; the critical |*t*| is raised from
  the naive 2.0 to **2.69** (family-wise alpha 0.05, normal approximation) before
  any test is allowed to "survive."
- **A second, independent placebo — the third-axis myth-check.** Beyond the base
  rate, the identical pivot-detection and D-touch-scanning pipeline reruns with
  each candidate's AB-retrace and D-extension *targets* replaced by a
  deterministic, seeded, off-Butterfly draw — "would ANY extension-and-reversal
  projection have worked", not specifically the 0.786 / 1.27-1.618 pair Carney
  trademarked. Only a Butterfly arm that *beats* this placebo would be evidence
  these particular ratios carry information an arbitrary equal-legged extension
  does not.
- **HAC / Welch statistics.** Newey & West (1987), *A Simple, Positive
  Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance
  Matrix* (Econometrica), for the within-arm mean test (Butterfly events on the
  same tape can cluster in time); Welch's *t* for the base-rate and placebo
  arm-vs-arm comparisons. Wilson (1927) interval on the hit rate.
- **One documented execution convention**, identical across arms: enter the fade at
  the touch bar's own close (the touch is observed intrabar, via that bar's
  high-low range, then executed at the same session's close).
- **Costs one-way × NAV per leg**, both legs charged (2× per round trip); a
  faithful-engine synthetic control (tunable mean-reversion knob) proves the
  detection + inference pipeline is unbiased on a null and recovers a planted
  effect — never cited in support of the real-tape stamp.

## Why the evidence is weak even before our test

- Carney's own books require a **confluence zone** — multiple overlapping
  Fibonacci projections from several legs converging near D, plus other filters —
  rather than the bare XABCD skeleton tested here. This study deliberately tests
  the *simplest, most literal* reading of the claim (the two defining ratios: AB
  retraces XA by 0.786, D extends XA by 1.27-1.618) because that is the version
  every retail charting tutorial and auto-scanner (TradingView, MetaTrader,
  Thinkorswim) implements first; a looser, multi-confluence definition would only
  add researcher degrees of freedom, not remove them.
- Academic tests of harmonic/Gartley-family patterns are scarce and largely
  negative: Boasiako, Iyke & Krige (2023) and the broader technical-pattern
  literature in the Sullivan, Timmermann & White (1999) tradition — *Data-Snooping,
  Technical Trading Rule Performance, and the Bootstrap* (Journal of Finance) —
  find that once a proper data-snooping / multiple-comparison correction is
  applied, most rule-based chart-pattern "edges" published without a randomized
  control and without a Bonferroni-style penalty do not survive. This study
  supplies both missing controls explicitly.

## Data sources

- **Yahoo! Finance daily bars** (via `yfinance`), total-return adjusted, six liquid
  instruments (SPY, QQQ, AAPL, MSFT, TSLA, NVDA) — the identical basket used by
  [698-abcd-harmonic](../../698-abcd-harmonic/). Daily history stretches back to
  2001 for five of the six names (TSLA from its 2010-06-29 IPO). Every headline
  number is pinned with a content fingerprint and as-of date; see
  [`docs/results.md`](results.md), reproduced by
  [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- **[468-gartley-harmonic](../../468-gartley-harmonic/)** — the classic Gartley:
  B retraces XA by **0.618** (not 0.786), and **D retraces back to 0.786 of XA —
  it never leaves the X-A range.** Gartley's whole reading is "price returns
  toward its origin and then turns"; the Butterfly's is "price *overshoots* the
  origin and then turns." Different geometry, different B ratio, and (per 468's
  own finding) a signal that only shows up at a 60-day horizon and fails its own
  ratio-grid placebo — this study runs the equivalent tests on the Butterfly's own
  ratios and finds the same *shape* of failure (a borderline uncorrected number
  that a placebo and a multiple-comparison correction both erase), independently.
- **[698-abcd-harmonic](../../698-abcd-harmonic/)** — the bare two-leg **AB=CD**
  skeleton (no X point at all, no confluence, just B retraces A by 0.618 and
  D projects an equal-length leg from C). This study's four-point X-A-B-C
  structure and its 0.786/1.27-1.618 grid is a strict superset of what 698 tests;
  698's negative finding (Signal `NONE`) does not by itself resolve whether the
  *extra* X-point confluence that defines the harmonic zoo adds anything — this
  study is the direct test of that question for the Butterfly variant.
- **[700-bat-harmonic](../../700-bat-harmonic/)** — the Bat variant: a *shallower*
  B retracement (0.382-0.50 of XA, vs the Butterfly's deep 0.786) and D projects
  back to **0.886 of XA — inside the X-A range**, like Gartley, not beyond it like
  the Butterfly. Different ratio grid, different D geometry.
- **[701-crab-harmonic](../../701-crab-harmonic/)** — the Crab variant: an even
  *more extreme* extension than the Butterfly, with D projecting **1.618x the XA
  leg** off a shallower B retracement (0.382-0.618) — the "sharpest" member of the
  zoo. Distinguishing the Butterfly's specific 0.786/1.27-1.618 grid from the
  Crab's 0.382-0.618/1.618 grid is exactly the kind of researcher-degree-of-freedom
  the placebo control in this study (and in 701) is built to catch.
- **[77-golden-mean](../../77-golden-mean/)** — plain Fibonacci *retracement*
  levels (38.2/50/61.8%) and round numbers as static support/resistance on a
  single swing, with the same placebo-control design this study borrows. It does
  **not** test a multi-pivot harmonic pattern or a projected (not merely retraced)
  target beyond the origin — this study's D point, built from four confirmed
  pivots and extending *past* X, is a distinct object. All of these studies
  independently converge on the same verdict shape: Fibonacci ratio labels add no
  detectable specificity over a randomized control on this basket.

None of the siblings test the Butterfly's own defining signature — a **D point
that extends 1.27-1.618x past the original X point** — against both a
drift-matched base rate and a Bonferroni correction; that is this study's own,
narrowest reading of the claim.
