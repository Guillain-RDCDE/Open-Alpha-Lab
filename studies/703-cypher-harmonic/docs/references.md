# References & literature map — Study 703 (Cypher-Harmonic)

## The claim under test

- **The Cypher harmonic.** A five-point **X-A-B-C-D** structure in the "harmonic pattern"
  family, but the outlier of the group: B retraces XA by **0.382-0.618** (the same shallow
  band the Crab uses), C **overshoots the original A swing** — extending **1.13-1.414x of
  the XA leg**, in the same direction as XA, *past* point A — and **D retraces the
  freshly-extended XC leg (not XA, not AB) by exactly 78.6%**, landing back inside the X-A
  range near X. Every sibling pattern's D is measured off XA (Gartley 0.786, Bat 0.886,
  Butterfly/Crab as extensions past X); the Cypher's is the only one referenced off the
  *XC* leg, which is itself only known once C has formed. Believers read this as an
  unusually precise "retest of origin" support/resistance confluence and treat D as a
  high-probability reversal point ("potential reversal zone", PRZ) — a buy in a bullish
  Cypher (D below X's neighborhood, expecting a bounce back up), sell in a bearish one.
- **The source.** **Darren Oglesbee** is generally credited with originating the Cypher
  pattern in the early 2000s trading-forum/webinar circuit; it was subsequently folded into
  the broader "harmonic trading" taxonomy popularized by **Scott M. Carney** (*The Harmonic
  Trader*, 1998; *Harmonic Trading, Volumes 1-2*, 2004/2007) alongside the Gartley, Bat,
  Butterfly and Crab. Unlike those four (all Carney's own codifications of H. M. Gartley's
  1935 five-point figure), the Cypher's C-overshoot-then-XC-retrace geometry is a distinct
  construction bolted onto the same X-A-B-C-D vocabulary.
- **The mechanism claimed.** As with every harmonic pattern, no causal story beyond
  "enough traders watch these ratios and place orders at the PRZ, so the confluence becomes
  self-fulfilling" — the same order-clustering logic examined (and found wanting for
  Fibonacci specifically) by siblings [77-golden-mean](../../77-golden-mean/) and
  [468-gartley-harmonic](../../468-gartley-harmonic/).

## What we measure, and the honesty rails

- **Confirmed pivots only, no look-ahead.** A percentage-threshold zigzag records a swing
  pivot only at its *confirmation* bar — the session where price has already reversed far
  enough to lock it in — never at the (earlier) extreme itself. The projected D level is
  fully computable, and the D-touch scan starts, the moment C confirms: a real trader could
  have placed this exact order in real time.
- **A random-day base rate, not just a placebo ratio grid.** The pooled and per-ticker
  Signal tests compare the Cypher D-touch fade against a base rate built from random entry
  days on the *same* ticker, matched to the *same* empirical mix of bullish/bearish setups
  (pooled over 20 seeds) — this is the specific confound named in the study brief: any
  directional rule on an upward-drifting tape inherits some of the market's own drift, and
  the base rate neutralizes it.
- **Bonferroni correction, stated up front.** The basket is tested seven ways (one pooled
  headline + six per-ticker splits, all at the 5-day horizon). Reporting the best of seven
  without correcting for having looked seven times is exactly the data-snooping trap this
  desk exists to catch; the critical |*t*| is raised from the naive 2.0 to the
  family-wise-corrected threshold (see `docs/results.md`) before any test is allowed to
  "survive."
- **A second, independent placebo — the third-axis myth-check.** Beyond the base rate, the
  identical pivot-detection and D-touch-scanning pipeline reruns with each candidate's
  D-retracement *target* replaced by a deterministic, seeded, off-Cypher draw — "would ANY
  XC-retracement zone have worked", not specifically the precise 0.786 ratio the pattern is
  built on. Only a Cypher arm that *beats* this placebo would be evidence that 0.786 (off
  XC specifically) carries information an arbitrary nearby retracement does not.
- **HAC / Welch statistics.** Newey & West (1987), *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix* (Econometrica), for
  the within-arm mean test (Cypher events on the same tape can cluster in time); Welch's
  *t* for the base-rate and placebo arm-vs-arm comparisons. Wilson (1927) interval on the
  hit rate.
- **One documented execution convention**, identical across arms: enter the fade at the
  touch bar's own close (the touch is observed intrabar, via that bar's high-low range,
  then executed at the same session's close).
- **Costs one-way x NAV per leg**, both legs charged (2x per round trip); a faithful-engine
  synthetic control (tunable mean-reversion knob) proves the detection + inference pipeline
  is unbiased on a null and recovers a planted effect — never cited in support of the
  real-tape stamp.

## Why the evidence is weak even before our test

- Carney's own books (which absorbed the Cypher into the wider system) require a
  **confluence zone** — multiple overlapping Fibonacci projections from several legs
  converging near D, plus other filters — rather than the bare XABC skeleton tested here.
  This study deliberately tests the *simplest, most literal* reading of the claim (the two
  defining structural constraints plus the single 0.786 D-ratio) because that is the
  version every retail charting tutorial and auto-scanner (TradingView, MetaTrader,
  Thinkorswim) implements first; a looser, multi-confluence definition would only add
  researcher degrees of freedom, not remove them.
- Academic tests of harmonic/Gartley-family patterns are scarce and largely negative:
  Boasiako, Iyke & Krige (2023) and the broader technical-pattern literature in the
  Sullivan, Timmermann & White (1999) tradition — *Data-Snooping, Technical Trading Rule
  Performance, and the Bootstrap* (Journal of Finance) — find that once a proper
  data-snooping / multiple-comparison correction is applied, most rule-based chart-pattern
  "edges" published without a randomized control and without a Bonferroni-style penalty do
  not survive. This study supplies both missing controls explicitly.
- The Cypher is specifically marketed (in the retail charting-software ecosystem) as a
  *higher-win-rate* pattern than its Gartley-family cousins precisely because its C-point
  overshoot + XC-referenced D is supposed to filter out weaker setups — this study is one
  of several on the desk to test that "sharper filter" comparative claim directly (via the
  third-axis placebo) rather than merely asking whether 0.786 clears zero.

## Data sources

- **Yahoo! Finance daily bars** (via `yfinance`), total-return adjusted, six liquid
  instruments (SPY, QQQ, AAPL, MSFT, TSLA, NVDA) — the identical basket used by
  [699-butterfly-harmonic](../../699-butterfly-harmonic/),
  [700-bat-harmonic](../../700-bat-harmonic/) and
  [701-crab-harmonic](../../701-crab-harmonic/). Daily history stretches back to 2001 for
  five of the six names (TSLA from its 2010-06-29 IPO). Every headline number is pinned
  with a content fingerprint and as-of date; see [`docs/results.md`](results.md),
  reproduced by [`examples/verify.py`](../examples/verify.py).

## Related desk studies (the dedup map — what this study is NOT)

- **[468-gartley-harmonic](../../468-gartley-harmonic/)** — the classic Gartley: B retraces
  XA by **0.618**, and D retraces back to **0.786 of XA** — the *same* 78.6% number the
  Cypher uses, but measured against a completely different leg (**XA**, not XC), and D
  never leaves the X-A range in the first place because C never overshoots A. Same digit,
  different geometry, different claim.
- **[698-abcd-harmonic](../../698-abcd-harmonic/)** — the bare two-leg **AB=CD** skeleton
  (no X point at all, no confluence, just B retraces A by 0.618 and D projects an
  equal-length leg from C). This study's five-point X-A-B-C-D structure and its
  0.382-0.618 / 1.13-1.414 / 0.786-of-XC grid is a strict superset of what 698 tests; 698's
  negative finding (Signal `NONE`) does not by itself resolve whether the *extra* X-point
  confluence that defines the harmonic zoo adds anything — this study is the direct test of
  that question for the Cypher variant.
- **[699-butterfly-harmonic](../../699-butterfly-harmonic/)** — the Butterfly variant: a
  *deeper*, fixed B retracement (0.786 of XA) and D extends **1.27-1.618x** past X, an
  extension of the *original* XA leg (like the Crab) — not a retracement of a
  freshly-extended XC leg (like the Cypher). Different C geometry entirely: the Butterfly's
  C sits *between* B and A; the Cypher's C overshoots past A.
- **[700-bat-harmonic](../../700-bat-harmonic/)** — the Bat variant: a similar shallow B
  band (0.382-0.50) but C also retraces *between* B and A (never overshoots), and D
  retraces back to **0.886 of XA** — inside the X-A range, but measured off XA, not XC.
- **[701-crab-harmonic](../../701-crab-harmonic/)** — the Crab variant: shares the Cypher's
  0.382-0.618 B band, but C retraces *between* B and A (0.382-0.886 of AB — never
  overshoots), and D extends the **original XA leg** by 1.618x, past X — the single most
  extreme extension in the zoo, and (like the Butterfly) measured off XA, not XC.
- **[702-shark-harmonic](../../702-shark-harmonic/)** — the Shark variant: a non-standard
  5-0 pattern using extensions (not retracements) off B and C, with D projecting
  0.886-1.13 of the XA leg — also built from extensions rather than the Cypher's
  overshoot-then-XC-retracement, and shares no fixed AB retracement band with this study.
- **[77-golden-mean](../../77-golden-mean/)** — plain Fibonacci *retracement* levels
  (38.2/50/61.8%) and round numbers as static support/resistance on a single swing, with
  the same placebo-control design this study borrows. It does **not** test a multi-pivot
  harmonic pattern or a projected target built from four confirmed pivots — this study's D
  point is a distinct object. All of these studies independently converge on the same
  verdict shape: Fibonacci ratio labels add no detectable specificity over a randomized
  control on this basket.

None of the siblings test the Cypher's own defining signature — a C point that **overshoots
the original A swing** and a D point that **retraces the resulting XC leg (not XA, not AB)
by 78.6%**, the only such construction in the harmonic zoo — against both a drift-matched
base rate and a Bonferroni correction; that is this study's own, narrowest reading of the
claim.
