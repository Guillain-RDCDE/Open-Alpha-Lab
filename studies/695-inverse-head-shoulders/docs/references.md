# References & literature map — Study 695 (Inverse Head-and-Shoulders)

## The claim under test

- **The folk recipe.** The head-and-shoulders is classical charting's flagship reversal figure —
  Richard Schabacker (*Technical Analysis and Stock Market Profits*, 1932) and then Robert Edwards
  & John Magee (*Technical Analysis of Stock Trends*, 1948, the field's founding text) codify it:
  a downtrend makes a low (**left shoulder**), rallies to a **neckline**, falls to a *deeper* low
  (the **head** — the exhaustion point), rallies again, falls to a *third*, shallower low (**right
  shoulder**, roughly matching the first) — and a confirmed close **above** the neckline signals
  the reversal is real. Bulkowski's *Encyclopedia of Chart Patterns* (2021 ed.) tabulates the
  inverse H&S among the higher hand-scored "success rate" bottoms in his curated sample. The
  second half of the folklore — **the measured-move target** (project the head-to-neckline height
  upward from the neckline) — is sold as a genuine price *forecast*, not just an entry trigger.
  We steelman this as: *forward returns after a mechanically-confirmed neckline break beat the
  name's own base rate, net of costs, and the measured-move target is reached meaningfully more
  often than chance.*

## Why the steelman is *almost* coherent — the real ideas it leans on

- **Support/resistance as memory.** A defended neckline level has a genuine micro-structure
  rationale — limit orders, round numbers, the disposition effect (Shefrin & Statman, 1985)
  parking supply near a prior high. Three touches at a shrinking-severity floor is at least a
  *plausible* exhaustion signature, not pure numerology.
- **Reversal exists — at other horizons.** Jegadeesh (1990) and Lehmann (1990) document one-month
  single-stock reversal; De Bondt & Thaler (1985) document 3–5-year reversal. A bottoming figure
  spanning weeks sits between those regimes — the open question this study actually measures.
- **Lo, Mamaysky & Wang (2000), "Foundations of Technical Analysis"** (Journal of Finance) build
  the field's only serious kernel-regression pattern detector and report head-and-shoulders (both
  orientations) among the patterns with *some* measurable conditional distributional difference —
  the most credible academic anchor for taking the shape seriously enough to test rigorously.

## The failure mode exposed

- **Subjectivity → selection by eye.** "Three troughs, deeper in the middle, roughly horizontal
  neckline" admits real discretion; a shape recognized only after the fact is curve-fittable. Our
  mechanical swing-pivot detector removes the hindsight, and a strictness sweep (tight/base/loose
  shoulder and neckline tolerances) shows the (non-)result is not a single-tolerance artefact.
- **The up-drift base rate.** Any *long* signal on equities inherits the market's positive drift —
  the raw post-breakout return looks positive almost by definition on a 20-year large-cap basket.
  Netting out each name's own base rate (excess return) and racing the result against a same-tape
  random-date placebo strips the illusion.
- **The measured-move target is untested against the null it needs.** A target N% above the entry
  will get hit *some* of the time on a rising tape purely by chance — "74% hit rate" sounds
  impressive read alone. The only honest question is whether it beats a **magnitude-matched**
  placebo: does a random long position aiming for a move of the *same size* hit it as often? This
  study runs that comparison directly — the reading no prior desk study on this bench performs.
- **This echoes the broad "technical figures don't survive an honest test" literature:** Brock,
  Lakonishok & LeBaron (1992); Sullivan, Timmermann & White (1999), *"Data-Snooping, Technical
  Trading Rule Performance, and the Bootstrap"* (Journal of Finance); Park & Irwin (2007), *"What
  Do We Know About the Profitability of Technical Analysis?"* (Journal of Economic Surveys).

## Method lineage (the desk's shared engine)

- **HAC / Newey-West t-stat.** Newey & West (1987), *"A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix"* (Econometrica) —
  [`strategy.hac_t`](../inverse_head_shoulders/strategy.py).
- **Wilson (1927) score interval** on the measured-move hit rate —
  [`strategy.wilson_interval`](../inverse_head_shoulders/strategy.py).
- **Same-tape random-date placebo** (excess-over-base-rate null) and a **magnitude-matched
  placebo** (same target size, random entry) —
  [`strategy.run_experiment` / `strategy.measured_move_hits`](../inverse_head_shoulders/strategy.py).
- **Reproducibility stamp.** [`quantlab/repro.py`](../../../quantlab/repro.py) — the as-of freeze
  and content fingerprint the headline run carries.

## Data sources used here

- **Yahoo! Finance daily bars** (via `yfinance`), auto-adjusted (split + dividend) OHLC across
  SPY + 29 long-listed US large-caps — the same 30-name **survivors** basket as sibling studies
  415/416 — 2005-01 → 2026-06 (as-of 2026-06-30). The offline reproducible core and the synthetic
  control run on the deterministic
  [`data.synthetic_panel`](../inverse_head_shoulders/data.py) generator, never the network. The
  headline is pinned with an as-of date and a content fingerprint (see
  [`docs/results.md`](results.md)).
- No scheduled-event calendar applies — the claim is a pure price-shape rule, so there is no
  "facts, no network" table to hardcode (contrast with the FOMC-calendar studies).

## Related desk studies (the dedup map — what this study is NOT)

- **[188-head-shoulders](../../188-head-shoulders/)** — the **bearish top** twin (and an earlier,
  smaller-basket pass at the inverse pattern too). That study's basket (10 names) found the
  inverse variant fires too rarely for inference (*n* = 20). This study widens the basket to the
  30-name survivors panel used by 415/416 (*n* = 229 confirmed breakouts) specifically to give the
  bottom variant the statistical power the top variant's companion piece never had, and adds the
  **measured-move target test** 188 does not run. This study does **not** re-test the bearish top
  — that axis is 188's.
- **[415-triple-top-bottom](../../415-triple-top-bottom/)** — the **three-tap** reversal figure
  (a flat support/resistance tapped three times), not a shape with a deepening-then-shallowing
  three-trough structure. Same basket, same base-rate-excess + placebo machinery, different figure.
- **[416-rounding-bottom](../../416-rounding-bottom/)** — a **continuous parabola** ("saucer"),
  no discrete three-trough structure and no neckline/measured-move rule at all. Same basket,
  same honest-teardown discipline, structurally different claim.
- **[696-double-bottom](../../696-double-bottom/)** — the **two-trough** cousin (no head, no
  middle exhaustion point) — the natural "is three troughs better than two" comparison, run as
  its own separate study rather than folded in here.
- **[411-ascending-triangle](../../411-ascending-triangle/)**, **[413-bull-flag](../../413-bull-flag/)**,
  **[414-falling-wedge](../../414-falling-wedge/)**: the rest of the mechanical-detector chart-figure
  cohort, same honest treatment, different shapes.

None of the siblings test the specific claim this study does: a **three-trough, deeper-in-the-
middle** bottoming figure, its **confirmed neckline break**, and its **measured-move target**.
