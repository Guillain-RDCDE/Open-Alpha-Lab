# Results — Study 581 (Term-Premium): does a model term-premium estimate time long-duration returns?

*Generated from [`term_premium/`](../term_premium/) over this study's cached yfinance tape: daily
adjusted close for **TLT** (iShares 20+yr Treasury), the **10-year yield** (`^TNX`) and the
**3-month bill** (`^IRX`), **2002-07-31 → 2026-06-26**, n = **6,009** trading days (tape fingerprint
`83b8f0823f11`). The term premium is the ACM-style proxy ``tp = y10 − EWMA₍₂₅₂₎(short)``
(fingerprint `c0a43303a561`), ranked out-of-sample over a trailing 252-day window; the signal at
close *t* trades at close *t+1* (one-bar lag). As-of **2026-06-30** (the 2026 tape ends 2026-06-26;
the partial week is dropped).*

## The verdict, earned — Signal `WEAK` · Tradability `MIRAGE` · "Term-premium times duration?" `MIXED`

The **term premium** — the part of a long bond's yield not explained by expected future short
rates (Adrian, Crump & Moench 2013; Fama & Bliss 1987) — is, in theory, the *time-varying
compensation* for duration risk: fat premium → duration richly paid → own long bonds; thin
premium → step aside. We build an ACM-style proxy (10-year yield minus an EWMA-expectations
component — the piece that separates a term-*premium* estimate from the raw curve slope), rank it
out-of-sample, and test whether the fattest-premium days precede higher forward TLT returns.

The **direction is right but the magnitude is faint.** On the headline **21-day** horizon, the
fattest-premium quintile (Q5) earned **+0.43%** forward vs the thinnest (Q1) **+0.17%**, a Q5−Q1
spread of **+0.26%** — but the autocorrelation-robust **HAC *t* = +0.43**, far below the desk's
bar of 2, and a block-shuffle placebo puts *p* = **0.719** (the spread is well inside the null).
The spread stays the right sign across horizons (5d → 126d) yet never clears *t* = 2, and it
**flips sign across sub-periods**: it is real only in **2002-2009** (*t* +2.09) and turns
*negative* in **2017-2021** (*t* −0.82). So `WEAK` on the signal axis (direction correct, decades
of literature behind it, but this real tape alone cannot certify it — a sub-2 HAC *t* and an
unstable sign), and `MIRAGE` on tradability: a timing overlay that owns TLT on fat-premium days
edges buy-and-hold on Sharpe (**0.291 vs 0.247**) only *before* costs bite the wrong way — its
mean-return spread is **−0.32 bps/day** (*t* −0.39) at 9.6 switches/year, i.e. the Sharpe "win" is
a volatility-reduction artefact, not alpha. `MIXED` on the myth: the premium times duration
**pre-2010, not after.**

*This is a **model-estimate** proxy for the ACM term premium — the NY-Fed ACM series is not
reachable from a no-key retail stack — so a `REAL` stamp (which needs a robust *t* ≥ 2 on a real
tape) is out of reach here even in principle: the honest ceiling is `WEAK`.*

## Data stamp

- **Tape**: TLT + `^TNX` + `^IRX`, daily adjusted close, 2002-07-31 → 2026-06-26, n = 6,009,
  fingerprint `83b8f0823f11`
- **Term-premium proxy** `tp = y10 − EWMA₍₂₅₂₎(short)` (percentage points), fingerprint `c0a43303a561`,
  range −1.10 → +3.66, mean +1.56
- **Signal**: out-of-sample 252-day rolling percentile rank of `tp`, lagged one day (trades at *t+1*)

## The headline sort — direction right, magnitude faint (21-day horizon)

| Term-premium quintile (n ≈ 1,182 each) | Forward 21-day TLT return |
|---|---|
| **Q1** (thinnest premium) | **+0.17%** |
| **Q5** (fattest premium) | **+0.43%** |
| **Q5 − Q1 spread** | **+0.26%** (HAC *t* **+0.43**, placebo *p* 0.719) |

The claim predicts Q5 > Q1, and it *is* — but the edge is tiny relative to its noise. The
block-shuffle placebo says a spread this size shows up 72% of the time by chance on this tape: not
a signal you can lean on.

## Horizon sweep — the sign holds, the *t* never clears the bar

| Forward horizon | Q5 − Q1 spread | HAC *t* |
|---|---|---|
| 5 days | +0.00% | +0.02 |
| **21 days (headline)** | **+0.26%** | **+0.43** |
| 63 days | +1.22% | +0.73 |
| 126 days | +2.37% | +0.70 |

The spread grows with horizon (a fat premium does drift into higher long-horizon returns) but the
HAC *t* peaks around +0.7 — the overlapping-return autocorrelation eats the apparent significance.
Direction: consistently *right*. Significance: consistently *absent*.

## Sub-period sweep — the sign is NOT stable

| Period | Q5 − Q1 spread (21d) | HAC *t* | Reads as |
|---|---|---|---|
| 2002-2009 | **+2.22%** | **+2.09** | signal present |
| 2010-2016 | +0.30% | +0.34 | faded |
| 2017-2021 | **−0.85%** | **−0.82** | inverted |
| 2022-2026 | +0.66% | +0.43 | weak/right |

The term-premium timing edge is real *only* in the 2002-2009 sample (the one window that clears
*t* = 2), fades to nothing in 2010-2016, and *inverts* in 2017-2021 (the ZIRP/low-premium era,
when the proxy premium carried little duration information). A signal whose sign depends on the
regime is `MIXED`, not bankable.

## Timing overlay — a Sharpe mirage (costs and turnover)

| | value |
|---|---|
| Timer net Sharpe (own TLT when rank > 0.5, else cash; 2 bps/switch) | **0.291** |
| Buy-and-hold TLT Sharpe | **0.247** |
| Timer net Sharpe @ 5 bps/switch | **0.261** |
| Switches / year | **9.6** |
| Days invested (fraction) | **0.429** |
| Timer − buy-and-hold mean-return spread | **−0.32 bps/day** (HAC *t* −0.39) |

The timer's Sharpe edge over buy-and-hold is **not alpha** — it is sitting in cash 57% of the time,
so it trims volatility more than it trims return. On the honest metric — the *mean-return* spread —
the timer **loses** 0.32 bps/day (*t* −0.39), and it churns ~10 round-trips a year. Costs one-way ×
NAV; the overlay never pays.

## Synthetic positive control — the engine is faithful (seed-robust, 25 seeds)

| Planted `tp_signal` | Mean Q5−Q1 spread (21d) | Mean HAC *t* (25 seeds) | |
|---|---|---|---|
| 0.000 (null) | +0.11% | **+0.20** | flat — no false signal |
| 0.002 | +2.58% | +4.76 | edge emerging |
| 0.004 | +5.06% | +8.88 | clears the bar |
| 0.008 | +10.01% | +15.08 | strong |
| 0.012 | +14.97% | +18.77 | very strong |

At the null the mean HAC *t* is ≈ 0 (no false positive); planting a genuine term-premium timing
edge drives the Q5−Q1 spread positive and the *t* far past 2 as it grows. The detector works — so
the real-tape result (a right-signed but sub-2, sign-unstable spread) is a statement about **this
tape**, not a broken engine. (Control only; never cited for the real-tape stamp.)

## Why it doesn't certify here

1. **A proxy, not the ACM series.** The NY-Fed ACM term premium is a five-factor affine-model
   estimate published as a CSV, not a Yahoo ticker. This study builds the *cheapest honest
   stand-in* — the 10-year yield minus an EWMA of the short rate — which captures the idea
   (strip out the expectations component) but not the full model. A `REAL` stamp needs a robust
   real-tape *t* ≥ 2; on a proxy with a sub-2 HAC *t*, `WEAK` is the ceiling.
2. **Overlapping-return autocorrelation.** Longer horizons show larger spreads but the HAC
   correction (Newey-West) correctly deflates their *t* — the apparent 63d/126d edge is mostly
   persistence, not fresh information.
3. **Regime-dependence.** The premium timed duration in the higher-rate 2002-2009 world and
   inverted in the ZIRP 2017-2021 era. Term-premium models are estimated over full cycles; a
   single 24-year retail proxy straddles regimes where the relationship changes sign.

## The honest takeaway

A model term-premium estimate points the *right way* at long-duration returns — fat-premium days
do precede modestly higher forward TLT returns — and the literature behind the effect is deep. But
on this ACM-style proxy over 2002-2026 the edge is **faint (HAC *t* +0.43), sign-unstable (real
only in 2002-2009), and untradable** (the timer ties buy-and-hold and loses on mean return after
9.6 switches/year). `WEAK` × `MIRAGE`, with the timing myth `MIXED`. The synthetic control confirms
the engine would bank a real edge — so this is the tape (and the proxy) talking, not the code.
