# Results — Study 560 (Odd-Lot-Ratio): the 'dumb-money' fade on a synthetic tape

*Generated from [`odd_lot_ratio/`](../odd_lot_ratio/) over the deterministic synthetic weekly
odd-lot tape (seed 560, `fade_alpha = 0` — the modern post-decimalization null): **1,039 weeks**
(2005-01-14 → 2024-12-06), tape fingerprint `2659c7bf0fd0`. There is **no real free odd-lot ratio
series** to fetch on a no-key retail stack, so this study is synthetic-only by construction (see the
SIGNAL-axis caveat). As-of **2026-06-30**.*

## The verdict, earned — Signal `NONE` · Tradability `MIRAGE` · "Alive post-decimalization?" `BUSTED`

The **odd-lot theory** (Garfield Drew, 1950s): the small retail investor who trades odd lots
(< 100 shares) is chronically wrong — buying tops, selling bottoms — so **fading** odd-lot buying
should pay. We build the textbook contrarian signal (position = −z of the prior-week odd-lot ratio),
test whether the fade slope of forward return on the ratio is negative at a robust *t*, and pin a
weekly contrarian overlay head-to-head against costs.

On the **modern null tape** (the post-decimalization world where algorithmic order-slicing has
scrambled the odd-lot signal, `fade_alpha = 0`) the fade **does not pay**: the HAC slope of
next-week return on the standardised prior-week odd-lot ratio is **+0.032%/z-unit** with a
Newey–West *t* of **+0.50** (the *wrong* sign for a fade, and nowhere near |*t*| ≥ 2); the
correlation is **+0.015**; the label-shuffle placebo puts the odds at *p* = **0.62**. The regime
sort has the panic tail earning **+8.1%/yr** vs the euphoria tail's **+9.7%/yr** — a spread of
**−1.6%/yr** (two-sample *t* −0.20), i.e. euphoria *out*-earned panic, the opposite of the fade. A
weekly contrarian overlay earns **−2.0%/yr gross** and **−2.7%/yr net** (Sharpe −0.11 → −0.15). So
`NONE` on the signal axis, `MIRAGE` on tradability, and `BUSTED` on the myth: the odd-lot fade is a
dead signal on the modern tape, and — this being synthetic-only — no real tape could earn it above
`WEAK` here anyway.

## Data stamp

- **Synthetic odd-lot tape** (seed 560, `fade_alpha = 0`, modern null): 1,039 weeks,
  2005-01-14 → 2024-12-06, fingerprint `2659c7bf0fd0`
- **No real tape.** A clean, long, point-in-time odd-lot ratio series is proprietary
  (TAQ / consolidated-tape odd-lot flags; odd lots were not on the SIP until 2013–14). The desk
  publishes no number it did not compute from reproducible data, so the study is synthetic-only —
  the data-availability limit is the finding, and it caps the SIGNAL axis at `WEAK`/`NONE`.

## The fade slope — the signal is dead on the modern tape

| | value |
|---|---|
| Slope (forward_ret on z prior-ratio) | **+0.032%** per z-unit |
| Newey–West (HAC, 4 lags) *t* | **+0.50** (a *negative* slope would be the fade) |
| corr(prior ratio, forward return) | **+0.015** |
| Label-shuffle placebo *p* | **0.62** |

A negative slope at *t* ≤ −2 would be the 'dumb-money' fade (high odd-lot buying → weak next week).
The tape delivers a tiny *positive* slope indistinguishable from zero, and the placebo says even that
is pure noise.

## The regime sort — panic did NOT beat euphoria

| Prior-ratio tercile (≈346 weeks) | Next-week return (annualised) |
|---|---|
| **Panic** (lowest odd-lot buying) | **+8.1%/yr** |
| **Euphoria** (highest odd-lot buying) | **+9.7%/yr** |
| **Spread (panic − euphoria)** | **−1.6%/yr** (two-sample *t* −0.20) |

The fade predicts panic > euphoria (a *positive* spread). The tape delivers a small negative spread
that is statistically zero — the odd-lot crowd's 'euphoria' weeks were, if anything, marginally
*better*, not worse.

## Robustness — the sign is not even stable, let alone significant

| Window | Fade slope/z-unit | HAC *t* | Reads as |
|---|---|---|---|
| 2005-01 → 2009-12 | **−0.105%** | −0.90 | weak fade-ish |
| 2010-01 → 2014-12 | **+0.101%** | +0.87 | anti-fade |
| 2014-12 → 2019-12 | **−0.151%** | −1.12 | weak fade-ish |
| 2019-12 → 2024-12 | **+0.247%** | **+2.20** | anti-fade, significant the WRONG way |

The slope flips sign across quarters of the sample and never clears |*t*| ≥ 2 *in the fade
direction*; the only window that clears the bar does so with the *wrong* sign. Noise, not a signal.

## Costs — a weekly overlay that loses before you pay for it

| | value |
|---|---|
| Gross return (contrarian overlay) | **−2.0%/yr** (Sharpe −0.11) |
| Net (2 bps/rebalance one-way + 50 bps/yr borrow on the short leg) | **−2.7%/yr** (Sharpe −0.15) |
| Annual turnover | **25.9×** |

The overlay is the wrong sign *before* costs; the ~26× annual turnover and the borrow on the
short-euphoria leg only deepen the hole. There is nothing to harvest.

## Synthetic positive control — the engine is faithful (seed-robust, 25 seeds)

| Planted `fade_alpha` | Mean fade-slope HAC *t* (25 seeds) | |
|---|---|---|
| 0.0000 (modern null) | **+0.19** | flat — no false fade |
| 0.0005 | −0.57 | fade emerging |
| 0.0010 | −1.33 | fade visible |
| 0.0015 | **−2.10** | clears the bar |
| 0.0020 | −2.86 | strong |
| 0.0030 | −4.39 | unmistakable |

At the modern null the slope-*t* is ≈ 0; planting a genuine 'dumb-money' fade (`fade_alpha > 0`)
drives the slope negative and past −2 as it grows. **The detector works** — so the null result is a
statement about a world where the fade is absent (the post-decimalization regime the folklore itself
concedes), not a broken engine. (Control only; never cited for a real-tape stamp — and there is no
real tape here.)

## Why the fade doesn't certify

1. **Data availability (the SIGNAL cap).** No free, no-key retail feed exposes a clean, long,
   point-in-time odd-lot ratio series — it needs proprietary consolidated-tape odd-lot flags, and
   odd lots were not even reported to the SIP until 2013–14. A synthetic-only study can never earn
   `REAL` (that needs a robust *t* ≥ 2 on a *real* tape) — it is capped at `WEAK`/`NONE`.
2. **Decimalization killed the premise.** Post-2001 algorithmic order-slicing shreds one
   institutional parent order into thousands of odd-lot child orders, so an odd lot is no longer
   'dumb retail money'. The signal the theory fades no longer means what it meant in 1950. Our
   `fade_alpha = 0` tape *is* that world.
3. **Sign instability.** Even in the synthetic modern null the fade slope flips sign across
   sub-windows and only crosses |*t*| = 2 in the *anti*-fade direction — the fingerprint of noise,
   not a decayed-but-real edge.

## The honest takeaway

The odd-lot ratio was a genuine sentiment gauge when odd lots really were mom-and-pop retail — the
synthetic control confirms a clean contrarian test *would* bank the fade if it existed. But the
premise died with decimalization and algorithmic order-slicing, the series itself is proprietary and
unmeasurable on a retail stack, and on the modern null tape the fade is statistically zero, sign-
unstable, and loses money net. `NONE` × `MIRAGE`, myth `BUSTED` — small retail trades no longer
mark the wrong side, because most 'odd lots' aren't retail at all.
