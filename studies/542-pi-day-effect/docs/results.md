# Results — Study 542 (Pi-Day-Effect): does the market "know" its digits of π?

*Generated from [`pi_day_effect/`](../pi_day_effect/) over this study's cached tapes. Headline:
daily **SPY** log-returns, 1993-02-01 → 2026-06-12 (8,399 days, return-series fingerprint
`3e185e607be5`). Robustness: **^GSPC** daily log-returns, 1928-01-03 → 2026-06-12 (24,728 days,
fingerprint `df3897aef897`). Constant-date membership is pure date arithmetic. As-of **2026-06-30**
(the tape ends 2026-06-12; no partial period is kept).*

## The verdict, earned — Signal `NONE` · Tradability `MIRAGE` · "Numerology vs data" `BUSTED`

The claim is a numerology anomaly: dates that encode a famous mathematical constant — **Pi Day**
(3/14), **Euler's day** (2/7), **Tau day** (6/28), **golden-ratio day** (1/6), **√2 day** (1/4),
**Feigenbaum day** (4/6) — supposedly see the market behave differently. On the headline SPY tape
**Pi Day is a non-event**: it earned a mean **+2.55 bps** vs **+4.08 bps** on all other days, a
contrast of **−1.53 bps** with a Welch *t* of **−0.08** (*p* = 0.94) — statistically
indistinguishable from an ordinary day, and if anything *below* average. The pooled six-constant
set does no better (contrast **+4.45 bps**, Welch *t* **+0.47**, *p* = 0.64), and its
**random-date-set placebo *p* = 0.66** says the constant dates are less extreme than a coin-flip set
of the same size. Bonferroni across the six constants leaves *nothing* significant (every corrected
*p* = 1.0 on SPY). `NONE` on signal, `MIRAGE` on tradability (a Pi-Day timing rule nets **−7.45
bps/event** after costs), and `BUSTED` on the myth.

## Data stamp

- **SPY** (headline): daily log-return, 1993-02-01 → 2026-06-12, 8,399 days, fingerprint `3e185e607be5`
- **^GSPC** (long-tape robustness): daily log-return, 1928-01-03 → 2026-06-12, 24,728 days,
  fingerprint `df3897aef897`
- **Constant dates**: π 3/14, e 2/7, τ 6/28, φ 1/6, √2 1/4, Feigenbaum 4/6 — pure calendar arithmetic

## Pi Day vs every other day — SPY (headline)

| | value |
|---|---|
| Pi Days observed | **24** |
| Mean Pi-Day return | **+2.55 bps** |
| Mean of all other days | **+4.08 bps** |
| Contrast (Pi − rest) | **−1.53 bps** |
| Welch two-sample *t* | **−0.08** (*p* = 0.94) |
| HAC (Newey-West) *t* on the Pi-Day mean | **+0.12** |

Pi Day is not merely insignificant — it is a hair *below* the average day. The fear/hope points
nowhere.

## The pooled constant-day set + its random-date-set placebo — SPY

| | value |
|---|---|
| Constant days observed (6 dates pooled) | **141** |
| Contrast (constant − rest) | **+4.45 bps** |
| Welch two-sample *t* | **+0.47** (*p* = 0.64) |
| Random-date-set placebo *p* | **0.66** |

The placebo is the kill shot for numerology: draw random sets of six calendar slots and see how
often they beat the constant set. **66% of random six-date sets are *at least as extreme*** as the
mathematical-constant set — the constants are, if anything, *less* special than a coin flip.

## Per-constant sweep with Bonferroni — SPY

| Constant | n | mean (bps) | contrast (bps) | Welch *t* | raw *p* | Bonferroni *p* |
|---|---|---|---|---|---|---|
| τ (Tau, 6/28) | 24 | +25.2 | +21.2 | +1.29 | 0.209 | **1.00** |
| √2 (1/4) | 23 | −15.5 | −19.7 | −0.66 | 0.518 | **1.00** |
| φ (golden, 1/6) | 24 | +16.7 | +12.7 | +0.60 | 0.556 | **1.00** |
| Feigenbaum (4/6) | 22 | +18.8 | +14.8 | +0.43 | 0.672 | **1.00** |
| e (Euler, 2/7) | 24 | +2.9 | −1.2 | −0.08 | 0.937 | **1.00** |
| **π (Pi, 3/14)** | 24 | +2.5 | −1.5 | −0.08 | 0.937 | **1.00** |

Even the *most extreme* constant (Tau, raw *p* 0.21) is nowhere near significance, and Bonferroni
across the six — the correction any numerology hunt must pay for choosing dates from the calendar —
sends every corrected *p* to **1.00**.

## Tradability — a Pi-Day timing rule (SPY)

| | value |
|---|---|
| Rule | hold the market only on Pi Days, flat otherwise |
| Gross per event | **+2.55 bps** |
| Cost (2 crossings × 5 bps, isolated single-day events) | **−10 bps/event** |
| **Net per event** | **−7.45 bps** |

Each Pi Day is an isolated round trip, so a naive timing rule pays 10 bps of frictions to harvest
2.5 bps of gross — a guaranteed loss. `MIRAGE`.

## Robustness — the sign is not stable (SPY, Pi Day by sub-window)

| Window | Pi Days | contrast (bps) | Welch *t* | Reads as |
|---|---|---|---|---|
| 1993-2000 | 4 | **+37.7** | +3.01 | positive (tiny n) |
| 2000-2008 | 7 | **−12.6** | −0.29 | negative |
| 2008-2016 | 5 | **−43.1** | −1.25 | negative |
| 2016-2026 | 8 | **+15.3** | +0.42 | positive |

The lone "significant" window (1993-2000, *t* 3.01) rests on **four** Pi Days and vanishes under
Bonferroni and out of sample — the textbook shape of a data-mined calendar coincidence. The sign
flips window to window.

## Long-tape robustness — ^GSPC 1928-2026

| | value |
|---|---|
| Pi Days observed | **70** |
| Pi-Day contrast | **−8.82 bps** (Welch *t* −0.77, *p* 0.44) |
| Pooled constant-day contrast | **+6.63 bps** (Welch *t* +1.09, *p* 0.28) |
| Random-date-set placebo *p* | **0.28** |
| Most extreme constant | φ (golden), raw *p* **0.030** → Bonferroni *p* **0.182** |

On 98 years Pi Day is still nothing (*t* −0.77, again *negative*). The one flicker — the golden
ratio at raw *p* 0.030 — is exactly the mirage the method is built to catch: it is the *best of six*
tries, and once you pay Bonferroni for having looked at six dates it collapses to *p* 0.18. Nothing
survives.

## Synthetic positive control — the engine is faithful (seed-robust, 25 seeds)

| Planted `constant_effect` | Mean constant contrast (bps) | Mean placebo *p* |
|---|---|---|
| 0.0 (null) | **+1.07** | **0.447** |
| 0.5 | +51.5 | 0.001 |
| 1.0 | +101.9 | 0.001 |
| 2.0 | +202.7 | 0.001 |
| 4.0 | +404.2 | 0.001 |

At the null the contrast is ≈ 0 and the random-date-set placebo *p* is ≈ 0.45 (uniform); planting a
genuine constant-day bump drives the contrast up linearly and the placebo *p* to ≈ 0.001. The
detector works — so the flat real-tape result is the tape talking, not a broken engine. (Control
only; never cited for the real-tape stamp.)

## Limitations, named on the SIGNAL axis

1. **Thin by construction.** Each constant date occurs at most once a year, so even 98 years of
   GSPC yields only ~70 Pi Days and ~419 pooled constant days. A numerology anomaly this thin can
   never earn a robust real-tape *t* ≥ 2 — the study is capped at `NONE`/`WEAK`.
2. **Digit-reading choices.** Some constants (e = 2/7, φ = 1/6, √2 = 1/4, δ = 4/6) require a
   month/day digit convention; different readings would pick different dates, but the placebo prices
   in *any* six-date choice, so the conclusion is robust to the convention.

## The honest takeaway

The market does not know its digits of π. On SPY, Pi Day is indistinguishable from — and slightly
*worse* than — an ordinary day (*t* −0.08); the pooled constant set loses to 66% of random date
sets; Bonferroni across the six constants leaves nothing; a timing rule nets −7.45 bps/event; and
the sign flips across windows. On 98 years of GSPC the one flicker (the golden ratio) is the
best-of-six mirage that Bonferroni erases. `NONE` × `MIRAGE`, myth `BUSTED`. The synthetic control
confirms the engine would light up on a real bump — so this is a clean null, not a dead detector.
