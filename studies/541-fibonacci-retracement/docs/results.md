# Results — Study 541 (Fibonacci-Retracement): do reversals cluster at the Fib levels?

*Generated from [`fibonacci_retracement/`](../fibonacci_retracement/) over this study's cached
yfinance tapes: daily auto-adjusted OHLC for **8 broad indices/ETFs** — SPY, QQQ, DIA, IWM,
^GSPC, ^IXIC, ^DJI, GLD — each trimmed to the as-of date (partial final day dropped). Combined
tape fingerprint `11844738632b`. Swings are marked by a **5% ZigZag**; a reversal ("the trend
resumes") is bet one bar after the pullback pivot is confirmed and held **20** trading days.
As-of **2026-06-30**.*

## The verdict, earned — Signal `NONE` · Tradability `MIRAGE` · "Fib levels special?" `NOT SUPPORTED`

Chartists claim price reverses **at** the Fibonacci retracement levels — 23.6% / 38.2% / 50% /
61.8% of the prior swing. We mark swings with a ZigZag, measure the realised retracement of each
pullback, bet the trend *resumes* at that pivot, and compare the forward return when the pullback
landed **near a Fibonacci depth** against a **placebo** of arbitrary non-Fibonacci fractions
(0.31 / 0.44 / 0.56) interleaved in the *same* depth band. Pooled across **225 Fibonacci-depth
swings** on 8 tapes over 33 years, the reversal bet earns **−15 bps** (one-sample *t* **−0.27**,
HAC *t* −0.23) — indistinguishable from zero and from a coin at the same swings (coin *p* 0.61).
The Fibonacci arm is **no better than the placebo** arm: the Fib-minus-placebo edge is **+64 bps**
but at **two-sample *t* = 0.80** — nowhere near significant, and it **flips sign** across the
robustness grid (from *t* +1.53 at a tight ZigZag to *t* −1.00 at a wide one). So `NONE` on the
signal axis (no reversal edge at Fib levels, and no advantage over arbitrary levels), `MIRAGE` on
tradability (nothing to harvest, sign-unstable, and the trade pays costs to lose), and
`NOT SUPPORTED` on the myth: **any round-ish retracement fraction is just as good — which is to
say, none of them are good.**

## Data stamp

- **Tapes**: SPY `824ee1ead232`, QQQ `440303bbf944`, DIA `a91408cc0ead`, IWM `e3095b6c8ed3`,
  ^GSPC `d58e59a33e3d`, ^IXIC `b801d1aacf8a`, ^DJI `fe1103d94a4c`, GLD `b21fb8c9ee8d`;
  combined `11844738632b`. Daily auto-adjusted OHLC (total-return), each `... → 2026-06-30`.
- **Swings**: 5% ZigZag pivots; realised retracement `(P1−P2)/(P1−P0)` per pullback, kept only for
  genuine retracements `0 < retr < 1` of an impulse-expansion leg (no >100% "retracements", no
  double-counting a leg as both swing and pullback).

## The head-to-head — Fibonacci depths vs arbitrary depths (headline: 5% ZigZag, 20-day hold)

| Arm | swings | reversal return (gross) | win-rate | one-sample *t* | HAC *t* |
|---|---|---|---|---|---|
| **Fibonacci** (0.236/0.382/0.5/0.618) | **225** | **−15 bps** | 0.587 | **−0.27** | −0.23 |
| **Placebo** (0.31/0.44/0.56, same band) | 197 | −79 bps | 0.508 | −1.38 | — |
| **Fib − placebo edge** | — | **+64 bps** | — | **two-sample *t* = 0.80** | — |

The Fibonacci reversal bet does not earn (t −0.27, and it loses a hair before costs). It is not
statistically better than betting reversal at arbitrary interleaved fractions (edge *t* 0.80). A
same-bars coin matches the Fibonacci arm (coin *p* 0.61): the "level" adds no directional
information.

## Robustness — the (non-)edge across the grid

| ZigZag | hold | Fib *n* | Fib arm *t* | Fib − placebo edge *t* |
|---|---|---|---|---|
| 3% | 10d | 505 | −0.68 | **+1.53** |
| 3% | 20d | 505 | −0.71 | +0.84 |
| 3% | 40d | 505 | −1.01 | +0.45 |
| 5% | 10d | 225 | −0.81 | +0.89 |
| **5%** | **20d (headline)** | **225** | **−0.27** | **+0.80** |
| 5% | 40d | 225 | +0.65 | +0.18 |
| 7% | 10d | 139 | −0.53 | +0.18 |
| 7% | 20d | 139 | −0.12 | −0.55 |
| 7% | 40d | 138 | +0.47 | **−1.00** |

Tolerance sweep at the headline config: edge *t* = +0.86 / +0.80 / +0.39 for a ±0.015 / ±0.025 /
±0.035 band. **Nowhere in the grid does the Fibonacci arm clear *t* ≥ 1, and the Fib-minus-placebo
edge wanders from +1.53 to −1.00 — noise, and the sign is not even stable.**

## Per-tape — no consistency

| tape | Fib *n* | Fib arm (bps) | edge vs placebo (bps) |
|---|---|---|---|
| SPY | 22 | −24 | −36 |
| QQQ | 24 | +442 | +768 |
| DIA | 21 | −108 | −215 |
| IWM | 19 | −228 | −478 |
| ^GSPC | 62 | −69 | −15 |
| ^IXIC | 47 | −107 | −55 |
| ^DJI | 20 | +273 | +347 |
| GLD | 10 | −301 | −21 |

The per-tape edge ranges from **−478 bps (IWM) to +768 bps (QQQ)** with no pattern — the pooled
+64 bps is a small average of large, cancelling noise.

## Costs

| | value |
|---|---|
| Fib arm gross (headline) | **−15 bps** |
| Fib arm net (1 bp/side one-way, round trip) | **−17 bps** |

Costs are a footnote: the reversal bet is already flat-to-negative before you pay to place it.

## Synthetic controls — the engine is faithful on both sides (seed-robust)

**Positive control (planted level-effect, 25 seeds).** A clean swing panel where swings that
retrace to a Fibonacci depth are given an extra ``fib_pull`` of forward return; the binning +
two-sample test must recover it.

| planted `fib_pull` | mean Fib−placebo edge *t* (25 seeds) |
|---|---|
| 0.00 (null) | **−0.01** — flat, no false edge |
| 0.02 | +4.58 |
| 0.04 | +9.17 |
| 0.06 | **+13.76** |

**Null on a genuine price tape (20 seeds).** The *whole* pipeline (ZigZag → retracement → bin →
two-sample test) run on a synthetic random-swing price path with no planted level-effect returns
a mean edge *t* of **−0.02** — no false edge from the machinery itself.

Together: at the null both controls sit at ≈ 0; a planted Fibonacci reversal edge is caught and
grows monotonically. The detector works — so the real-tape null is a statement about **the tape**,
not a broken test. (Controls only; never cited for the real-tape stamp.)

## Why the claim doesn't certify here

1. **No reversal edge at Fibonacci levels.** Betting the trend resumes at a Fib-depth pullback
   earns −15 bps (*t* −0.27) over 33 years and 8 tapes — a coin at the same swings does as well.
2. **No advantage over arbitrary levels.** Interleave three non-Fibonacci fractions in the same
   depth band and they perform the same (edge *t* 0.80, and it flips sign across the grid). The
   "magic" of 38.2% / 61.8% is indistinguishable from any round-ish fraction.
3. **Self-fulfilling only if crowded — and it isn't measurable here.** Fibonacci trading is often
   defended as self-fulfilling; if so it would leave a footprint in the reversal rate at those
   depths. On daily index/ETF swings it leaves none.
4. **Data availability.** Free daily OHLC from yfinance limits us to broad, liquid, *surviving*
   tapes; the 5% ZigZag on daily bars yields tens-to-hundreds of swings per tape (pooled for
   power). A tick-level or single-name study could add samples but the level claim, tested this
   cleanly, already fails on the most-watched tapes.

## The honest takeaway

The 38.2% / 61.8% Fibonacci retracement levels do **not** mark where a pullback stops on daily
index/ETF swings: the reversal bet at a Fib level earns nothing (*t* −0.27), it is no better than
the same bet at an arbitrary interleaved fraction (edge *t* 0.80, sign-unstable), and a coin at the
same swings matches it. `NONE` × `MIRAGE`, myth `NOT SUPPORTED`. The synthetic controls confirm the
engine would light up if a real level-effect existed — so this is the tape talking, not the code.
