# Results — Study 311 (Government-Shutdown) on real SPY total-return tape

*Generated from [`government_shutdown`](../government_shutdown/) against the shared
total-return SPY cache. The "buy the dip when the government shuts down" folklore tested
as a clean event study: enter at the close of the first session on/after each federal
funding-gap shutdown start, hold H trading sessions, exit at the close. Raced against a
**synthetic event-null** — the same horizon measured around 20,000 random dates on the
same tape — and re-run with a conservative **+1-session** entry lag. As-of **2026-05-31**
(partial June dropped); match the fingerprint to confirm you hold the same tape.*

## Data stamp

| Ticker | Mode | Window | Days | Fingerprint |
|---|---|---|--:|---|
| SPY | total return | 1993-01-29 → 2026-05-29 | 8,390 | `fbc1be821974` |

## The event table — 5 funding-gap shutdowns in the SPY era

| Shutdown | Snapped entry | +20-session return |
|---|---|--:|
| 1995 (Nov, Clinton/Gingrich) | 1995-11-14 | **+6.00%** |
| 1995-96 (Dec-Jan, Clinton) | 1995-12-18 | +0.05% |
| 2013 (Oct, ACA/Obama) | 2013-10-01 | +4.62% |
| 2018 (Jan, Schumer) | 2018-01-22 | **−3.99%** |
| 2018-19 (Dec-Jan, border wall) | 2018-12-24 | **+12.46%** |

Five events. The single +12.46% (the 2018-12-24 Christmas-Eve low, a famous coincidence)
is doing most of the work; drop it and the mean roughly halves.

## The headline — buy the dip, hold H sessions, gross

| Horizon H | n | win-rate | mean | HAC *t* | random-date mean | excess vs random | permutation *p* | block-boot 95% CI |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| 5 | 5 | 80% | +1.76% | +1.65 | +0.24% | +1.52% | 0.138 | [−0.65%, +4.63%] |
| **20** | 5 | 80% | **+3.83%** | **+2.33** | +0.93% | +2.90% | **0.129** | [−1.19%, +8.69%] |
| 40 | 5 | 80% | +6.57% | +3.48 | +1.90% | +4.67% | 0.081 | [+0.04%, +13.63%] |

- The naive HAC *t* clears the inference bar at H ≥ 20 (+2.33 at 20 sessions, +3.48 at
  40). **But that *t* is against zero, not against the right null.** Over 20–40 sessions
  SPY drifts up regardless of any shutdown — the random-date baseline already earns
  +0.93% / +1.90%.
- The honest test is the **excess over a random date**, and there the **permutation
  *p*-value never clears 0.05** (0.129 at H=20, 0.081 at H=40). With **n = 5 events** the
  block-bootstrap CI straddles zero at the short horizon and is enormous at the long one.
- An 80% win-rate on 5 trades is **4 out of 5** — not a number any honest desk would
  stake a claim on.

## Look-ahead check — conservative +1-session entry lag

The shutdown start is calendar-known, so the canonical window uses no lag. Entering one
session later (a strict no-peek fill) still measures essentially the same trade:

| Fill | mean (H=20) | HAC *t* | win-rate |
|---|--:|--:|--:|
| Event close (canonical) | +3.83% | +2.33 | 80% |
| **+1 session (conservative)** | **+2.20%** | **+1.90** | **60%** |

One day of delay knocks the mean down a third, the *t* below 2, and the win-rate from 4/5
to 3/5. A "signal" that fragile to a one-day entry shift was never robust.

## Costs are irrelevant — and that is the point

| round-trip cost | net mean (H=20) |
|---|--:|
| 0 bps (gross) | +3.83% |
| 5 bps | +3.78% |
| 10 bps | +3.73% |

With one round-trip per event, costs barely register. This is **not** a cost story: the
trade dies on inference (n=5, not significant vs drift), not on frictions.

## Synthetic positive control & null — the engine is a faithful detector

The same event-study engine recovers a planted post-event bounce only when one is really
there:

| Synthetic tape | n | mean (H=20) | HAC *t* | excess vs random-date | permutation *p* |
|---|--:|--:|--:|--:|--:|
| Planted bounce (effect = 0.002/day × 20) | 40 | +6.31% | +8.31 | +5.88% | 0.0000 |
| Null (no planted effect) | 40 | +0.99% | +1.08 | +0.43% | 0.5899 |

The harness banks the planted effect at p < 0.0001 and correctly finds nothing in the
null (p = 0.59). Across 20 seeds the null's excess-vs-random averages ~0 and only ~5% of
seeds give p < 0.05 — exactly the nominal false-positive rate. So the real-tape result is
a statement about the *market*, not a broken detector: there simply isn't enough
shutdown-specific signal in 5 events to clear the right null.

## Verdict

- **Signal — WEAK.** The naive HAC *t* clears 2 at long horizons (+2.33 at H=20, +3.48 at
  H=40), but that is mostly equity drift you'd earn on *any* random date; the excess over
  random dates is **not significant** (permutation *p* = 0.13 at H=20, 0.08 at H=40), the
  effect is fragile to a one-day entry lag (*t* → 1.90), and **n = 5** makes any *t* a
  small-sample illusion. Real-data inference cannot certify it — WEAK, not REAL.
- **Tradability — MIRAGE.** The "edge" is the market's normal drift, dressed up as a
  shutdown trade. Costs don't touch it; significance over the right benchmark is what
  kills it. You are paid for holding SPY, not for the shutdown.
- **Buy the dip *every time*? — NOT SUPPORTED.** Five events, one of them negative
  (−3.99% in 2018), the headline driven by a single Christmas-Eve coincidence. The folk
  rule rests on survivorship of memorable rebounds, not on a reproducible effect.
