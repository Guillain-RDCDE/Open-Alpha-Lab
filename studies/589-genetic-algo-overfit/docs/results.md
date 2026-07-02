# Results — Study 589 (Genetic-Algo-Overfit): a GA "discovers" an edge that isn't there

*Generated from [`genetic_algo_overfit/`](../genetic_algo_overfit/) on a **deterministic, offline
synthetic tape** (seed 589). This is a research-method demo, so the tape is built on purpose: the
**null** is a pure random walk (`signal_strength = 0`) on which — by construction — no timing rule
can work, and the **positive control** is a tape with a genuine planted edge (`signal_strength > 0`).
Real free data can never certify "zero edge", so there is no real-tape stamp; the data-availability
limitation is named on the SIGNAL axis and the study is capped at `NONE`. Null tape fingerprint
`e350016fa3d2` (2,999 rows, 7 features). As-of **2026-06-30**.*

## The verdict, earned — Signal `NONE` · Tradability `MIRAGE` · "Does GA search manufacture a false edge?" `CONFIRMED`

We let a **genetic algorithm** evolve a long/flat trading rule — a weighted combination of seven
technical features — to maximise the **in-sample Sharpe** on the first half of a pure random walk.
The GA works: it crowns a rule with an in-sample Sharpe of **0.99**, a backtest that looks like a
real strategy. Then we run that frozen champion, untouched, on the out-of-sample half: the Sharpe
collapses to **0.087** — a shrinkage of **0.90**, essentially all of the in-sample brilliance. The
book is judged on *timing alpha* (forward returns are demeaned, so a long-biased rule cannot inherit
any drift), so this collapse is the overfitting, not a beta story.

Two diagnostics catch the trap cold. The **Deflated Sharpe Ratio** is **0.00**: given the GA
evaluated **2,371** distinct genomes, the expected maximum Sharpe from pure noise is **3.49** — the
observed 0.99 is *below* the luck bar. A **label-shuffle placebo** (re-running the whole
evolve-then-validate protocol on shuffled forward returns) puts the OOS result at *p* = **0.073** —
indistinguishable from searching noise. Across **20 seeds** the null's OOS Sharpe averages **−0.07**
(*t* = **−0.84**): flat, no false signal. And the synthetic **positive control** proves the harness
is not blind — plant a real edge and the GA-found rule *keeps* its Sharpe out of sample.

So `NONE` on the signal axis (a synthetic-only method demo; the "edge" is a search artefact that
dies OOS and clears no deflated-Sharpe bar), `MIRAGE` on tradability (the OOS Sharpe of **+0.087** is
economically zero, and a mild 2 bps turnover cost erases it to **+0.012** — there is nothing to
harvest), and `CONFIRMED` on the myth-check
(the GA reliably manufactures a beautiful in-sample backtest from nothing).

## Data stamp

- **Null tape** (`signal_strength = 0`, pure random walk): 2,999 daily rows, 7 standardised
  features + next-day return, fingerprint `e350016fa3d2`, seed 589.
- **Positive-control tapes** (`signal_strength ∈ {0.15, 0.20, 0.30, 0.40, 0.50}`): same generator,
  a hidden linear combination of the features genuinely predicts the next return.

## The headline — the GA overfits a random walk

| Quantity | Value |
|---|--:|
| GA champion in-sample Sharpe | **0.99** |
| Champion **out-of-sample** Sharpe | **0.087** |
| IS − OOS shrinkage | **0.90** |
| Distinct genomes evaluated (effective trials) | **2,371** |
| OOS long exposure | 0.49 |
| Fitness climb (gen 1 → final best IS Sharpe) | 0.76 → 0.99 |

The in-sample fitness curve climbs generation by generation — the GA *is* getting "better" — but
every bit of that improvement is fitting the training noise. Out of sample it is worth nothing.

## The Deflated Sharpe Ratio — the IS Sharpe is *below* the luck bar

| Quantity | Value |
|---|--:|
| Observed IS Sharpe (annualised) | **0.99** |
| Effective trials (distinct genomes) | **2,371** |
| Expected max Sharpe from noise, that many trials | **3.49** |
| **Deflated Sharpe Ratio** | **0.00** |

The expected-maximum-Sharpe bar rises with the trial count — the more the GA searches, the higher a
Sharpe pure luck delivers for free:

| Trials searched | Expected max Sharpe (noise) |
|---|--:|
| 10 | 1.57 |
| 100 | 2.53 |
| 1,000 | 3.26 |
| 2,371 (this GA) | 3.49 |
| 5,963 (biggest budget below) | 3.73 |

A Sharpe of 0.99 from a 2,371-trial search is not evidence of skill — it is *less* than what noise
would hand you. DSR ≈ 0.

## The placebo — the OOS is indistinguishable from shuffled searches

Re-run the entire evolve-then-validate protocol (a lighter GA budget, 40×25, for tractability) on
**40 shuffles** of the forward returns against the features:

| Quantity | Value |
|---|--:|
| Observed OOS Sharpe (light-budget run) | 0.47 |
| Placebo *p* (fraction of shuffles ≥ observed) | **0.073** |
| Mean shuffled OOS Sharpe | −0.06 |

The real OOS result does not sit in the tail of the shuffle distribution — it looks like one more
search over noise.

## Costs — nothing to harvest

| Quantity | Value |
|---|--:|
| Gross OOS Sharpe (headline champion, demeaned timing book) | **+0.087** |
| Net OOS Sharpe (2 bps per long/flat switch) | **+0.012** |
| OOS turnover (switches per day) | 0.18 |

On the demeaned timing book the champion's OOS Sharpe is an economically meaningless **+0.087**; a
mild 2 bps turnover cost erases it to **+0.012** — a rounding error. The long/flat rule has no short
leg, so there is no borrow: even the mildest possible cost leaves nothing to trade.

## Robustness — the more you evolve, the prettier the IS, the deeper the collapse (null)

| Search budget (pop × gen) | Trials | IS Sharpe | OOS Sharpe | Shrinkage |
|---|--:|--:|--:|--:|
| 15 × 10 | 143 | 0.76 | 0.52 | 0.25 |
| 30 × 20 | 586 | 0.84 | −0.01 | 0.85 |
| 60 × 40 (headline) | 2,371 | 0.99 | 0.09 | 0.90 |
| 100 × 60 | 5,963 | 0.98 | 0.17 | 0.82 |

The in-sample Sharpe rises with the search budget while the out-of-sample Sharpe orbits zero — the
signature of overfitting by search. (The tiny 143-trial budget barely overfits — too few genomes to
find a lucky one — which is itself the point: overfitting *grows* with the search.)

## Synthetic positive control — the harness banks a REAL edge

The same evolve-then-validate protocol on tapes with a *planted* edge:

| Planted `signal_strength` | IS Sharpe | OOS Sharpe | Shrinkage |
|---|--:|--:|--:|
| 0.00 (null) | 0.99 | **0.09** | 0.90 |
| 0.15 | 1.59 | **0.83** | 0.76 |
| 0.30 | 2.42 | **2.14** | 0.28 |
| 0.50 | 3.82 | **3.56** | 0.25 |

When a real edge exists, the GA-found rule *keeps* its Sharpe out of sample and the shrinkage
shrinks toward zero. The method is a detector, not a generator of edges — it fails OOS only when
there was nothing to find.

### Seed-robust control (20 seeds, lighter budget — the house rule)

| Planted `signal_strength` | Mean IS Sharpe | Mean OOS Sharpe | Mean shrinkage | OOS *t* (across seeds) |
|---|--:|--:|--:|--:|
| 0.00 (null) | 1.26 | **−0.07** | 1.34 | **−0.84** |
| 0.20 | 2.08 | **1.24** | 0.84 | **13.0** |
| 0.40 | 3.10 | **2.55** | 0.55 | **17.7** |

Averaged over 20 synthetic worlds so no lucky seed can fake it: the **null's OOS Sharpe is flat**
(mean −0.07, *t* −0.84 — no false signal), while a planted edge is banked OOS with an overwhelming
*t*. (Synthetic control = machinery proof, never market evidence — see METHODOLOGY → inference bar.)

## Why the verdict is what it is

1. **The "edge" is a search artefact.** On a tape we *built* to have zero timing edge, the GA still
   crowns a Sharpe-0.99 rule — and it dies OOS (0.09), clears no deflated-Sharpe bar (DSR 0.00,
   below the 3.49 luck line), and looks like noise to the placebo (*p* 0.073). There is nothing real
   to detect. **Signal `NONE`.**
2. **Nothing to trade.** The demeaned timing book's OOS Sharpe is an economically meaningless
   +0.087; a mild 2 bps cost erases it to +0.012. **Tradability `MIRAGE`.**
3. **The myth is confirmed.** A genetic algorithm reliably manufactures a beautiful backtest from
   pure noise, and the beauty grows with the search budget — exactly why a Sharpe is meaningless
   without its trial count. **`CONFIRMED`.**

## The honest takeaway

Evolutionary optimisation is powerful precisely because it searches hard — which is exactly why, on
finite data, it will *always* find a rule that fits the past beautifully, whether or not the future
cooperates. The Deflated Sharpe Ratio (which needs the trial count the GA hands you for free) and an
honest in-sample/out-of-sample split catch the trap; the synthetic control confirms the same machine
banks a *real* planted edge. `NONE` × `MIRAGE`, myth `CONFIRMED`. This is a method demo on a
synthetic world by design — it can never earn `REAL`, which requires a robust *t* ≥ 2 on a real tape.
