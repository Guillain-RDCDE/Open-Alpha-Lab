# Study 1008 — The Start-Date Lottery 🎰

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — how much of a lifetime outcome is decided by the accident of when you started? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Large, and larger than it looks in the usual telling. Contributing the same amount every month into SPY for 25 years, the best available start date turned each unit invested into 5.91× and the worst into 2.85× — a **2.08× gap between two people following an identical plan**, with the 5th-to-95th percentile spanning 1.69×. The best start was 2001-05-25 and the worst 1995-03-30. Two caveats keep that honest: these paths overlap heavily and are worth roughly 1.3 independent observations, and the *lump-sum* spread over the same windows was 1.75× — so much of this is the return distribution rather than sequence. The clean separation is the shuffle test: reordering a single path leaves a lump sum unchanged to 2e-16 relative precision while scattering a contributor's outcome across 2.42×. **That** is pure sequence risk, with the distribution held exactly fixed. And it is not spread evenly: the final sixth of the horizon correlates 0.43 with a contributor's outcome against -0.67 for the first, whereas for a lump sum every period matters equally by construction. |
| **Tradability** — can anything an investor controls reduce that, and by how much? | ![Useful](https://img.shields.io/badge/Useful-2ea44f?style=flat-square) | Yes, and the profile above says which remedy to reach for. Because the exposure is concentrated late, de-risking late buys most of the protection cheaply. Scoring every rule on the same paths — dispersion removed against median wealth given up — the best was **late glide to 30%**, cutting the coefficient of variation by 40% for a median cost of 22%, an efficiency of **1.81× dispersion removed per unit of wealth sacrificed**. A conventional linear glide managed 1.67×, and simply holding a constant 60/40 throughout 1.68×. The ranking is what matters more than the levels, since the levels inherit the overlapping-window problem. Two things worth saying plainly: none of this eliminates the lottery — the residual spread under the best remedy was still 1.31× — and the largest lever is not on this table at all. Contributing for forty years instead of thirty, or retaining the flexibility to defer retirement by two, moves the distribution more than any asset-allocation rule tested here. |

> **In one sentence:** Identical thirty-year plans differed by 2.1× on start date alone, and the cheapest defence is de-risking late rather than early — 1.8× as much dispersion removed per unit of wealth given up as the conventional glide.

## What we tested

Two investors follow an identical plan for decades and end up in different
worlds because of when they were born. This study measures the gap, separates its causes and
prices the remedies.

**The engine is that multiplication commutes.** A lump sum has no sequence risk at all — order
cannot change the product. Everyone with cash flows does. That gives a clean experiment:
shuffling a return path holds the distribution *exactly* fixed and destroys only the ordering,
so the dispersion that survives the shuffle is pure sequence risk and the rest is different
eras having different distributions. Both are measured.

**One result cuts against the usual telling.** Regular contributions *reduce* start-date
dispersion relative to a lump sum, at every volatility tested. The lottery is worse for someone
deploying a windfall than for a saver.

**Where the exposure sits is then measured rather than asserted**, by correlating each slice of
the horizon with the final outcome. For a lump sum every slice matters equally — a check that
the measurement works — while a contributor is dominated by the final years and a retiree by the
first. That profile decides which remedy to reach for.

**Remedies are scored on two columns.** Dispersion removed *and* median wealth given up, with
the ratio between them, because a rule that halves the spread by halving the outcome has helped
nobody. Every dispersion figure carries an effective-sample count — the 30-year windows on a
33-year tape are worth about one independent observation — and a synthetic control with genuinely
independent paths supplies the scale that overlapping windows cannot.
**Dedup:** distinct from **1007-time-diversification** (horizon and the definition of risk),
**997-rebalance-timing-luck** (implementation-date noise within a strategy) and
**1004-how-many-stocks** (cross-sectional dispersion); the subject here is path dependence of a
lifetime plan.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | how much of a lifetime investing outcome is decided by your birth year, and what can actually be done about it |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | a shuffle test isolating pure sequence risk, the exposure profile across the horizon, glide-path shapes scored on two columns, and independent-path controls |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`startdate/`](startdate/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
