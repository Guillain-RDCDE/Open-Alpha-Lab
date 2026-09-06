# Study 1012 — Choose Your Benchmark 🎯

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — how much does a fund's measured alpha depend on the benchmark chosen? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Enormously, and by more than the noise everyone does report. Running 17 funds against 9 defensible benchmarks, the median fund's alpha ranged over **4.07% a year** depending only on what it was measured against — **1.8× the median standard error** of any single estimate. The specification is a larger source of uncertainty than the sampling error that gets the confidence interval, and it is the one with no error bar at all. **47% of funds change the sign of their alpha** across benchmarks, and 0% are *significantly positive against one benchmark and significantly negative against another* — both at |t| > 2, both defensible, both publishable. The worst case here was VNQ, spanning 7.25%. Nor is this only a matter of picking obviously wrong comparators: climbing the specification ladder from a single market factor to five, alpha moved from +0.60% to +1.62% for the headline fund, with every rung a specification someone publishes. |
| **Tradability** — can the benchmark be chosen from the data, or is it always a judgement call? | ![Useful](https://img.shields.io/badge/Useful-2ea44f?style=flat-square) | Only sometimes, and knowing when is the practical contribution. Bootstrapping which benchmark fits best, a single candidate won more than 80% of resamples for just **88% of funds**; for the rest the winner changed from one resample to the next, so "which benchmark" is a judgement call and the alpha should be reported as a range. The synthetic control prices the danger exactly, because there the truth is planted. A fund with a **true alpha of zero** loading on two correlated factors, benchmarked against only one of them, showed a measured alpha of +4.07% a year — and it was **statistically significant at |t| > 2 in 17% of simulations**, against 12% with the correct benchmark. A plausible mis-specification is a machine for manufacturing publishable alpha. Two things a reader can do. Ask for the **grid**, not the number: every alpha should arrive with the range across reasonable comparators, which costs nothing to compute. And treat a best-fitting benchmark as a search: choosing the highest-R² of 9 candidates and then quoting a conventional t-statistic ignores the selection, which here was worth 3.35% a year between the best-fitting and the best-flattering choice. |

> **In one sentence:** The median fund's alpha moves 4.07% a year across defensible benchmarks — 1.8× its own standard error — and a mis-specified benchmark makes a genuinely zero alpha look significant 17% of the time.

## What we tested

Alpha is what is left over after a benchmark is subtracted, which makes it a
statement about two things and only one of them gets discussed.

**The spread is compared against the noise.** Every fund is run against every defensible
benchmark and the range of resulting alphas is set beside the standard error of any single one.
The specification turns out to be the larger source of uncertainty — and the one with no error
bar. Funds change the *sign* of their alpha across comparators, and some are significantly
positive against one benchmark and significantly negative against another, both at |t| > 2.

**Two ways the choice gets made, both measured.** A specification ladder from one factor to five
shows alpha shrinking at every rung with no statistical rule saying where to stop. Choosing the
best-*fitting* benchmark is a search over candidates, and the gap between the best-fitting and
the most-flattering choice is reported so a reader can see what the selection was worth.

**Can the data decide?** A block bootstrap counts how often each candidate wins on R². Where one
wins consistently the question is empirical; where the winner changes from resample to resample
it is a judgement call, and the alpha belongs in a range. Encompassing tests add the case where
*both* candidates are significant, meaning no single-index alpha was ever well defined.

**A control where the truth is planted.** A simulated fund with a **true alpha of exactly zero**
loading on two correlated factors is benchmarked against one of them. The measured alpha is not
merely biased but frequently *significant* — and the damage is largest when the two factors are
least correlated, inverting the intuition that similar benchmarks are interchangeable.
**Dedup:** distinct from **1005-beta-stability** (the shelf life of one loading),
**1001-purged-cv-embargo** (validation) and **860-backtest-overfitting** (parameter search); the
subject here is the specification of the performance measurement itself.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | how much of any reported alpha is a fact about the manager and how much is a fact about the yardstick |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the full alpha surface, spread against standard error, a specification ladder, a bootstrap over benchmark choice, encompassing tests, and a planted-zero-alpha control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`benchmark/`](benchmark/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
