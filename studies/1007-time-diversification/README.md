# Study 1007 — Time Does Not Diversify ⏳

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does holding longer actually reduce the risk of owning equities? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Both sides of this argument are quoting true statements about the same windows. On SPY over 33 years: the standard deviation of the **annualised** return falls from 16.7% at one year to 1.6% at 20 — the adviser's chart, and it is correct. Over the same windows the standard deviation of **terminal wealth** rises from 0.17× to 1.48× — Samuelson's point, also correct. The question is whether the first is *more* than arithmetic. Under i.i.d. returns annualised dispersion must fall like 1/√T, a log-log slope of exactly −0.5. Measured here: **-1.003** (±0.147). A block bootstrap that destroys multi-year mean reversion while keeping the short-run dynamics puts the null slope at -0.673 with a 5th percentile of -0.889, so the observed value is inside it (p = 0.660). Lo-MacKinlay variance ratios agree: at 756 days VR = 0.913 with z = -0.20. The convergence is not merely the √T in the denominator. |
| **Tradability** — should an investor's equity weight depend on their horizon at all? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Which leaves the decision, and there the answer is cleaner than the debate. Maximising CRRA certainty equivalent over the real windows, the optimal equity weight moved by at most **68% across horizons from one to 20 years** — at γ = 3 it went from 98% to 100%. Samuelson's 1969 theorem, that horizon drops out under CRRA and i.i.d. returns, survives contact with the data. The practical reading is not "ignore your horizon" but something more precise: horizon should enter an allocation through **the things that genuinely depend on it** — the size and certainty of future contributions, the flexibility to defer spending, the ability to keep earning through a drawdown — and not through a belief that equities become less risky if you wait. The shortfall probability does fall, from 20% at one year to 0% at 20, but the worst outcome gets worse: the poorest 20-year window ended at 2.52× against 0.55× for the poorest single year. Less likely, more costly, which is precisely the trade the glide-path argument leaves out. |

> **In one sentence:** Annualised dispersion narrows at a log-log slope of -1.00 against the −0.50 that arithmetic alone requires, so there is no time diversification here — and the CRRA-optimal equity weight barely moves with horizon regardless.

## What we tested

A sixty-year-old argument in which both sides are right, because three
different quantities are being called "risk".

**Annualised return dispersion** falls with horizon — the adviser's chart. **Terminal wealth
dispersion** rises with it — Samuelson's. **Shortfall probability** falls while the **worst
outcome** gets worse. All four are computed on identical windows so they cannot be quoted
selectively.

**The question that actually matters** is whether annualised dispersion narrows *faster* than
the 1/√T that arithmetic alone guarantees. A log-log slope is fitted and compared against −0.5,
Lo-MacKinlay variance ratios are computed with the heteroscedasticity-robust statistic — the
homoscedastic version over-rejects on equity returns and would manufacture the finding — and a
block bootstrap that destroys multi-year mean reversion while keeping volatility clustering
provides the null. A synthetic world that is **i.i.d. by construction** shows the −0.5
convergence appearing where mean reversion is impossible; a planted mean-reverting world shows
the machinery detects the real thing when present.

**The decision is then addressed directly.** Samuelson's 1969 theorem says the CRRA-optimal
equity weight is independent of horizon; that is checked numerically on the real windows rather
than assumed. Sample honesty is enforced throughout: an `effective_n` column reports how many
genuinely independent observations each long-horizon row is worth, and it falls below two.
**Dedup:** distinct from **1008-start-date-lottery** (path dependence of a single start date),
**1002-best-days-missed** (extreme days) and **1006-most-stocks-underperform-cash** (the
cross-section); the subject here is horizon and the definition of risk.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why 'stocks are safe in the long run' and 'no they are not' are both true, and which one your decision depends on |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | three risk metrics on identical windows, a 1/√T benchmark, robust variance ratios, a block-bootstrap null, and Samuelson's theorem checked numerically |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`timediv/`](timediv/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
