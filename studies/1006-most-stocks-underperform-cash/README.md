# Study 1006 — Most Stocks Lose 🎲

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the median stock underperform cash on a surviving large-cap basket? | ![Busted](https://img.shields.io/badge/Busted-c0392b?style=flat-square) | **Not on this basket, and the failure is the finding.** These 50 names all survived to 2026 — the exact opposite of Bessembinder's universe, which is every firm that ever listed and mostly delisted. Here **91%** of 10-year holding periods beat Treasury bills, against 67% at one year: the share beating cash **rises** with horizon, the reverse of the headline. So the claim is not a universal law about equities, and the useful question becomes what it takes to produce it. The mechanism is real and verified — measured drag across these names averaged 5.87% a year against a theoretical 5.90%, a correlation of 1.00 name by name — but the median compounds at roughly drift minus σ²/2, and these names' drift is large enough to absorb it. The condition is exact: the median beats cash while σ < sqrt(2·(drift − cash)). At an average drift of 16.8% against cash's 1.3%, that threshold is **55% volatility**, and these names average 33% — a headroom of 22%, with 2% of them over the line. Bessembinder's population — small caps and firms on their way to delisting — sits on the other side of that threshold. The synthetic control quantifies the survivorship half of the gap: removing simulated failures raises the median terminal wealth by 0.86×. |
| **Tradability** — what volatility does a cross-section need before its median holder loses? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The implication is not "don't buy stocks", it is "don't buy few stocks", and the reconciliation makes that precise. Over the full sample the median single name returned 15.3× against cash's 0.6×, while the **rebalanced equal-weight basket of those same names** returned 414.2×. The index does not escape variance drag; it escapes most of it, because drag is proportional to σ² and diversification cuts the average single-name volatility of 33% to a portfolio 20% — worth 3.52% a year, compounding. Rebalancing is doing real work rather than housekeeping. Priced as odds over 10 years: a five-stock portfolio beat the index 43% of the time, a twenty-stock 51%, the full basket 14%. A concentrated book is not a slightly worse diversified one — it is a different bet, on the right tail, that most often loses. |

> **In one sentence:** On surviving large caps the famous result reverses — 91% of 10-year holdings beat bills — because the median only loses once volatility exceeds sqrt(2·(drift−cash)) ≈ 55%, and these names run at 33%.

## What we tested

Bessembinder (2018): most US stocks underperform Treasury bills over their
lifetimes. This study set out to reproduce that on fifty large caps and **failed** — the share
of holdings beating bills *rises* with horizon here, from roughly two-thirds at one year to
nearly all at fifteen. The rejected hypothesis is kept visible, because working out why it
failed is what the study delivers.

**The engine is variance drag.** A name's average outcome grows at its arithmetic mean and its
typical outcome at its **log** growth rate, a gap of σ²/2 per year. Both are annualised linearly
here: the identity holds in log space, and comparing against a *compounded* return reports
negative drag for a riskless series, which is ordinary compounding rather than drag. The
measured gap is checked against the formula name by name before anything is built on it.

**The condition that replaced the hypothesis.** The median compounds at drift − σ²/2, so it
beats cash exactly while **σ < √(2·(drift − cash))**. Surviving large caps sit well below that
threshold; small caps and firms heading for delisting sit above it. "Most stocks lose to bills"
is not a law about equities but a description of a cross-section on one side of an inequality,
and the inequality can be evaluated for any universe before its data is examined.

**The reconciliation.** An index of the same names wins because drag is proportional to
variance, and diversification cuts variance — an explicit decomposition, not a story.
Rebalancing does that work: left alone, the basket becomes a concentrated bet and hands the
saving back.

**What still survives.** The concentration half of Bessembinder's result holds even here: the
top decile of names produced most of the basket's excess wealth. And a synthetic control
reproduces the full effect from a **single knob** — with expected return held exactly constant,
raising volatility alone moves the median without moving the mean — while a simulated delisting
rule quantifies how much of the gap survivorship accounts for, instead of footnoting it.
**Dedup:** distinct from **1004-how-many-stocks** (portfolio size and the diversification
curve), **1002-best-days-missed** (extreme days) and **304-fat-tails** (the return
distribution); the subject here is the cross-section of long-run outcomes and its reconciliation
with the index.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | how the typical stock can lose to cash while an index of those same stocks beats it comfortably |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | variance drag checked against σ²/2, the horizon paradox, dollar wealth concentration, an explicit index reconciliation, concentrated-portfolio odds, and a one-knob control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`moststocks/`](moststocks/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
