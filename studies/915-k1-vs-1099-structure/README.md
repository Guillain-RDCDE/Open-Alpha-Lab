# Study 915 — K-1 vs 1099 🧾

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the 1099 wrapper cost performance? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | PDBC (1099) trailed DBC (K-1) by **0.127 %/yr**, HAC *t* = **−0.21**, paired block-bootstrap CI **[−1.03 %, +0.78 %]**; excess-of-cash Sharpe advantage **−0.008**, CI [−0.059, +0.042]. Null in **both** eras (+0.01 % then −0.32 %, \|*t*\| ≤ 0.73), 3/11 annual wins with a coin flip inside the Wilson interval, unmoved by a punitive 50 bp spread. **Informative, not blind:** the sample could have caught any cost above **~0.9–1.2 %/yr**. The after-tax model cannot promote it — its gap spans **[−0.36, +0.59] pp/yr and changes sign** with the assumed distribution payout. *Named on this axis: selection on salience (the pair was picked because it is the famous K-1/No-K-1 twin), and a **confound** — PDBC is actively managed, so this is a null on the joint wrapper+basket difference, not on the wrapper alone.* |
| **Tradability** — is it bankable? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Nothing to bank. The wrapper difference is tens of basis points against **1.46–5.30 %/yr** of tracking error and **11.4 pp/yr** of index-choice dispersion (USCI +4.68 % vs USO −6.70 % over the same window), and the modelled after-tax gap is smaller than the assumptions generating it. The usable output is the *non*-finding — the No-K-1 convenience appears to be **free** — and free convenience is not alpha. |

> **In one sentence:** the tax-friendly commodity wrapper does **not** visibly give back its convenience in performance — PDBC tracked its K-1 twin DBC to within **0.13 %/yr** (*t* = −0.21) over eleven and a half years, a null tight enough to rule out any wrapper-plus-basket cost above ~1 %/yr, while the after-tax arithmetic that was supposed to break the tie **flips sign** depending on how much the fund distributes.

## What we tested

**DBC** (a commodity pool issuing a **Schedule K-1**, futures taxed under **§1256** at the
60/40 blend but marked to market every year) raced against **PDBC** (the same manager's
*"No K-1"* ETF — a 1940-Act fund whose futures sit in a Cayman subsidiary, so it issues a
**Form 1099**, loses 60/40, and gains deferral). Daily **total-return** closes, both legs
**excess of cash** (BIL), 2014-11-07 → 2026-06-30 (2,926 bars). One execution lag; no
shorting, so no borrow. HAC *t* and a paired 21-day block bootstrap on the daily
difference, an explicit **minimum-detectable-difference**, a daily-vs-monthly tracking
decomposition (lag-1 autocorrelation **−0.43** — mostly microstructure), an era cut, an
asymmetric cost sweep, and an after-tax **model** whose every input — marginal rates,
distribution payout share, the BIL interest proxy, the two expense ratios — is labelled an
ASSUMPTION and **swept**. Named caveats: the expense ratios are **current** ones applied to
the whole window (hindsight, but never subtracted from a return series); PDBC's actual
payout and holdings were **not parsed**, so the two outlier years are interpreted, not
decomposed; PDBC is **actively managed**, so basket drift is inside this comparison.
**Dedup:** [908-optimized-roll-commodities](../908-optimized-roll-commodities/) races
*different indices* (USCI vs DBC/GSG/DJP) to price the roll methodology; 915 holds the
index family constant and varies only the **legal wrapper**. [35-contango](../35-contango/)
and [794-commodity-carry](../794-commodity-carry/) trade the roll yield itself;
[661-uso-roll-decay](../661-uso-roll-decay/) and [619-bito-roll-drag](../619-bito-roll-drag/)
price one vehicle's roll — which cancels in this difference.
[913-tracking-difference-persistence](../913-tracking-difference-persistence/) asks whether
tracking difference *predicts itself* across many pairs; here it is the nuisance term and
the **tax regime** is the subject. [378-etf-nav-premium](../378-etf-nav-premium/) trades the
premium; here it is only diagnosed. [599-tax-loss-harvesting](../599-tax-loss-harvesting/)
treats tax as an action; here it is a fixed property of the envelope you bought.
As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a K-1 actually costs you, why the 5 % "tracking error" is exchange plumbing, and why the tax tie-breaker refuses to break the tie |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the excess-of-cash race, HAC and paired-bootstrap CIs, the minimum detectable difference, the fee-gap reconciliation, the era cut, the full bracket × payout after-tax grid, and the planted-drag synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`wrapper_tax/`](wrapper_tax/). Both legs measured excess of cash (BIL); the
difference carries a Newey-West HAC *t* and a paired circular-block-bootstrap CI. The
after-tax layer is a stated model, not a measurement. **Not investment advice** — research
& education. See [LICENSE](../../LICENSE).*
