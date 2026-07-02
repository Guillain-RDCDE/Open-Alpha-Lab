# Study 544 — Oyster-R-Months 🦪

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do R-months (Sep–Apr) out-earn R-less months (May–Aug)? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | No. On the **market** (S&P 500, 1928–2026) R-months earned **less**: gap **−24.5 bp/mo** (R +7.1%/yr vs R-less +10.3%/yr), Welch *t* **−0.72**, placebo *p* **0.457** — the **wrong sign**. On **staples** (XLP) the gap is right-sign but invisible: **+17.9 bp/mo**, *t* **+0.45**, placebo *p* **0.670**. No *t* ≥ 2 anywhere. |
| **Tradability** — does "hold in R-months, cash otherwise" pay? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | No. Sitting in cash for four months that on the market *out-earn* the R-months compounds to a fraction of buy-and-hold: market **28x** vs **425x** (Sharpe 0.31 vs 0.43) over 98 years; staples 3.5x vs 5.8x. The long-R/short-R-less book is negative before borrow. |
| **"R-month rule ≠ a dressed-up sell-in-May"?** | ![Myth](https://img.shields.io/badge/Myth-8b949e?style=flat-square) | It **is** sell-in-May — a *worse* one. The R-split (Sep–Apr vs May–Aug) moves **September** to the hold side, and September is the market's **single worst month** (−111.7 bp/mo over 98 years). That flips the soft-but-positive Halloween gap (+40.6 bp, *t* +1.31) to negative (−24.5 bp, *t* −0.72). |

> **In one sentence:** the folk rule "only eat oysters in months with an R" is food-safety wisdom, not a market seasonal — on the tape it is just **sell-in-May in a costume**, and a *worse* costume, because the R-month calendar hands September (the market's worst month) to the buy side, turning the already-soft Halloween seasonal into a wrong-sign, insignificant null (market *t* −0.72, placebo *p* 0.46).

## What we tested

The **R-month oyster rule** (folk wisdom since ~1599): eat oysters only in months whose name
contains the letter **R** — September through April — and skip the R-less months of **May, June,
July, August**. That split is calendrically almost identical to **sell-in-May / Halloween**
(Bouman & Jacobsen 2002), one month wider on the winter side. We test whether R-months out-earn
R-less months on the broad **market** (S&P 500, 98 years) and on **consumer staples** (XLP, the
food/beverage/grocery ETF an "oyster" seasonal would touch): a two-sample **Welch *t*** on the
monthly-return gap, a **label-shuffle placebo** null, a side-by-side with the sell-in-May cousin, a
September diagnostic, the tradable "hold-in-R-months" rule vs buy-and-hold with switching costs (and
a short-leg borrow sensitivity), pre/post-2000 sub-periods, and a deterministic, seed-robust
synthetic positive control that plants an R-month premium and proves the engine catches it.
*Distinct from [Study 55 — Summer-Lull](../../55-summer-lull/), the straight sell-in-May study: this
one is the **oyster-rule variant** (the R/non-R split that adds September) plus a staples instrument.*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the oyster rule is, why it's sell-in-May in disguise, and why adding September breaks it |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the R-month split with a Welch *t*, the placebo null, the sell-in-May comparison, the September diagnostic, the hold-in-R-months race with costs, the sub-periods, and the seed-robust synthetic control |

The fingerprinted real-data run (S&P 500 fp `e89dca918e03`, XLP fp `15bb025c5ac1`, as-of
2026-06-30) is in [docs/results.md](docs/results.md); the offline machinery proof runs on the
deterministic synthetic world in [`oyster_r_months/data.py`](oyster_r_months/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`oyster_r_months/`](oyster_r_months/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
