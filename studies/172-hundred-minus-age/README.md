# Study 172 — Hundred-Minus-Age

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | HMA genuinely lowers variance of terminal wealth vs 100% equities (Vol 1.23 vs 2.13) — the defensive mechanism is real. But it does so at a certified return cost: HAC *t* = **−5.4** vs 60/40, trailing in **63.5%** of historical 40-year cohorts. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | 60/40 dominates HMA on mean return, ties on floor risk, and requires less implementation fuss (no annual weight adjustment). There is no investor preference under which HMA is the optimal choice between these two options. |
| **Smarter de-risking?** | ![Fragile](https://img.shields.io/badge/Smarter_de--risking%3F-Fragile-8b949e?style=flat-square) | The Pfau-Kitces rising-equity glidepath earns *more* than HMA (HAC *t* = +2.6) but still trails 60/40 (HAC *t* = −3.4). The "smarter" variant wins the within-glidepath race but doesn't overtake the constant-mix benchmark. |

> **In one sentence:** the famous "put your age in bonds" rule mechanically lowers the variance of retirement outcomes vs 100% equities — but 60/40 beats it on both mean return and floor protection, making the rule a dominated strategy over 142 years of Shiller data.

## What we tested

The folk rule: invest *(100 − age)%* in stocks, the rest in bonds, gliding down one percentage point per year. We simulate 40-year accumulation lifecycles (age 25 to 65) across all 1,234 rolling 40-year cohorts in the Shiller real-return dataset (1881-2023), comparing HMA against a **constant 60/40** (the mainstream alternative), **100% equities** (the return ceiling), and the **Pfau-Kitces rising-equity glidepath** (the leading academic challenger). Inference uses a Newey-West HAC t-stat on overlapping cohort terminal-wealth differences. A synthetic positive control confirms the engine correctly prices the trade-off as a function of the planted equity risk premium.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the rule, the intuition behind it, the two ways it can fail (return cost, floor risk), and a plain-English comparison against 60/40 |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | cohort distributions, HAC t-stats, the Pfau-Kitces challenge, synthetic positive control, equity premium sensitivity |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`hundred_minus_age/`](hundred_minus_age/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
