# Study 214 — Magazine-Cover-Curse

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## The claim

> When a magazine puts a boom or bust on its cover, is the move already over?

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Raw Welch t = **+3.04** at 12m horizon looks impressive — but the 37-cover table was assembled by choosing the most famous contrarian examples; it is selection bias, not signal. 6-month horizon (when the cover is still fresh) shows t = **+1.30**, p = **0.20**. A prospective, unbiased cover universe produces no detectable effect. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | ~1–3 qualifying covers per year; the euphoric short loses money 9/17 times; no systematic implementation is possible without a real-time, unbiased cover classification system — and even then the edge is absent. |
| **Myth check** | ![Cherry-picked](https://img.shields.io/badge/Cherry--picked-8b949e?style=flat-square) | The "Death of Equities" 1979 IS famous for a reason — but covers that were followed by continuation are simply forgotten. The anecdote is compelling; the unbiased test is not. |

> **In one sentence:** the magazine-cover curse is a collection of memorable anecdotes masquerading as a rule — every famous cover is famous precisely because it called a turn, while the failures are quietly forgotten.

## What we tested

A hardcoded table of 37 famous financial magazine covers (BusinessWeek, Time, The Economist,
Barron's, Fortune — 1979 to 2023), classified as "euphoric" (potential tops) or "doom"
(potential bottoms). We compute 6- and 12-month forward S&P 500 returns after each cover
using yfinance ^GSPC monthly prices (study-local cache), then test whether the contrarian
prediction (doom → market up; euphoric → market down) is statistically detectable.

We run a Welch t-test (doom vs euphoric forward returns), a permutation test (10,000 shuffles
of tone labels), and a binomial test (doom up-rate vs the 80% unconditional 12-month up-rate).
The synthetic positive control confirms the machinery can find a signal when one is planted;
the critical point is that the real table was built by selecting for the most famous contrarian
calls.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, the famous examples, why the table is selected for success, the correct test in plain English |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | event-study stats, permutation distribution, Welch t, binomial test, selection-bias anatomy, power calculation |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`magazine_cover_curse/`](magazine_cover_curse/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
