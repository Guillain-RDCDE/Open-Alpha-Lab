# Study 234 — Olympic-Year

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Do stocks win gold in Olympic years?

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Olympic-year mean +8.89% vs non-Olympic +7.74%; contrast = **+1.15 pp**; HAC *t* = **+0.29**; two-sided permutation p = **0.80**. No detectable signal in 97 years / 23 Olympic events. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Olympic-only timer yields **+1.75%/yr** vs **+6.17%/yr** buy-and-hold: being in cash 76% of the time destroys compounding, even before the absent signal problem. |
| **Myth-check: do stocks really rally in Olympic years?** | ![BUSTED](https://img.shields.io/badge/Myth--check-BUSTED-8b949e?style=flat-square) | The raw mean is marginally positive but statistically indistinguishable from noise. The contrast reverses sign post-1972. No mechanism has ever been proposed. |

> **In one sentence:** Olympic years have averaged +1.15 pp more than non-Olympic years since 1928, but with a t-stat of +0.29 and a permutation p of 0.80, this is as close to pure noise as a financial folklore claim can get.

## What we tested

The folk claim: *Summer Olympic years are good for stocks* — a piece of calendar
lore that circulates in financial media every four years.  We take it literally:
on the yfinance ^GSPC annual dataset (1928–2024, n = 97), we separate the 23
Summer Olympic years (hardcoded IOC table) from the 74 non-Olympic years, measure
the mean return contrast with a Newey-West HAC t-stat, run a 50,000-iteration
permutation test, and check whether the pre-1972 pattern holds post-1972.  The
timing strategy (hold only in Olympic years) provides the tradability reality check.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the Olympic year calendar, bar chart of all 23 games, plain-language permutation illustration, why being in cash 76% of the time is structurally unworkable |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t-stat, two-sided permutation anatomy, pre/post 1972 split, positive control confirming the engine detects planted effects |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`olympic_year/`](olympic_year/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
