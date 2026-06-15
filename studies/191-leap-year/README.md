# Study 191 — Leap-Year

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Welch *t* = **+0.255**, p = **0.80**; Bonferroni-corrected p = **1.00**. The raw +1.02 pp leap-year premium (leap 8.77% vs non-leap 7.75%) is well inside the noise envelope of a 18% vol, 25-observation test. Sub-period contrast reverses sign — noise, not structure. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | One trade per year; no exploitable edge over the unconditional mean (+8.01%/yr). Any transaction cost kills a one-trade-per-four-years strategy. |
| **Confound exposed** | ![Election_year_in_disguise](https://img.shields.io/badge/Election_year_in_disguise-8b949e?style=flat-square) | In the modern era leap years and election years are identical (year % 4 == 0). Neither is the strongest year-of-presidential-term — that's year 4 (mid-term, +13.2%) not year 1 (election=leap, +8.8%). |

> **In one sentence:** the leap-year premium is pure noise — a +1 pp raw gap over 25 observations in an 18% vol world, with a sub-period sign reversal, Bonferroni p = 1.00, and no separation from the unconditional equity drift.

## What we tested

The folk claim: because leap years coincide with US presidential elections and the Summer Olympics (all on the four-year cycle), stocks earn systematically better or worse returns in leap years. We take it literally: calendar-year S&P 500 price returns (Shiller dataset, 1928–2025, n = 98 years, 25 leap years) split by the Gregorian leap-year flag, tested with a Welch t-test and HAC inference, and controlled for the presidential-cycle confound (year-of-term 1–4). A Bonferroni correction over the three quadrennial hypotheses a researcher could have tested (leap premium, election-year premium, mid-term-year premium) confirms that nothing survives the multiple-comparisons bar.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the folk claim, the raw mean comparison, the presidential-cycle confound, the sub-period sign-flip, why n=25 is fundamentally too small to prove a 1 pp effect |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | Welch t-test, HAC t-stats, Bonferroni table (k=3), presidential-cycle breakdown, power analysis, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`leap_year/`](leap_year/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
