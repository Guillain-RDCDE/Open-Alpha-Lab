# Study 192 — Tax-Day

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Pre-tax window (+5.55 bps/day contrast, HAC t = +1.87, Welch p = 0.15, Bonf-p = **0.29**); post-tax window (+7.02 bps, HAC t = +1.82, p = 0.24, Bonf-p = **0.49**). Neither hypothesis survives Bonferroni correction. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | ~15 trading days per year carry the signal; no cost-surviving excess return over a passive buy-and-hold. Any round-trip cost is fatal. |
| **April IRA inflow — real?** | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Point estimates are elevated (pre +7.77 bps, post +9.32 bps vs +2.44 bps unconditional) but cannot clear the statistical inference bar with 99 years of ~10-events/yr windows. |

> **In one sentence:** The 10-day pre-April-15 window and the 5-day post-deadline window show elevated S&P 500 returns consistent with the IRA-contribution folk narrative, but neither survives a Bonferroni-corrected significance test on 99 years of data — positive direction, wrong magnitude of evidence.

## What we tested

The folk claim: US equity returns are systematically elevated in the ~10 trading days before the April 15 federal filing deadline, as individual investors top up IRA contributions or deploy tax refunds, and/or show a distinct pattern in the 5 days immediately after. We test it on **^GSPC daily returns from 1928-01-03 to 2026-06-12** (24,728 days, 99 years) with two honest controls: (1) the unconditional all-day baseline, (2) a same-sized October-15 placebo window (no IRS event). A **Bonferroni correction** (k=2, one per hypothesis) is applied to both p-values.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the folk narrative, the window returns, the April seasonal, the placebo reveal, why the signal cannot be claimed |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t-stats, Welch tests, Bonferroni table, power analysis, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`tax_day/`](tax_day/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
