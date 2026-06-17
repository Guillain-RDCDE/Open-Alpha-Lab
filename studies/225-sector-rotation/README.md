# Study 225 — Sector-Rotation

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Gross HAC *t* = +3.46 over 26 years, but that is equity beta. Active return vs the equal-weight sector basket is only +0.1%/yr (*t* = +0.09) — statistically zero. No lookback (3, 6, 9, 12 months) produces a meaningful active edge. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The strategy underperforms SPY by −3.7%/yr (HAC *t* = −2.12) — a statistically significant drag. Equal-sector weighting is a structural disadvantage vs the cap-weighted index; the momentum sort does not repair it. Any transaction cost widens the gap. |
| **Survivorship?** | ![Named](https://img.shields.io/badge/Survivorship-Named-8b949e?style=flat-square) | XLC (launched 2018) and XLRE (2015) have shorter histories; the early-period universe had 9 original sectors from 1998/1999. Results across all 9 original sectors are similar. |

> **Can you rotate into the right sector for each phase of the business cycle?**

> **In one sentence:** sector rotation by 6-month momentum earns a real absolute return over 26 years — but so does randomly picking 3 sectors each month, and the strategy lags SPY by 3.7%/yr with statistical significance; the "rotation" is just equity beta in a less efficient wrapper.

## What we tested

A staple of practitioner investing (Stovall 1996; Fidelity sector investing): rank the 11 SPDR sector ETFs (XLK, XLV, XLF, XLY, XLI, XLP, XLE, XLU, XLB, XLRE, XLC) by their trailing 6-month total return each month, go **long the top-3**, equal-weight, rebalance monthly. We pit it against three controls — an **equal-weight all-sector basket** (does the sort add anything over just holding all sectors?), **SPY** (does it beat the passive index?), and a **random-rotation control** (does the 6-month signal beat an uninformed picker?) — and sweep transaction costs. A deterministic synthetic panel with a tunable persistent cross-sector drift serves as the positive control: the machinery harvests momentum when we plant it, zero when we don't.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the recipe, the equity-beta trap, why SPY beats it, the random-picker test |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC *t*-stats, active return vs EW/SPY/random, sub-period breakdown, lookback sweep, cost/Sharpe table, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`sector_rotation/`](sector_rotation/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
