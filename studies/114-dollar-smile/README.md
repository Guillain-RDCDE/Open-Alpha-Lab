# Study 114 — Dollar-Smile

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Predictive weekly IS *t* SPY = **+0.07**, EEM = **+0.39**; OOS R² **−0.96%** (SPY) and **−0.94%** (EEM) — not merely weak, worse than the naive mean. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Momentum signal *t* = −0.04 (SPY), −0.41 (EEM) — a coin flip after costs. No break-even dollar threshold exists. |
| **Contemporaneous or Forecast?** | ![Confirmed](https://img.shields.io/badge/Coincident--only-8b949e?style=flat-square) | Contemporaneous weekly R² **8.5%** (SPY, *t* = −4.65) and **17.4%** (EEM, *t* = −7.54) — the dollar moves *with* equities, not *before* them. |

> **In one sentence:** the "strong dollar is bad for stocks" claim is true as description (contemporaneous R² 8–17%) but fails completely as a forecast — predictive IS *t* < 0.5, OOS R² negative, confirming the same pattern as Dr-Copper: a coincident risk-factor link masquerading as a tradable timing signal.

## What we tested

The folk macro claim: *a rising US Dollar Index (DXY) is bad for equities, and even worse
for EM.* The steelmanned version asks whether last week's dollar change forecasts next
week's SPY/EEM return — the signal you would need to trade it. We pull 22 years of daily
data (DX-Y.NYB, SPY, EEM; 2004-2026, 1,170 weekly periods), run an in-sample predictive
regression with Newey-West HAC *t*-statistics, expanding-window Goyal-Welch OOS R², and
a naive dollar-momentum timing strategy vs a random-direction coin. We separate the
**contemporaneous** link (the dollar and equities moving together right now — real and
robust) from the **predictive** link (the dollar leading equities — needed for trading,
and absent). A deterministic synthetic panel with tunable predictive and contemporaneous
knobs serves as the positive control.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the DXY story in plain language, the contemporaneous vs forecast split, why the chart fools you, why OOS kills it |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC *t*-stats, Goyal-Welch OOS R², DM test, horizon sensitivity (weekly vs monthly), EEM vs SPY, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`dollar_smile/`](dollar_smile/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
