# Study 321 — Earnings-Season-Tide 🌊

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | In-window days drift +6.86 vs +3.28 bps/day out — a +3.59 gap, but the HAC *t* on the difference is only **+1.27** and the block-bootstrap CI **[−1.67, +9.17]** straddles zero. No single window clears *t* = 2. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | "Long only in the windows, cash otherwise" earns **+3.5%/yr at Sharpe 0.40** — *below* buy-and-hold's **+10.2%/yr at Sharpe 0.54** — while sitting in cash **79%** of the year. |
| **A separable market-wide tide?** | ![Not supported](https://img.shields.io/badge/Not_supported-8b949e?style=flat-square) | The single-stock earnings premium (PEAD) is real, but it does **not** aggregate into a directional index seasonal — the in-window days are just *some* of the up-drift days. |

> **In one sentence:** the index really does drift a touch faster during peak-earnings weeks, but the gap is inside the noise band and carving it out under-performs simply holding the index — a tide you can see but cannot stand on.

## What we tested

A recurring market-commentary trope says the *whole* market behaves differently during the
few weeks each quarter when most large-caps report — mid-late January, April, July and
October. At full strength: a distinct seasonal drift pulls the aggregate index in those
windows, so you could lean on it. We take that literally on the **SPY total-return** daily
tape (1993–2025): four hard-coded peak-earnings windows, a Newey-West *t* on the in-window
minus out-of-window mean difference, a block-bootstrap CI, a per-quarter breakdown, and a
"long in-window / cash otherwise" overlay raced **excess-of-cash** against buy-and-hold.
The position is **calendar-known, so no execution lag**. A deterministic synthetic tape with
a tunable in-window tide is the positive control.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "earnings season" feels like it should move the market, and what the data actually shows |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the HAC *t* on the difference, the bootstrap CI, the per-window selection trap, the excess-vs-excess race, the synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`earnings_season_tide/`](earnings_season_tide/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
