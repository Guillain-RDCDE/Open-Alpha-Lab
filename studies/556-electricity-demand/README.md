# Study 556 — Electricity-Demand ⚡

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does power-demand growth lead returns? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The **broad market shows nothing** (SPY hot-minus-cold spread Welch *t* **+0.37**, placebo *p* 0.71; its only \|*t*\| ≥ 2, the 12-month slope **−2.77**, is the *wrong* sign and overlap-inflated). **Utilities** carry one borderline reading — XLU's clean 1-month spread **+1.19%/mo** (Welch *t* **+2.13**, placebo *p* **0.032**) — but it **dies ex-COVID** (*t* **+0.96**, *p* 0.34) and the HAC slope is sub-2 (*t* +1.54). Literature-plausible, one COVID-driven reading, not robust. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The "hold when demand is hot, else cash" overlay **underperforms buy-and-hold** on both tapes — gross *and* net (SPY **+6.1%** vs **+10.8%**/yr; XLU **+3.5%** vs **+8.5%**/yr, Sharpe 0.38 vs 0.63 after 5 bps/switch). Acting on the pulse *destroys* return. |

> **In one sentence:** the electricity the economy burns really is a hard-data pulse — but on the tape it's a **coincident, seasonal, already-priced** one: the broad market shows no forward signal (its only significant slope points the *wrong* way), utilities carry a single borderline reading (*t* +2.13) that is **entirely a COVID-2020 artefact** and evaporates to *t* +0.96 the moment you drop that one year, and turning any of it into a position loses to buy-and-hold.

## What we tested

The macro-alt-data folklore says **aggregate electricity demand is the pulse of the real economy**
— factories, data centres, offices, un-spinnable and printed monthly — so demand-growth should
*lead* equity (or utility) returns as a hard-data nowcast of activity. We build **demand-growth
momentum** (year-over-year % change of U.S. total net generation, which strips the huge
summer/winter seasonal), lag it **two months** so it is strictly public (1 for the EIA publication
delay + 1 for the signal→return convention), and test whether hot-demand months precede stronger
forward SPY and XLU returns: a HAC (Newey-West) predictive regression at 1/3/6/12-month horizons,
a hot-vs-cold conditional split with a Welch *t* and a label-shuffle placebo null, an ex-COVID
robustness cut, and a tradable hold-when-hot overlay vs buy-and-hold. (The EIA v2 API is
firewalled in this build, so the demand series is a hardcoded monthly snapshot of the settled
EIA `ELEC.GEN.ALL-US-99.M` net-generation print — public and frozen, caveated on the Signal axis;
the COVID-2020 demand crash is included faithfully.) A deterministic synthetic control with a
*planted* demand→returns link confirms the engine recovers a real edge and can't manufacture one
from noise. *Closest cousin: [385 Jobless-Claims-Momentum](../385-jobless-claims-momentum/) — the
same coincident-echo-as-leader trap, on a different hard series.*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "electricity is the pulse of GDP" means, why the broad market ignores it, and why the one real-looking signal is just the 2020 lockdown — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the HAC predictive regression across horizons, the hot-vs-cold Welch *t* + placebo, the ex-COVID collapse, the overlay vs buy-and-hold, and the seed-robust synthetic positive control |

The fingerprinted real-data run (234 months 2007-2026, demand fp `625867832b4d`, panel fp
`e7d78ee9af2d`, as-of 2026-06-30) is in [docs/results.md](docs/results.md); the offline machinery
proof runs on the deterministic synthetic world in [`electricity_demand/data.py`](electricity_demand/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`electricity_demand/`](electricity_demand/). Demand here is a hardcoded **snapshot** of EIA `ELEC.GEN.ALL-US-99.M` (the settled print, not the real-time vintage), named as such. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
