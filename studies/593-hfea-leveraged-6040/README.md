# Study 593 — HFEA (UPRO/TMF 55/45) 🎢

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the levered 55/45 compound faster than SPY? | ![Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square) | *Real on the ρ < 0 regime · reversed since the flip.* Full period (2002–2026): gap **+5.42%/yr** but only **HAC t = 1.31** — the tape can't certify it. In the negative-correlation era (2002–2021) the gap was **+11.22%/yr at HAC t = 2.98**; since the 2022 corr flip it runs **−19.71%/yr**, and the regime difference is itself significant (**Welch t = 2.23**). |
| **Tradability** — can you deploy it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | Liquid retail ETFs, quarterly costs ≤ 5 bps/yr — costs aren't the problem. The problem: **−70.8% drawdown**, still −32% under water 4.5 years later, Sharpe **0.64 < SPY's 0.68**, and the insurance leg rests on a stock-bond correlation that has been **positive since 2022**. A levered regime bet, not a durable allocation. |
| **Did 2022 falsify the thesis or just bruise it?** | ![Mixed](https://img.shields.io/badge/2022_falsified_it%3F-Mixed-8b949e?style=flat-square) | More than a bruise: the corr flipped (**−0.30 → +0.51**), both legs crashed together (UPRO **−57%**, TMF **−73%**, HFEA **−64.2%** in 2022), the risk-adjusted case died. Less than a falsification: the full tape *including* the disaster still shows $1 → **×46.8 vs SPY's ×12.7**. HFEA compounds faster *when and only when* the ρ < 0 + bond-carry regime holds. |

> **In one sentence:** Hedgefundie's Excellent Adventure really did out-compound SPY (×46.8 vs ×12.7 over 24 years, and the 55/45 pair beats *both of its own 3x legs* — the diversification engine is real machinery), but the full-period edge can't clear t ≥ 2, the Sharpe never beat plain SPY, and 2022 exposed the load-bearing assumption — negative stock-bond correlation — as a regime, not a law: **Mixed, Fragile**.

## What we tested

The Bogleheads "HFEA" recipe at full strength: **55% UPRO / 45% TMF, quarterly rebalanced**, vs SPY and a 60/40 SPY/TLT mix — real fund tapes from 2009, extended to 2002 (TLT inception) with the per-leg daily-leverage identity `3·r − 2·r_bill − fee/252`, its all-in fee calibrated on the real-fund overlap and **validated at daily-return corr 0.998/0.997** (the [study-100](../100-melting-ice) bar). "Compounds faster" is tested as a HAC *t* on the monthly log-return gap; Sharpe races are excess-vs-excess against ^IRX; costs are one-way × NAV traded at each reset. The 2022 autopsy measures the stock-bond correlation flip (−0.30 → +0.51) that crashed both legs at once, and the regime split it defines (ex-ante, mechanism-based — with a Welch *t* on the difference). A seeded synthetic control plants the diversification engine (ρ = −0.6 + bond carry) or removes it (ρ = +0.6, the 2022 world made permanent), 20 seeds averaged. Distinct from [61 — slow-burn](../61-slow-burn) / [100 — melting-ice](../100-melting-ice), which test *single-LETF decay mechanics* — this is the leveraged **portfolio allocation** claim. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what HFEA promised, the ×46.8 mountain and the −71% cliff on the same chart, why the bond leg was supposed to be insurance, and what 2022 did to it — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | synthesis calibration/validation, the HAC log-gap race, the mechanism-based regime split with a Welch difference test, the 2022 correlation autopsy, cost sweep, and the planted/removed diversification-engine control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`hfea_leveraged_6040/`](hfea_leveraged_6040/). No forecast signal — the rebalance calendar is known in advance; resets trade at the quarter-end close and earn from the next session (the one documented lag). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
