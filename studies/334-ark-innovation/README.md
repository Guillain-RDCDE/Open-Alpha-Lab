# Study 334 — ARK-Innovation 🚀

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | Only the slow 50/200 trend overlay clears the bar (HAC *t* = **+2.09**, bootstrap CI low **+0.22 bps**) — and only by going to cash through the crash; time-series momentum (*t* = +1.36) and mean-reversion (*t* = +0.33) are **None**. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The trend overlay's excess Sharpe over ARKK buy-and-hold is **+0.07**, and *far* below a plain QQQ hold (0.60 vs **0.93**). The only "edge" is dodging a −81% drawdown a QQQ investor never took. |
| **A buy-the-top machine?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | ARKK rose **+737%** to its 2021 peak, fell **−81%**, and ended **+13.8%/yr** — but the money arrived near the top, so the *dollar-weighted* return the average investor earned was deeply negative (Morningstar: **~$7–10bn** of investor wealth destroyed). |

> **In one sentence:** ARKK's price did fine over the full ride, but almost nobody held it for the full ride — the cash piled in at the 2021 top and rode the −81% bust down, so chasing Cathie Wood was a buy-the-top machine, not a momentum win.

## What we tested

Cathie Wood's **ARK Innovation ETF (ARKK)** was *the* retail-innovation-momentum trade of 2020-21: a concentrated, high-beta "disruptive innovation" basket that rose ~150% in a year and pulled in a flood of retail and adviser money. The bull case, at full strength: innovation is a durable momentum theme and a star manager riding it compounds wealth far faster than the index. We take ARKK's real daily tape since its 2014-10-31 inception (QQQ and XLK as innovation-beta benchmarks) and ask two things: (1) is there a tradable **trend / momentum / mean-reversion** signal in its own price, and (2) the question that actually decides whether the hype paid — the gap between its **time-weighted** return (buy-and-hold) and its **dollar-weighted** return (what the average dollar earned). A deterministic synthetic hype-cycle tape, with tunable boom-bust and performance-chasing flows, is the positive control for the behaviour-gap machinery.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the boom-and-bust arc, why a +13.8%/yr fund still lost its investors money, the buy-the-top trap in plain English |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the overlay race with HAC *t* and block-bootstrap CIs, excess-vs-excess Sharpe, the dollar-weighted IRR machinery, the synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`ark_innovation/`](ark_innovation/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
