# Study 306 — Crack-Spread

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Yesterday's crack-spread change does **not** forecast today's refiner-basket return: predictive HAC *t* = **−0.51**, R² ≈ 0.0001, and the slope flips sign across halves. The only real link is *same-day* (*t* = +2.06) — unforecastable by construction. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | No crack-timer beats always-long buy-and-hold on the **active** return (best gross +1.7%/yr at *t* = +0.27, negative at any cost). The timers' "higher Sharpe" is pure cash-drag, inside the buy-and-hold bootstrap band. |
| **Coincident ≠ predictive?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | The crack spread *is* the refiner's margin in real time, so it tracks the stock; mistaking that co-movement for a timing signal is the entire illusion. |

> **In one sentence:** the crack spread is a faithful mirror of refiner profitability *today* — which is exactly why it can't tell you anything about *tomorrow*, and a crack-timer earns nothing buy-and-hold didn't.

## What we tested

Energy-desk lore says a **fat or rising refining margin (the crack spread) predicts refiner stocks** — so you can time a Valero/Marathon/Phillips-66 basket from the [3-2-1 crack](https://www.cmegroup.com/) (2 gasoline + 1 heating oil − crude). We take that literally on daily futures (RB=F/HO=F/CL=F) and an equal-weight total-return refiner basket since 2012: we run the **predictive** regression the whole claim rests on — does *yesterday's* crack change forecast *today's* basket return? — with a Newey-West *t*, and race a level-regime and a momentum crack-timer against always-long buy-and-hold on the cost-charged **active** return. A deterministic synthetic tape with a tunable lead-lag is the positive control, and it doubles as the cleanest demonstration of why a same-day correlation proves nothing about timing.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a crack spread is, why it mirrors refiners, and why a mirror can't see the future |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | predictive vs coincident HAC *t*, the cash-drag Sharpe illusion, cost sweep, the synthetic lead-lag control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`crack_spread/`](crack_spread/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
