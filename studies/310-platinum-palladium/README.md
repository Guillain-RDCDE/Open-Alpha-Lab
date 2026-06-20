# Study 310 — Platinum-Palladium

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Gross **−101 bps/trade**, HAC *t* = **−0.32** on 26 trades over 16 years — on the *wrong* side of zero; bootstrap CI [−705, +431] bps swamps it. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | A 73% win-rate masks a **negative** expectancy (skew −2.23): rare regime-inversion blow-ups eat the many small wins. Costs only make it worse. |
| **Ratio mean-reverts?** | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | Full-sample ADF rejects a unit root (p = 0.011), but the half-life is **475 trading days** (~22 mo), the pair is *not* cointegrated (p = 0.70), and the round-trip was a one-off 2018–2022 demand regime change — not a tradable restoring force. |

> **In one sentence:** platinum and palladium are both autocatalyst metals, but their price ratio was hijacked by a one-off 2018–2022 palladium demand boom (ratio swung 0.31–3.70), so the z-score reversion trade earns a *negative* −101 bps/trade at t = −0.32 with a high-win-rate / fat-loss profile — there is no tradable mean reversion here.

## What we tested

The folk trade says the platinum/palladium ratio always snaps back: when palladium gets rich relative to platinum, short palladium and buy platinum, since both metals are substitutable in catalytic converters and their relative price "should" anchor. We implement this as a z-score rotation between PL=F and PA=F futures (252-day rolling window, enter at |z| > 1.5, exit at |z| < 0.5), compare it against a **random-direction control** on identical entry/exit dates, test whether the ratio is actually stationary and cointegrated (ADF, Engle-Granger, OU half-life, block-bootstrap CI), and benchmark against buy-and-hold of each metal. A deterministic synthetic OU pair with tunable mean-reversion speed is the positive control — it confirms the engine harvests an edge when reversion is genuinely planted.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the ratio's wild inversion, the high-win-rate-but-losing trap, why "both are catalyst metals" wasn't enough |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | ADF + Engle-Granger + OU half-life, HAC t-stat & bootstrap CI vs a random control, cost sweep, synthetic positive control sweep |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`platinum_palladium/`](platinum_palladium/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
