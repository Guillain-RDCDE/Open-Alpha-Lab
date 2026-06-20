# Study 326 — Stablecoin-Depeg 🪙

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The two market-wide depegs point in **opposite directions** — UST −15.4% but USDC **+32.9%** over a [-5,+10] window. Mean CAR **+8.7%** (*positive*, the wrong sign), p = **0.78** against a synthetic control of random windows, and with n = **2** no HAC *t* is even definable. The engine's positive control fires at *t* = −5.7 on a planted crash — there just isn't one here. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | "Sell on the depeg" **underperforms** buy-and-hold by −5.1% before a cent of cost, because it forces you out of exactly the window you wanted to be in (the USDC-week rebound). No edge to charge costs against. |
| **"Depeg = sell signal"?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | A depeg is a *symptom* whose market meaning depends on *why* the coin broke: an endogenous crypto collapse (UST) drags the market down; an exogenous TradFi shock (USDC/SVB) can send it **up**. One bit — "a coin lost its peg" — carries no direction. |

> **In one sentence:** breaking the buck looks like a sell signal only if you remember UST and forget USDC — across the two real market-wide depegs the abnormal return is positive on average, indistinguishable from a random week, and selling on the depeg loses money.

## What we tested

The heuristic, at full strength: *when a major stablecoin loses its peg, it signals systemic
stress and contagion — sell BTC/ETH on the depeg.* We test it as an **event study** on an
equal-weight BTC+ETH daily basket (Yahoo, 2017→) around the two undisputed market-wide
depegs — the **UST/Luna** algorithmic death spiral (May 2022) and the **USDC** break to
$0.88 over the SVB-failure weekend (March 2023). We measure each window's cumulative
abnormal return, benchmark it against a **synthetic control** of randomly-placed non-event
windows, and run a tradable "go to cash after a depeg" rule (one execution lag, one-way
costs × NAV). A deterministic synthetic positive control confirms the engine recovers a
*planted* depeg crash. (Distinct from [Study 295 — Stablecoin-Supply](../../295-stablecoin-supply/),
which tests supply *growth* as a continuous predictor; this is a discrete event study on
peg *breaks*.)

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the two depegs that went opposite ways, why "stress" isn't a direction, why selling on the news cost you money |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | CAR windows, the synthetic-control p-value, the n = 2 inference wall, the tradable rule's two-trade record, the synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`stablecoin_depeg/`](stablecoin_depeg/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
