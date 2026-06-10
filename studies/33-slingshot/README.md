# Study 33 — Slingshot 🎯

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | Dollar-neutral, gross-of-cost Sharpe **0.70** with **+0.73 skew** on 467 S&P names, 2010–2026 — a genuine cross-sectional reversal premium (vs Rip-Tide's 0.08 on futures). |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Turnover **0.63**/day ⇒ **break-even 3.31 bp**, inside the realistic S&P cost band — and the premium concentrates in the *least* liquid names where costs are highest. Net @5 bp: −0.36. |
| **Decayed since 2020?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | Net sub-period Sharpe **−0.14 / −0.04 / −0.85** — worst in the HFT era, the competed-away fade Khandani-Lo warned of. |

> **In one sentence:** the same "fade the move" that's worth *nothing* on deep futures is a *real* premium in the single-stock cross-section — but it turns over 63% of the book daily, lives in the least-tradable names, and has decayed, so it's real to measure and a mirage to trade.

## What we tested

Kakushadze & Serur's *151 Trading Strategies* §3.9 catalogues **single-group mean-reversion**: within one
universe, short the names that have outrun their peers and buy the laggards, dollar-neutral. Short-term
reversal is one of the most-documented anomalies in equities (Jegadeesh 1990, Lehmann 1990) — the rent
for providing liquidity to the crowd's overreaction. We run a daily-rebalanced, gross-1 dollar-neutral
book on the **current S&P 500** (467 names, 2010–2026) and ask the only question that matters: does the
real gross edge survive its own turnover? It is the deliberate equity mirror of
[Study 32 (Rip-Tide)](../32-rip-tide/), which found *no* such premium on deep futures.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "buy the laggard, sell the leader" is real for stocks but not for oil, and how turnover quietly kills it |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | gross-vs-net Sharpe, the cost wall & break-even, the horizon knife-edge, the holding-period rescue, the decay |

The fingerprinted real run is in [docs/results.md](docs/results.md); the beat-7 "slow it down" rescue
(a marginal, non-investable +0.10) is worked in [docs/extension.md](docs/extension.md).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
