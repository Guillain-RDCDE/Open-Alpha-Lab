# Study 222 -- Altseason-Rotation

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Best net HAC *t* = +1.05 (dom_change_20d, 40 bps RT); no variant clears the |*t*| >= 2 bar. Strategy underperforms BTC and alt buy-and-hold on a Sharpe basis (net Sharpe 0.45 vs BTC 0.53). |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Net Sharpe 0.45 at 40 bps drops to 0.27 at 80 bps. Max drawdown -89%. Does not beat buy-and-hold. No tradable edge. |
| **Can BTC dominance time the rotation into altcoins?** | ![BUSTED](https://img.shields.io/badge/Alt--season_timing%3F-BUSTED-8b949e?style=flat-square) | One crypto cycle (2020-2026), dominated by the 2021 alt season. The strategy captures that episode but is indistinguishable from luck on a single-cycle sample. |

> **In one sentence:** the long-alt/short-BTC rotation timed by BTC dominance earns a modest gross return in one crypto cycle, but it does not survive transaction costs, fails to beat simpler buy-and-hold benchmarks, and the t-stat on net returns (1.05) is well below the inference bar -- the edge is a mirage.

## What we tested

The "alt season" rotation recipe: when Bitcoin's basket dominance (BTC's share of total market cap, proxied from price x fixed supply) has been *falling*, go long an equal-weighted alt basket (ETH, XRP, ADA, SOL, BNB, DOGE) and short BTC; stay flat otherwise. Two signal variants are tested -- a 20-day dominance *change* threshold (fire when dominance drops >= 2 pp over 20 days) and a 52-week *level* signal (fire when dominance is below its rolling annual median). Transaction costs are applied at 40 bps round-trip per rotation event (and 80 bps for stress-testing). Performance is evaluated vs BTC buy-and-hold and equal-weight alt buy-and-hold.

The study is the explicit *strategy-level* companion to [Study 134 (Bitcoin-Dominance)](../134-bitcoin-dominance/), which tests only the underlying regression signal. Even taking Study 134's weak signal (regression *t* = -1.88) and building the best possible rotation strategy around it, the result is still a mirage once costs, drawdowns, and the buy-and-hold comparison are applied.

Known limits: the panel covers one crypto cycle (~6 years, SOL start-date-limited); the alt basket is survivorship-biased (2024 winners only); the dominance proxy uses fixed rather than live circulating supply.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the alt-season narrative, how the rotation strategy works, why timing doesn't beat holding, the single-cycle fragility |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | full PnL simulation, Sharpe vs benchmarks, cost sweep, HAC t-stats, positive control, survivorship accounting |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`altseason_rotation/`](altseason_rotation/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
