# Study 32 — Rip-Tide 🌊

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Gross-of-cost Sharpe **0.08** on 18 liquid futures, 2000–2026 — indistinguishable from zero, negative in all three sub-periods. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Turnover **1.38**/day ⇒ **break-even cost 0.24 bp**, below even the ~1 bp it costs to trade ES. Net @2 bp: Sharpe −0.56, −88% DD. |
| **Rescue by slowing down?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | Holding 1→21 days cuts turnover >10× but the *net* Sharpe never crosses zero — the gross edge fades as fast as the cost. |

> **In one sentence:** short-term "fade the recent move" reversion is a real anomaly in illiquid single stocks, but on the world's deepest futures there is no premium to begin with — and what little there is, daily turnover eats long before you can trade it.

## What we tested

Kakushadze & Serur's *151 Trading Strategies* §10.3 catalogues the **contrarian futures** play: fade
each market's recent move — short what just rose, buy what just fell — the exact mirror of the §10.4
trend-following entry that [Study 31 (Trade-Winds)](../31-trade-winds/) found to be a real (if fragile)
premium. We run it on the **same 18 liquid continuous futures** (equities, rates, commodities, FX,
2000–2026) with the **same** equal-risk, vol-targeted machinery — flipping only the sign of the signal —
so any difference in the verdict is the strategy, not the universe. The offline core proves the
apparatus on a synthetic Ornstein-Uhlenbeck mean-reverting panel; the verdict is measured on the market.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "buy the dip" is real for stocks but not for oil futures, and how turnover quietly kills it |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | gross-vs-net Sharpe, the cost wall & break-even, the holding-period rescue, horizon & sub-period decay |

The fingerprinted real run is in [docs/results.md](docs/results.md); the beat-7 "slow it down" rescue
(`BUSTED`) is worked in [docs/extension.md](docs/extension.md).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
