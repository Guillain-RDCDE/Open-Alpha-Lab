# Study 764 -- SOPR 🔗

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | SOPR stretch does not robustly predict next-month BTC returns: HAC *t* = **+1.32**, R^2 ~ 0.01 (n=141); the slope collapses to *t* = **+0.18** in a price-momentum horse race. The band ordering (greed > neutral > capitulation) is monotone in the folk direction but no *t* clears 2 on the real tape. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The ">1 / <1" regime rule **loses -7.3%/yr** to buy-and-hold (HAC *t* = -0.73) and trails at every threshold; a time-shuffle placebo shows it beats a coin-flip schedule yet still can't beat holding. Sitting in cash 38% of the time forfeits part of a 150x. |
| **Single-survivor bias** | ![Named](https://img.shields.io/badge/Single--survivor-8b949e?style=flat-square) | BTC is the surviving moonshot; SOPR is computed from its own on-chain spending against past prices; the regime thresholds are fitted to ~four cycles. All results are conditioned on that survival. |

> **In one sentence:** SOPR has the right *story* -- coins moving in profit above 1, at a loss below 1 -- and a faint monotone grain of truth in the bands, but on the real BTC tape it can't clear HAC *t* = 2, dies in a price-momentum horse race, and the famous ">1 / <1" timing rule *loses* to simply holding, making it a textbook **None / Mirage**.

## What we tested

The Shirakashi (2019) / Glassnode recipe: read **SOPR** (Spent Output Profit
Ratio -- the aggregate "sale price / cost basis" of coins moving on-chain) as a
momentum/regime gauge. SOPR **> 1** = coins sold in profit (bullish, be long);
SOPR **< 1** = coins sold at a loss (capitulation, step aside); the chart-lore
adds that 1 acts as support in bulls and resistance in bears. We join a curated
month-end SOPR series (a **labelled proxy**, digitised from the public Glassnode
"Adjusted SOPR" chart) to the real BTC-USD monthly close and ask three honest
questions: (1) does this month's SOPR stretch predict next month's return in a
HAC regression; (2) does it add anything beyond BTC's own price momentum; (3)
does the literal ">1 / <1" timing rule beat **buy-and-hold** net of 30 bps
one-way costs -- with a time-shuffle placebo and a deterministic synthetic
positive control.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the SOPR story in plain language, why "1 is support" is hindsight, and why stepping to cash in capitulation months loses to just holding |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC predictive regression, price-momentum horse race, per-band forward returns, regime timing vs buy-and-hold, the time-shuffle placebo, cost/lag honesty, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`sopr/`](sopr/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
