# Study 587 — NFT-Floor-Beta 🖼️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

**Do blue-chip NFT floor prices lead crypto risk appetite — or just follow ETH with extra noise?**

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does NFT floor momentum *predict* forward ETH? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | On the synthetic null world floor momentum forecasts forward ETH at Newey-West HAC *t* **−0.47** (placebo *p* **0.73**), R² **0.0005**; **+0.41** after controlling for ETH's own momentum; |HAC *t*| < 1.5 at every horizon. Floors are a **lagged high-beta echo** (beta on lagged ETH **1.61**, lead-lag corr peaks at lag **−1**). **Synthetic-only** (no free NFT floor tape) → capped below REAL on data availability. |
| **Tradability** — does a floor-momentum ETH overlay pay? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Long-ETH-when-floor-momentum-positive is **gross +1.4%/yr → net −3.1%/yr** after 10 bps/turnover (Sharpe **−0.05**). It is just timed partial ETH beta — nothing to harvest. |
| **"NFT floor leads crypto?"** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | Floors **follow** ETH by a day and amplify it (beta 1.61, *t* 49.7, corr 0.79 at lag −1) with **zero** lead at every positive lag. It is high-beta noise, not a leading indicator. |

> **In one sentence:** blue-chip NFT floors don't *lead* crypto risk appetite — they're a lagged, ~1.6× high-beta echo of ETH (floor momentum forecasts forward ETH at HAC *t* −0.47, placebo *p* 0.73, and a floor-timed ETH overlay loses money net), and because no clean free NFT floor tape exists the signal can't rise above `NONE` on data availability alone.

## What we tested

The claim (crypto alt-data folklore; Dowling 2022; Ante 2022): **NFT floor-price momentum is a
risk-appetite signal that leads crypto returns.** We build a deterministic synthetic world where the
NFT floor index is a *lagged high-beta echo* of ETH (`floor_t ≈ β·eth_{t−1} + noise`, the null:
floors follow, they don't lead), with a single knob (`lead_alpha`) that can *plant* a genuine lead.
The engine measures: a **predictive regression** of forward ETH return on floor momentum with a
**Newey-West HAC** *t*, a **lead-vs-follow** decomposition (lagged beta + lead-lag cross-correlation),
a control for ETH's own momentum, a **block-shuffle placebo** null, a long/flat ETH overlay with
gross-and-net costs, a five-horizon robustness sweep, and a **seed-robust (25-seed) synthetic
positive control** proving it catches a planted lead and stays flat at the null. **No clean, free,
no-key NFT floor tape exists** (marketplace APIs are key-gated, wash-traded, survivorship-ridden), so
this is **synthetic-only** — the data-availability limitation is named on the SIGNAL axis and caps it
below REAL, like the desk's [273 Lego](../273-lego-returns/) / [275 Whisky](../275-whisky-cask/) /
[276 Sneaker](../276-sneaker-resale/) alt-data studies.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what an NFT floor is, why "floors predict crypto" sounds smart, and why floors just *echo* ETH a day late |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the predictive HAC-*t*, the lagged-beta & lead-lag decomposition, the placebo null, the horizon sweep, gross/net overlay costs, and the seed-robust synthetic positive control |

The fingerprinted headline run (synthetic null world, seed 587, 1500 days, series fp `01ef0bf578c2`,
as-of 2026-06-30) is in [docs/results.md](docs/results.md); the offline machinery runs on the
deterministic world in [`nft_floor_beta/data.py`](nft_floor_beta/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`nft_floor_beta/`](nft_floor_beta/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
