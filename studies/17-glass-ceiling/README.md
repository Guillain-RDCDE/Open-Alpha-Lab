# Study 17 — Glass-Ceiling 🪟

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The breakout win rate is a coin flip: **51.8%** on the driftless null (95% CI **[48.2, 55.3]**) and **[35, 56]** on real BTC-USD — every interval straddles 50%. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | At 1:1 the break-even win rate is ~**50.9%** at just 2 bps; the spread paid on *both* legs turns the coin flip net-negative. On real BTC-USD: **−0.11 R per trade**. |
| **Do the filters help?** | ![Not supported](https://img.shields.io/badge/Not_supported-8b949e?style=flat-square) | Staircase + volume + clean-trend conditioning adds **no win-rate lift** beyond noise; it only thins the sample. On real tapes the "A-grade" subset is **1–9 trades** — selection illusion, not edge. |

> **In one sentence:** a 1:1 breakout bracket is a symmetric coin-flip whose only reliable feature is the spread it pays twice — and the three "optimal environment" filters add nothing but the illusion of selectivity on a shrinking sample.

## What we tested

Koroush AK's viral *["My Breakout Trading Strategy"](https://x.com/KoroushAK)* (309k views): go long when price clears resistance on **two confirming 1-minute closes**, stop at the swing low (floored at 1%), take profit at **1R**, and only take trades in an "optimal environment" — a slow **staircase** approach, **building volume**, and a **clean trend** (few 30-SMMA crossovers). We mechanize the setup *charitably* (a clean rolling-high level, confirmation before entry, a real swing-low stop) and resolve every bracket trade-by-trade with pessimistic intrabar fills. The core runs on a synthetic minute tape with the post-breakout answer **baked in** — a driftless null (coin flip *by construction*) plus continuation, exhaustion and grind-gated tapes that prove the test has power — then we sanity-check on real BTC-USD / SPY / QQQ 5-minute bars.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a 1:1 bracket is a coin flip, where the spread quietly eats it, and how a filtered screenshot manufactures a fake win rate |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the ±1R expectancy identity, Wilson intervals on the win rate, the cost-in-R sweep and break-even line, and a filter-lift test with baked-in power checks |

Sources & literature map: [docs/references.md](docs/references.md).

---

*Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
