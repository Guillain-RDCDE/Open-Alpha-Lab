# Study 103 — Turtle-Trader

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | System 1 breakout: **+327.7 bps/trade**, HAC *t* = **+6.97** vs random entry +28.7 bps; long-only *t* = **+11.33**. The new-high timing genuinely matters. |
| **Tradability** — does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | Edge decayed ~40% post-2003 publication (703→303 bps), short entries reliably lose on ETF basket (−534 bps, *t* = −5.5), and it requires patience through multi-year drawdowns. Long-futures only is the survivable version. |
| **Beats a random entry?** | ![Confirmed](https://img.shields.io/badge/Beats_random_entry%3F-Confirmed-8b949e?style=flat-square) | Breakout +327.7 bps vs random +28.7 bps — the *timing* of entry at a new 20-day high is the edge, not just the holding period. |

> **In one sentence:** the legendary Turtle Donchian channel breakout captures a real momentum premium on the long side (t = +11), but shorts are a structural trap on upward-drifting ETF baskets, the edge halved after the rules were published in 2003, and harvesting it requires years of patience through deep, prolonged drawdowns — real but fragile.

## What we tested

The recipe: Richard Dennis trained the "Turtle Traders" on a pure Donchian channel breakout — buy when the daily close sets a new 20-day high, exit when it drops below the 10-day low (System 1); a 55-day/20-day long-term version (System 2). Both long and short, in both directions. We run it across a basket of eight liquid trend-following instruments (SPY, GLD, TLT, USO, UUP, QQQ, IEF, DBA) on daily bars from 1993 to 2026, pin every trade against a **random-entry control** (same trade count, random timing), split pre/post the 2003 publication of the rules, and sweep round-trip costs. A deterministic synthetic daily tape with tunable AR(1) momentum serves as the positive control.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the Turtle legend, the long-vs-short trap, the random-entry comparison in plain language, the post-publication decay story |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-instrument HAC *t*, pre/post-2003 decay, direction breakdown, the synthetic positive control, cost sweep |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`turtle_trader/`](turtle_trader/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
