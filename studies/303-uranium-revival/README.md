# Study 303 — Uranium-Revival ☢️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Trend-following is real in the literature, and our harness banks a *planted* trend at HAC *t* = **+2.85** while staying null on a coin (*t* = −0.75). But **no live URA/URNM tape ships here**, and REAL is earned by the *real* tape — so the honest stamp is Weak, not Real. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | On a realistic boom-bust regime the rule posts **+57.7%/yr** over buy-and-hold (*t* = +3.68) — but that is one crash-dodge on a single thin theme. Cost-robust to 50 bps and entirely a function of *which* rocket you fitted: concentration risk dressed as timing. |
| **Durable trend or hype rocket?** | ![Hype rocket](https://img.shields.io/badge/Hype_rocket-8b949e?style=flat-square) | Hold the rule fixed, redraw the boom-bust, and the HAC *t* swings from ~+1 to ~+5. A *real* edge is stable across draws; this one is a coin flip on which rocket you got. |

> **In one sentence:** trend-following is a genuine *category* of edge, but bolting it onto one hype-driven uranium-miner ETF means "I rode the trend" and "I got lucky on one rocket" produce the *same* backtest — and with no out-of-sample real tape, only one of them repeats.

## What we tested

The thematic-ETF pitch markets the **"nuclear renaissance"** as a trend you simply ride: hold URA / URNM (uranium miners) while price is above its 200-day moving average, step aside on a breakdown, and you "capture the rocket and skip the crashes." Trend-following is a real, heavily-replicated effect (Moskowitz, Ooi & Pedersen 2012) — *across dozens of markets pooled*. We take the rule literally (200-day SMA overlay, one execution lag, costs one-way × NAV, excess-of-cash vs excess-of-cash vs buy-and-hold) and, because **no live uranium tape ships with this study**, we stress it on three deterministic synthetic regimes: a *planted trend* (positive control), a *coin* (null), and a *boom-bust hype rocket* (the realist's case). The point isn't another in-sample uranium backtest — it's to show why a gorgeous one-cycle backtest on a single thematic ETF proves nothing.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the trend pitch, the harness banking a real trend, and the rocket that makes timing *look* like genius |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC inference, block-bootstrap CIs, the excess-vs-excess race, and the single-asset trend trap (same rule, twelve boom draws) |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`uranium_revival/`](uranium_revival/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
