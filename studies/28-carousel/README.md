# Study 28 — Carousel 🎠

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the hot sector stay hot? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Strong on our synthetic control (alpha over the basket at HAC *t* ≈ **7.6**) — but on the 11 real SPDR sectors the pure long-short sector-momentum factor is flat-to-negative (**−1.5%/yr**, *t* = **−0.5**). No premium where it counts. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Top-3 rotation earns a Sharpe of **+0.43** — essentially tying the do-nothing equal-weight basket (**+0.45**, gain **−0.02**) while running **7.3×/yr** turnover and concentrating into a handful of sectors. Alpha over the basket **+0.4%/yr** (*t* = **+0.3**): indistinguishable from zero. |
| **Beats the basket?** | ![Not supported](https://img.shields.io/badge/Not_supported-8b949e?style=flat-square) | It beats the equal-weight basket on only **17%** of plausible "how-many-sectors-to-hold" choices — a coin flip; the best (k = 1) is just the luckiest cell, not a robust edge. |

> **In one sentence:** rotating into the hottest sectors is real-looking, active, and clever — and on the actual SPDR sectors it merely re-buys the market with extra turnover and concentration, never beating the basket you'd have owned by doing nothing.

## What we tested

The desk's eleventh idea from Kakushadze & Serur, *151 Trading Strategies* (strategy **§4.1**, sector momentum rotation). The steelman (Moskowitz & Grinblatt, *"Do Industries Explain Momentum?"*, **Journal of Finance** 1999): rank the equity sectors by trailing momentum and rotate into the leaders, on the premise a hot sector stays hot. We prove the engine on a synthetic sector panel where leaders persist by construction (and a no-momentum null), then run the top-3 rotation on the 11 real SPDR sector ETFs — and judge it against the only fair benchmark for a concentrated bet: **holding all eleven sectors equal-weight.**

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the story in plain language: why chasing the hot sector feels smart, why it ties the do-nothing basket, and why the "winning" setup is hindsight |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the machinery: the top-minus-bottom spread, the rotation alpha over the basket with HAC errors, the long-short factor *t*, and the top-k sweep |

The real run — every fingerprinted, as-of'd SPDR number — is in [docs/results.md](docs/results.md); the **beat-7 worked complement** (the *top-k sweep* — rotation beats the basket on only 1 of 6 concentration choices) is in [docs/extension.md](docs/extension.md). Reproduce offline via [examples/run_synthetic_demo.py](examples/run_synthetic_demo.py); on the real tape via [examples/verify.py](examples/verify.py) and [examples/extension.py](examples/extension.py) (`--fetch` once to populate the sector-ETF cache).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
