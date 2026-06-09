# Study 21 — Fools-Gold ✨

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the crossover informative? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | The golden-minus-death next-day spread is significant *only* on the trending US equity indices (SPY Newey–West *t* = **+2.7**); across **8** liquid ETFs the timing book beats buy-and-hold on just **3** (median Sharpe gain **−0.01**) — a coin flip. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Where it "works" the edge is mostly *less exposure* (SPY beta **0.52**, alpha *t* = **+2.2**) — a constant cash blend at the same average exposure captures most of the calm — and net of every whipsaw it *loses* on most assets. Even on the S&P it lagged buy-and-hold through the post-2009 bull (sat in cash on the dips). |
| **Generalises?** | ![Not supported](https://img.shields.io/badge/Not_supported-8b949e?style=flat-square) | It beats buy-and-hold on **76%** of (fast, slow) parameter pairs for SPY and **0%** for another index — a strong trend on one cherry-picked asset, found by a filter, not an edge. |

> **In one sentence:** the most-quoted chart pattern on finance TV is a crude trend filter that shines on the one secularly-trending index everyone cites, mostly captures lower exposure, and falls apart across assets and parameters — fool's gold.

## What we tested

The desk's fourth idea from Kakushadze & Serur, *151 Trading Strategies* (strategies **§3.11–3.13**, moving averages). The steelman, at full strength: the **golden cross** — when the 50-day moving average crosses above the 200-day, the trend is up and you go long; the **death cross** (below) means step aside. We prove the apparatus on a synthetic close with a *baked-in* persistent trend (and a driftless random-walk null that must — and does — yield nothing), then run the long/flat 50/200 crossover across a basket of liquid ETFs vs simply buying and holding.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the story in plain language: why a long/flat rule *looks* calm for free, why it only "works" on the S&P, and why its record is one dodged crash |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the machinery: the golden−death HAC *t*, alpha & beta vs buy-and-hold, the risk-matched cash-blend control, and the parameter grid |

The real run — every fingerprinted, as-of'd ETF number — is in [docs/results.md](docs/results.md); the **beat-7 worked complement** (the *parameter-robustness test* — % of (fast, slow) pairs that beat buy-and-hold, asset by asset) is in [docs/extension.md](docs/extension.md). Reproduce offline via [examples/run_synthetic_demo.py](examples/run_synthetic_demo.py); on the real tape via [examples/verify.py](examples/verify.py) and [examples/extension.py](examples/extension.py) (`--fetch` once to populate the close cache).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
