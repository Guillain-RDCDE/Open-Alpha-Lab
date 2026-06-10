# Study 34 — Aftershock 📈

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does price keep drifting after an earnings surprise? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | Yes — post-earnings-announcement drift is one of the most durable anomalies in finance (Ball-Brown 1968; Bernard-Thomas 1989). Our synthetic control recovers it (gross Sharpe **+3.73**, CAGR **+10.2%**) while the null is flat (**+0.40**), and the drift-decay curve has the textbook rise-then-flatten shape. |
| **Tradability** — does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | The drift is *small* and **concentrates in the illiquid, small, high-cost names** (Chordia et al. 2009) while shrinking toward zero in the liquid stocks you can trade at scale — and it has attenuated since publication. Real, but thin where you can actually harvest it. |
| **Real-tape run?** | ![Pre-reg](https://img.shields.io/badge/Pre--reg-8b949e?style=flat-square) | The real measurement is **pre-registered and pending** an earnings-history fetch: no free source gives the *years* of reported-earnings dates + surprises a credible PEAD cross-section needs (yfinance exposes only ~6-8 quarters). Apparatus, protocol and mirage-line are fixed; one `--fetch` away. |

> **In one sentence:** post-earnings drift is a real, decades-documented premium that our synthetic control and the literature both confirm — but it is small and lives in exactly the illiquid names that cost the most to trade, so its tradability is `FRAGILE`, and the real-tape measurement is pre-registered and pending an earnings-history source.

> ⚠️ **Real run pending a fetch.** A credible PEAD backtest needs years of earnings dates + surprises, which no free feed supplies here — so this study ships the desk's *pending-fetch* pattern (cf. [Study 27](../../27-steamroller/)): a validated offline synthetic control, a literature-grounded verdict, and a [docs/results.md](docs/results.md) that states the real run is pre-registered. Wire an earnings feed and `python examples/verify.py --fetch` writes the fingerprinted numbers. Reproduce the core offline via [examples/run_synthetic_demo.py](examples/run_synthetic_demo.py).

## What we tested

The desk's idea from Kakushadze & Serur, *151 Trading Strategies* (strategy **§3.2**, earnings momentum / PEAD). The steelman: a stock under-reacts to its earnings *surprise*, so the price keeps drifting in the surprise's direction for weeks after the announcement (Ball-Brown 1968; Bernard-Thomas 1989) — and a book that goes long positive-surprise names and short negative-surprise names, rolled as earnings land, harvests that drift. We prove the engine on a synthetic stock panel where every surprise carries a *known* decaying post-event drift (and a null where the same surprises are pure noise), run the dollar-neutral long-positive/short-negative-surprise book, and reproduce the Bernard-Thomas drift-decay curve.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the story in plain language: why prices drift after earnings, the aftershock, and why the only slice you can trade cheaply has almost no drift |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the machinery: gross-vs-net Sharpe on control vs null, the cost wall & break-even, the drift-decay curve, the holding-period sweep |

The real run — every fingerprinted, as-of'd number — is pre-registered in [docs/results.md](docs/results.md) (after one `--fetch`); the **beat-7 worked complement** (the drift-decay curve + holding-period sweep) is in [docs/extension.md](docs/extension.md). Reproduce offline via [examples/run_synthetic_demo.py](examples/run_synthetic_demo.py); the real-tape hook is [examples/verify.py](examples/verify.py) (`--fetch`).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
