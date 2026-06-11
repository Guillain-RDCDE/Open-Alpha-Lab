# Study 46 — Bargain-Bin 🛒

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do cheap stocks beat expensive ones? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Regime-dependent. Value worked strongly **before 2007** (+8%/yr) — but over 2000–2026 as a whole, value has **trailed growth** (IVE−IVW −1.1%/yr, Sharpe −0.12), and value's own Sharpe (0.54) is below growth's (0.59). |
| **Tradability** — can you actually hold it? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | A premium that needs a **2007–2020 "lost decade"** of −5%/yr (Sharpe −0.7) — thirteen years underwater — is not investable. You'd have abandoned it long before any payoff. |
| **"A dependable premium"?** | ![Not supported](https://img.shields.io/badge/Not_supported-8b949e?style=flat-square) | Every pair tells the same story: strong pre-2007, brutal 2007–2020, flat since. A regime bet, not a reliable factor. |

> **In one sentence:** the value premium isn't a fraud — it paid handsomely in the early 2000s — but it's a *regime-switching bet*, not the dependable factor it's sold as: on tradable proxies it has trailed growth since 2000, value's Sharpe sits *below* growth's, and its one strong era was followed by a 2007–2020 lost decade deep enough to break any holder.

## What we tested

**Value** — buy cheap (high book-to-market) stocks, the HML factor, the other half of Fama-French ([paperswithbacktest](https://github.com/paperswithbacktest/awesome-systematic-trading) lists it at Sharpe `0.526`). Rather than reconstruct book-to-market, we test the premium the way an investor actually accesses it: **tradable value/growth ETF pairs** (IVE/IVW, VTV/VUG, RPV/RPG), 2000–2026. We measure the HML spread and its Lo (2002) t-stat, compare each leg's standalone Sharpe, and — the crux of the modern debate — split the sample into **pre-2007 / the 2007–2020 lost decade / 2021-on** to ask whether the premium is dependable or regime-bound. The offline control is a synthetic world whose value premium switches regime (and a null).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "buy cheap" is sound *and* lost to growth for a generation, and what a lost decade does to an investor |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the HML spread with its Lo t-stat, value-vs-growth Sharpe, the three regimes, the lost-decade drawdown |

The fingerprinted real-data run (value/growth ETF pairs, 2000–2026, fp `79a51e0c7a87`) is in [docs/results.md](docs/results.md). Reproduce via [examples/verify.py](examples/verify.py) (`--fetch` to download); the offline machinery proof runs on the synthetic world in [bargain_bin/data.py](bargain_bin/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
