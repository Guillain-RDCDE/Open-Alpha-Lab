# Study 397 — Hurst-Regime 📐

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the Hurst exponent forecast which style pays? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | No. On SPY the Hurst-gated switch's daily edge over the best static style is a **wrong-signed t = −0.32**, the bootstrap CI straddles zero (p = **0.65**), and a **shuffled regime label does just as well** (placebo p = **0.22**). The synthetic control confirms it's a true null, not a power failure. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The gated book (Sharpe **+0.17**) is beaten by always-revert (**+0.25**) and **−0.49 Sharpe worse than just holding SPY** (**+0.66**) — in **5 of 5** markets. It's a worse buy-and-hold wearing a market-timing costume. |
| **Free lunch?** | ![Busted](https://img.shields.io/badge/Free_lunch%3F-Busted-8b949e?style=flat-square) | Two facts kill the romance: trailing R/S Hurst on real markets is **pinned above 0.5** (84–97% of valid days "trending", so the switch barely switches), and where it *does* switch it carries **no forecasting power**. |

> **In one sentence:** the Hurst exponent is a real, well-defined statistic, but reading a trend-vs-mean-revert *trade* off it is a category error — on 31 years of SPY a rolling-R/S-gated style switch underperforms the best static style **and** plain buy-and-hold, its edge over a benchmark is wrong-signed and indistinguishable from a shuffled regime label (placebo p = 0.22), and the estimator is so upward-biased on year-long windows that it almost never even calls the mean-reverting regime the claim depends on.

## What we tested

The folklore says the **Hurst exponent** *self-diagnoses* a market — H > 0.5 means it's trending (so trend-follow), H < 0.5 means it's mean-reverting (so fade) — and that a rolling-Hurst gate switching between the two styles harvests the right premium in every regime. We estimate a trailing **R/S** Hurst (252-day window) on SPY plus four other liquid markets, build three books — always-trend, always-revert, and the Hurst-gated switch — on a 1-day execution lag net of one-way costs, and ask the decisive question: does conditioning the *style* on H beat the same style mix with the regime label **block-shuffled**? A deterministic synthetic control (a fractional-Gaussian path with a *known* Hurst, plus an alternating-regime path with a planted-edge knob) proves the engine recovers a real Hurst-regime edge when one is planted — and manufactures nothing when it isn't.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the Hurst exponent is, why "above 0.5 = trend, below = fade" *sounds* like a self-driving strategy, and why on real markets the gate is a worse buy-and-hold — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | trailing R/S estimation and its upward bias, the trend/revert/gated books, a block-bootstrap Sharpe-difference test, a regime-label placebo null, per-market and window robustness, and a synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`hurst_regime/`](hurst_regime/). The Hurst estimate is classical **R/S** (transparent, dependency-free); DFA shares the same finite-sample bias and does not rescue the premise. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
