# Study 435 — Guppy Multiple MA 🎏

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the ribbon carry an edge? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The GMMA ribbon-cross excess Sharpe is **+0.301** (HAC *t* = **+1.86**, below the 2-bar). It is **significantly worse than buy-and-hold** (race *t* = **−2.10**), no better than a random in/out schedule (*t* = −0.50), and a **single 50-day EMA out-Sharpes it** (+0.337). The permutation placebo lands at **p = 0.969** — the real signal beats only 3% of random rotations. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | There is no edge to charge costs against. **Even gross** the excess Sharpe (+0.315) trails buy-and-hold (+0.431); the long/short version is sharply negative. The lower drawdown is just reduced exposure — twelve moving averages dressed as a ribbon are one trend filter with extra steps. |
| **"Does the ribbon reveal conviction?"** | ![Busted](https://img.shields.io/badge/Reveals_conviction%3F-Busted-8b949e?style=flat-square) | Guppy's signature claim — a **wide, parallel** ribbon marks a high-conviction continuing trend — performs *worst*: gating the long leg on ribbon width **lowers** Sharpe (+0.301 → +0.210), CAGR and the *t*-stat (conviction-vs-ribbon *t* = −1.30). Width is hindsight, not foresight. |

> **In one sentence:** Daryl Guppy's twelve-line twin-ribbon indicator, turned into an honest long/flat timing rule on 33 years of SPY, earns an excess Sharpe of +0.301 (*t* = +1.86) that is **beaten by plain buy-and-hold** (*t* = −2.10), **beaten by a single 50-day EMA**, no better than a coin-flip schedule (permutation p = 0.97) — and its trademark "conviction" width gate makes things strictly worse.

## What we tested

A retail-charting staple: *"Plot two ribbons of EMAs — a fast trader group (3–15 days) and a slow investor group (30–60 days). Buy when the fast ribbon is above the slow one, and trust the trend most when the ribbons are wide apart and parallel — that's conviction."* We take it literally on SPY daily total-return closes (1993–2026): the canonical **ribbon cross** (long when the short-ribbon mean is above the long-ribbon mean) and Guppy's distinctive **conviction gate** (long only when the normalised ribbon spread is wider than its trailing median), each as a long/flat rule with one-day execution lag and 2 bps one-way costs. Every Sharpe is **excess-of-cash**; the ribbon is raced against buy-and-hold, a **random-timing control** (same in-market fraction, random days), and the obvious simpler benchmark — a **single 50-day SMA/EMA** filter — so the "the ribbon is better" claim is actually tested. Inference is a HAC *t*, a return-difference *t* for each race, and a 2,000-draw circular-rotation permutation placebo. A deterministic two-regime synthetic control with a planted trend edge confirms the harness *can* bank a real signal (ribbon beats BH at *t* = +2.35, permutation p = 0.014) — so the real-tape null is evidence, not a broken engine.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the two ribbons are, what "the ribbons reveal conviction" means, why twelve moving averages still equal one trend filter, and why a random in/out schedule keeps pace — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the ribbon-cross and conviction-gate rules, excess-vs-excess Sharpe races with HAC *t*-stats, the single-EMA benchmark, the random-timing control, the rotation permutation placebo, the cost sweep, and the synthetic planted-edge / null positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`guppy_mma/`](guppy_mma/). SPY daily **total-return** closes (`auto_adjust=True`); cash leg a flat 4%/yr proxy; every Sharpe is excess-of-cash. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
