# Study 478 — Percentage Price Oscillator (PPO) 📊

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the crossover forecast? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The bullish PPO crossover does **not** beat a drift-matched **random-entry** baseline: crossover − random = **−14.2 / −24.9 / −63.6 / −44.4 bps** at 5/10/20/60 days — *negative at every horizon*, and **significantly so at 20 days** (Welch *t* = **−3.13**, *p* = 0.002, the wrong sign of "significant"). The big one-sample *t*'s (20d **+5.98**, 60d **+7.86**) are **pure beta** — the upward drift every long-only entry inherits. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No residual edge once the free drift is removed — the crossover actually *under-times* relative to a coin flip, and costs only deepen the hole. You'd capture the same drift more cheaply by **holding the index**. Nothing to scale. |
| **"Does normalizing MACD add edge?"** | ![Busted](https://img.shields.io/badge/Normalizing_adds_edge%3F-Busted-8b949e?style=flat-square) | The **raw MACD** crossover and the **normalized PPO** crossover lose to random by statistically indistinguishable margins at every horizon (Δ within a few bps), and scrambling the crossover's sign structure (placebo) leaves the result intact: **99.8%** of nonsense crossovers match or beat the real one (*p* = **0.998**). Dividing by EMA26 changes the *units*, not the *timing*. |

> **In one sentence:** The PPO is just MACD wearing a percent sign — encode the famous "buy the bullish crossover" rule mechanically and fire it **1082 times** across 5 indices over 21 years, and it **loses to buying on random days** at *every* horizon (significantly worse at 20 days), the normalization buys comparability-not-edge (raw MACD is identical), and scrambling the crossover leaves it untouched (*p* = 0.998): all tide, no tool.

## What we tested

We encode the tightest mechanical version a proponent would accept. The PPO = 100·(EMA12 − EMA26)/EMA26 with a 9-EMA signal line; a long fires the first bar the PPO crosses strictly **above** its signal, entered at the **next close** (one documented lag), and we measure the forward 5/10/20/60-day return on SPY, QQQ, IWM, DIA and GLD (yfinance daily total-return, 2005→2026). The Signal axis is **crossover vs a drift-matched random-entry baseline** (a Welch *t*) — the only honest test on an upward-drifting tape — plus a **shuffled-sign placebo** that destroys the crossover structure while keeping the |PPO − signal| marginal. The thesis axis runs the **raw MACD** crossover side-by-side to ask whether normalization adds timing power. Tradability charges costs on every crossover. A deterministic synthetic control with a *planted* post-crossover continuation proves the detector is live (edge 0 → *t* = 1.09; planted continuation → *t* = +19.21), so the flat real-tape result is a genuine "nothing there".

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the PPO is, why a crossover on a rising market always looks good, the crossover-vs-random race, MACD-vs-PPO, and the sign scramble — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the oscillator math, one-sample HAC *t* vs the beta trap, the random-entry Welch test, the raw-MACD comparator, the shuffled-sign placebo, per-ticker deltas, costs, and a synthetic planted-continuation control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`ppo/`](ppo/). PPO/MACD use the standard 12/26/9 EMAs (`adjust=False`); the oscillator is read on the close of *t*, entry is the next close (one lag). Basket is surviving liquid ETFs — but this is a single-instrument trend study, so the random-entry baseline neutralizes the drift/survivorship. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
