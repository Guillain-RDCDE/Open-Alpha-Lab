# Study 498 — Dual Thrust 🚀

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the breakout pay? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The "buy the upper-trigger break" rule does **not** beat a drift-matched **random-entry** baseline — it loses to it at **every** horizon: breakout − random = **−33.1 / −34.6 / −55.2 / −147.1 bps** at 5/10/20/60 days, and the breakout-vs-random Welch *t* is **significantly negative** (−3.39 / −2.63 / −2.80 / −4.48, all *p* ≤ 0.01). The fine one-sample *t*'s (20d **+4.84**, 60d **+5.42**) are **pure beta** — and *mistimed*: you buy after the thrust, paying a worse price than a dart. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The rule earns *less* drift than a random day and pays costs on every trigger. You'd capture the same drift more cheaply — and more fully — by **holding the index**. Nothing to scale. |
| **"Does the range breakout forecast?"** | ![Busted](https://img.shields.io/badge/Range_breakout_forecasts%3F-Busted-8b949e?style=flat-square) | Permute which day each Range belongs to (scrambled-Range placebo) and the result is intact: **94%** of volatility-nonsense bands match or beat the real one (*p* = **0.942**). The specific Dual-Thrust geometry carries no information. |

> **In one sentence:** Dual Thrust looks compelling because breakouts precede big green candles and indices drift up — encode it mechanically (Chalek's N = 5, k = 0.5, no parameter-fishing) and fire the upper-trigger breakout 1 251 times across 5 indices over 21 years, and it **loses to buying on random days at every horizon** (Welch *t* significantly negative; the geometry placebo leaves it untouched, *p* = 0.94): all tide, *late* — you buy after the move.

## What we tested

We encode the tightest mechanical version a proponent would accept. From the **prior N = 5 bars** we form the Dual-Thrust **Range** = max(HH−LC, HC−LL) (known at the open, no look-ahead), draw a buy line at **open + 0.5·Range**, and fire a long on the first close **above** it, entered at the **next close** (one documented lag); we then measure the forward 5/10/20/60-day return on SPY, QQQ, IWM, DIA and GLD (yfinance daily total-return, 2005→2026). The Signal axis is **breakout vs a drift-matched random-entry baseline** (a Welch *t*) — the only honest test on an upward-drifting tape — plus a **scrambled-Range placebo** that destroys the opening-range geometry while keeping the Range marginal and the *k* coefficients. Tradability charges costs on every trigger. A deterministic synthetic control with a *planted* breakout-continuation proves the detector is live (edge 0 → Welch *t* ≈ 0 vs random; planted continuation → Welch *t* = +3.25), so the *negative* real-tape result is a genuine "worse than nothing".

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what an opening-range breakout is, why buying *after* the thrust on a rising market still loses to random, the breakout-vs-random race, and the Range scramble — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | mechanical Dual-Thrust bands, one-sample HAC *t* vs the beta trap, the random-entry Welch test (significantly negative), the scrambled-Range placebo, per-ticker deltas, costs, and a synthetic planted-continuation control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`dual_thrust/`](dual_thrust/). Range is the trailing N = 5-bar max(HH−LC, HC−LL), shifted one bar (known at the open); entry is the next close (one lag). Basket is surviving liquid ETFs — but this is a single-instrument breakout study, so the random-entry baseline neutralizes the drift/survivorship. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
