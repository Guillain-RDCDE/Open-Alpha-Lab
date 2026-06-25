# Study 455 — Rising/Falling Three Methods 🕯️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the pattern beat random? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The three-methods trade does **not** beat a drift- and mix-matched **random-entry** baseline: Δ = pattern − random = **+14.1 / +11.6 / −187.3 / −235.5 bps** at 5/10/20/60 days. No positive Welch *t* reaches 2; the **only** result that clears \|2\| is **−2.50** at 20 days (*p* = 0.015) — the pattern does *worse* than random. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Only **44** patterns in 21 years, a signed return that is flat at short horizons and **negative** at 20 days, and a loss to random entries. No edge to scale; costs only deepen the hole. |
| **"Does the consolidation forecast continuation?"** | ![Busted](https://img.shields.io/badge/Forecasts_continuation%3F-Busted-8b949e?style=flat-square) | Scramble the five-candle geometry onto random dates and nothing moves: **54%** of scrambled runs match or beat the real one (*p* = **0.539**). Over 20 days the trend *gives back* the breakout — the opposite of continuation. |

> **In one sentence:** the rising/falling three-methods looks like a clean "pause then resume" continuation, but encode it mechanically (five closed candles, no eyeballing) and fire it 44 times across 5 indices over 21 years, and it **does not beat entering on random days** — at 20 days the signed return is *negative* (the trend gives the breakout back, Welch *t* = −2.50), and the geometry placebo leaves the result untouched (*p* = 0.54): the pause forecasts nothing.

## What we tested

We encode the tightest mechanical version a proponent would accept. The pattern is **five closed
candles**: a long **anchor** candle (body > 1.0 × trailing-20 average body), three **small**
candles (each body < 0.7 × the anchor, held *inside* the anchor's high–low range with a 10% wick
tolerance), and a long **confirm** candle that closes **past** the anchor in the same direction.
Rising → go **long**, falling → go **short**, entered at the **next close** (one documented lag);
we measure the forward 5/10/20/60-day return *signed by the pattern direction* on SPY, QQQ, IWM,
DIA and GLD (yfinance daily total-return, 2005→2026). The Signal axis is **pattern vs a
random-entry baseline matched in count *and* long/short mix** (a Welch *t*) — the only honest
test on a directional rule — plus a **shuffled-date geometry placebo** that destroys the
five-candle shape while keeping the price marginal. Tradability charges costs on every trade. A
deterministic synthetic control with a *planted* post-pause continuation proves the detector is
live (edge 0 → *t* = +0.88; planted continuation → *t* = +3.36), so the flat/negative real-tape
result is a genuine "nothing there".

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a three-methods is, why a directional rule on a drifting market looks good, the pattern-vs-random race, and the date scramble — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | mechanical five-candle detection, one-sample HAC *t* vs the beta trap, the mix-matched random-entry Welch test, the shuffled-date placebo, per-ticker deltas, costs, and a synthetic planted-continuation control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`three_methods/`](three_methods/). The pattern is five closed candles (no look-ahead, no confirmation lag); entry is the next close (one lag); returns are signed by direction (long rising / short falling). Basket is surviving liquid ETFs — but this is a single-instrument candlestick study, so the mix-matched random-entry baseline neutralizes the drift/survivorship. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
