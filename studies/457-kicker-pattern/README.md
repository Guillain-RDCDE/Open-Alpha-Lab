# Study 457 — Kicker-Pattern 🥾

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the kicker forecast a turn? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Trading the kicker direction does **not** beat a **direction-matched random-entry** baseline: kicker − random = **−37.7 / −55.2 / −53.5 / −20.9 bps** at 5/10/20/60 days, and the kicker-vs-random Welch *t* is **negative at every horizon** (best **−0.15** at 60d). The pattern is a ~50% coin (win 46–54%) with a slightly negative mean — *worse* than entering on random days. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | It fires only **~4×/year per index** (80 events in 21 years across 5 tapes), wins ~50%, averages negative, and loses to darts; costs only deepen the hole. No edge to scale. |
| **"Does the gap-reversal forecast?"** | ![Busted](https://img.shields.io/badge/Gap--reversal_forecasts%3F-Busted-8b949e?style=flat-square) | The gap-scramble placebo is **incoherent**: the only two low-*p* tickers (SPY/QQQ, *p* = 0.002) are noisy small-sample positives that *fail* the random test, while IWM/DIA/GLD give *p* ≥ **0.976**. The gap-in-the-new-direction carries no forecasting information. |

> **In one sentence:** The kicker — two opposite marubozu candles split by a gap, sold as "one of the most reliable reversals" — encoded mechanically and fired 80 times across 5 indices over 21 years is a **50/50 coin with a negative tilt that loses to random entries** at every horizon; scramble the gap geometry and the result is an incoherent mess: a vivid name for two big candles and a gap, not a turn predictor.

## What we tested

We encode the tightest mechanical version a proponent would accept. A **marubozu** is a strong candle (body / range ≥ **0.60** — at the textbook-strict 0.80 the canonical kicker prints only **6 times in 21 years**, itself a finding). A **kicker** at bar *t* is an **opposite-colour marubozu pair** where bar *t* gaps in its own direction past bar *t-1*'s open; it is read on the **close of bar *t*** (using only *t-1* and *t* — no look-ahead). We go **long** a bullish kicker / **short** a bearish one, entered at the **next close** (one documented lag), and measure the signed forward 5/10/20/60-day return on SPY, QQQ, IWM, DIA and GLD (yfinance daily total-return, 2005→2026). Because the rule mixes longs and shorts on a drifting tape, the Signal axis is **kicker vs a direction-matched random-entry baseline** (same long/short mix, same instrument, epoch and hold) — the only honest test — plus a **gap-scramble placebo** that randomises the gap signs while keeping the candle marginal. Tradability charges costs on every print. A deterministic synthetic control with a *planted* kicker continuation proves the detector is live (edge 0 → *t* = +0.02, win 49%; planted continuation → *t* = +10.35, win 79%), so the flat real-tape result is a genuine "nothing there".

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a kicker is, why a reversal rule on a rising market needs a fair baseline, the kicker-vs-random race, and the gap scramble — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | mechanical marubozu+gap kickers, the direction-matched random Welch test, the incoherent gap-scramble placebo, per-ticker deltas, costs, and a synthetic planted-continuation control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`kicker_pattern/`](kicker_pattern/). A marubozu is body/range ≥ 0.60; a kicker is an opposite-colour marubozu pair gapping in the new direction, read on the close of bar *t*, entered the next close (one lag). Basket is surviving liquid ETFs — but this is a candlestick-pattern study, so the direction-matched random-entry baseline neutralizes the drift/survivorship. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
