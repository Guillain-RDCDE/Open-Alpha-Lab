# Study 463 — Bear-Flag 🏴

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the breakdown continue the drop? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The "short the flag breakdown" rule does **not** beat a drift-matched **random-short** baseline: breakdown − random = **−47.2 / −3.7 / −65.1 / −184.2 bps** at 5/10/20/60 days, and the breakdown-vs-random Welch *t* **never clears 2** (most negative **−1.90** at 60d, *p* = 0.059) — and in the believer's-*wrong* direction. The short itself simply **loses** (60d **−303 bps**, one-sample *t* = −3.39): that's the index's upward drift bleeding the short, not continuation. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | There is no continuation to capture — only negative drift carry from standing short in front of a rising index, made worse by costs. Nothing to scale; the expectancy is negative at every horizon. |
| **"Does the flag forecast continuation?"** | ![Busted](https://img.shields.io/badge/Forecasts_continuation%3F-Busted-8b949e?style=flat-square) | Replace the up-sloping-flag test with a coin (shuffled-flag placebo) and the result barely moves: **32%** of coin-flip flags match or beat the real one (*p* = **0.323**). The flag's specific geometry carries no information. |

> **In one sentence:** the bear flag looks ominous because a sharp drop *feels* like it must continue — encode it mechanically (a ≥6% pole, a 7-bar up-sloping flag, a confirmed breakdown, no eyeballing) and fire the "short the breakdown" rule 156 times across 5 indices over 21 years, and the short **loses money at every horizon** and **loses to shorting on random days** (the flag-geometry placebo leaves it untouched, *p* = 0.32): the breakdown is closer to a local bottom than a second leg down.

## What we tested

We encode the tightest mechanical version a proponent would accept. A **pole** is a log-close fall of ≥ 6% over a 10-bar lookback; a **flag** is the next 7 bars forming a *positive-slope* (up-sloping) consolidation that retraces at most 60% of the pole (a pause, not a reversal); a **short** fires on the first close **below the flag's lower trendline**, entered at the **next close** (one documented lag), and we measure the forward 5/10/20/60-day return **of the short** on SPY, QQQ, IWM, DIA and GLD (yfinance daily total-return, 2005→2026). The Signal axis is **breakdown vs a drift-matched random-short baseline** (a Welch *t*) — the only honest test on an upward-drifting tape, since any short carries negative drift — plus a **shuffled-flag geometry placebo** that replaces the up-slope test with a coin while keeping the pole filter and the price marginal. Tradability charges costs on every breakdown. A deterministic synthetic control with a *planted* post-breakdown continuation proves the detector is live (edge 0 → *t* = +0.67; planted continuation → *t* = +10.74, 87% win), so the negative real-tape result is a genuine "nothing there".

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a bear flag is, why shorting a rising market always looks "almost right", the breakdown-vs-random race, and the flag scramble — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | mechanical poles + up-flags, one-sample HAC *t* vs the short-side drift trap, the random-short Welch test, the shuffled-flag placebo, per-ticker deltas, costs, and a synthetic planted-continuation control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`bear_flag/`](bear_flag/). The pole/flag geometry uses only bars up to *t*; the breakdown is read on the close of *t* and the short is entered at the next close (one lag). Basket is surviving liquid ETFs — but this is a single-instrument continuation study, so the random-short baseline neutralizes the drift/survivorship. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
