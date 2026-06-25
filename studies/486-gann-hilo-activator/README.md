# Study 486 — Gann Hi-Lo Activator ⚡

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the flip pay? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The "buy the flip" rule does **not** beat a drift-matched **random-entry** baseline: flip − random = **+3.6 / −5.6 / +2.1 / −61.9 bps** at 5/10/20/60 days, and the flip-vs-random Welch *t* **never clears 2** (largest magnitude **−1.88** at 60d, *p* = 0.060 — and *negative*). The big one-sample *t*'s (20d **+6.20**, 60d **+6.24**) are **pure beta** — the upward drift every long-only entry inherits. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No residual edge once the free drift is removed; costs only deepen the hole. You'd capture the same drift more cheaply by **holding the index**. Nothing to scale. |
| **"Does the Hi-Lo flip forecast trend?"** | ![Busted](https://img.shields.io/badge/Flip_forecasts_trend%3F-Busted-8b949e?style=flat-square) | Move the flip to random dates (shuffled-flip timing placebo) and the result barely moves: **63%** of random-date draws match or beat the real flip (*p* = **0.629**). The flip's specific timing carries no information. |

> **In one sentence:** the Gann Hi-Lo Activator looks prophetic because indices drift up — encode the flip mechanically (a shifted SMA-of-highs / SMA-of-lows line, no eyeballing) and fire the "buy the flip" rule 1,234 times across 5 indices over 21 years, and it **ties or loses to buying on random days** at every horizon (and scrambling the flip's timing leaves the result untouched, *p* = 0.63): all tide, no tool.

## What we tested

We encode the tightest mechanical version a proponent would accept. The activator is a period-10 SMA of **highs** and of **lows**, each **shifted one bar** so today's line uses only prior data; the line **flips** short→long when the close prints above it (a trailing stop-and-reverse). A long fires on the first bar of each short→long flip, entered at the **next close** (one documented lag), and we measure the forward 5/10/20/60-day return on SPY, QQQ, IWM, DIA and GLD (yfinance daily total-return, 2005→2026). The Signal axis is **flip vs a drift-matched random-entry baseline** (a Welch *t*) — the only honest test on an upward-drifting tape — plus a **shuffled-flip timing placebo** that keeps the flip count and the price marginal but moves the entry dates at random. Tradability charges costs on every flip. A deterministic synthetic control with a *planted* post-flip trend proves the detector is live (edge 0 → *t* = −0.22; planted trend → *t* = +9.37, 85% win), so the flat real-tape result is a genuine "nothing there".

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the Hi-Lo Activator is, why a trailing line on a rising market always looks good, the flip-vs-random race, and the timing scramble — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the flipping activator, one-sample HAC *t* vs the beta trap, the random-entry Welch test, the shuffled-flip placebo, per-ticker deltas, costs, and a synthetic planted-trend control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`gann_hilo_activator/`](gann_hilo_activator/). The activator is a period-10 SMA of highs/lows shifted +1 bar (flip read on the close of t); entry is the next close (one lag). Basket is surviving liquid ETFs — but this is a single-instrument trend study, so the random-entry baseline neutralizes the drift/survivorship. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
