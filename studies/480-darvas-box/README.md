# Study 480 — Darvas Box 📦

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the box breakout channel price? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The "buy the close above the box top" rule does **not** beat a drift-matched **random-entry** baseline: breakout − random = **−16.4 / −14.8 / −8.2 / −9.2 bps** at 5/10/20/60 days, and the breakout-vs-random Welch *t* is **never positive** (most negative **−1.56** at 5d, *p* = 0.119). The big one-sample *t*'s (20d **+6.41**, 60d **+6.67**) are **pure beta** — the upward drift every long-only breakout inherits. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No residual edge once the free drift is removed; costs only deepen the hole. You'd capture the same drift more cheaply by **holding the index**. Nothing to scale. |
| **"Does the box breakout forecast?"** | ![Busted](https://img.shields.io/badge/Box_breakout_forecasts%3F-Busted-8b949e?style=flat-square) | Scatter the breakout dates at random (shuffled-box placebo) and the result barely moves: **19%** of random-date entries match or beat the real breakout (*p* = **0.192**). The box-top timing carries no information. |

> **In one sentence:** Darvas' box looks like momentum magic because indices drift up — encode it mechanically (trailing-high box top, a real consolidation, no eyeballing) and fire the "buy the breakout" rule 949 times across 5 indices over 21 years, and it **loses to buying on random days** at every horizon (and scattering the entry dates leaves the result untouched, *p* = 0.19): all tide, no tool.

## What we tested

We encode the tightest mechanical version a proponent would accept. The **box top** is the highest close over a trailing 20-day window (a Donchian upper band, shifted one bar — no look-ahead), counted as a *box* only after the close has sat **below it for ≥ 5 consecutive bars** (the consolidation must have formed); the **box bottom** is the trailing 20-day low (the ATR/box stop reference). A long fires on the first close **above the box top**, entered at the **next close** (one documented lag), and we measure the forward 5/10/20/60-day return on SPY, QQQ, IWM, DIA and GLD (yfinance daily total-return, 2005→2026). The Signal axis is **breakout vs a drift-matched random-entry baseline** (a Welch *t*) — the only honest test on an upward-drifting tape — plus a **shuffled-box placebo** that scatters the entry dates while keeping the same count and price marginal. Tradability charges costs on every breakout. A deterministic synthetic control with a *planted* post-breakout continuation proves the detector is live (edge 0 → *t* = −1.56, no false positive; planted continuation → *t* = +9.77), so the flat real-tape result is a genuine "nothing there".

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a Darvas box is, why a breakout-buy on a rising market always looks good, the breakout-vs-random race, and the date scramble — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | mechanical boxes, one-sample HAC *t* vs the beta trap, the random-entry Welch test, the shuffled-box placebo, per-ticker deltas, costs, and a synthetic planted-continuation control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`darvas_box/`](darvas_box/). Box top/bottom are trailing windows (lookback = 20) with a one-bar shift and a 5-bar consolidation requirement; entry is the next close (one lag). Basket is surviving liquid ETFs — but this is a single-instrument momentum study, so the random-entry baseline neutralizes the drift/survivorship. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
