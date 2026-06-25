# Study 487 — Elder's Triple Screen 🖥️🖥️🖥️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do three aligned screens forecast? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The triple-screen long does **not** beat a drift-matched **random-entry** baseline: triple − random = **−14.0 / +2.3 / +6.4 / −37.1 bps** at 5/10/20/60 days, and the triple-vs-random Welch *t* **never clears 2** (max **+0.44** at 20d; it is **−1.92** at 5d — *worse* than a dart). The big one-sample *t*'s (20d **+5.21**, 60d **+5.97**) are **pure beta** — the upward drift every long-only entry inherits. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No residual edge once the free drift is removed; costs only deepen the hole. You'd capture the same drift more cheaply by **holding the index**. Nothing to scale. |
| **"Does the multi-timeframe filter add edge?"** | ![Busted](https://img.shields.io/badge/Filter_adds_edge%3F-Busted-8b949e?style=flat-square) | Shift the weekly trend out of alignment with price (screen-scramble placebo) and the result barely moves: **65%** of misaligned filters match or beat the real one (*p* = **0.649**). The weekly-to-daily alignment carries no information. |

> **In one sentence:** Elder's Triple Screen looks bulletproof because it stacks three trend-following filters on a market that drifts up — encode all three mechanically (weekly MACD-histogram trend, daily Force-Index pullback, prior-high breakout, no eyeballing) and fire the long 2,369 times across 5 indices over 21 years, and it **ties buying on random days** (and shifting the trend out of alignment leaves the result untouched, *p* = 0.65): three screens, all tide, no tool.

## What we tested

We encode the tightest mechanical version a proponent would accept. **Screen 1** is the slope of a *weekly* MACD-histogram (computed on resampled weekly closes, forward-filled to days and **shifted one day** — no look-ahead). **Screen 2** is a daily Force-Index proxy (EMA of the close-to-close change) dipping below zero within the last 5 bars — the oversold pullback against the up-tide. **Screen 3** is a close above the **prior bar's high** — the breakout trigger. A long fires when all three align, entered at the **next close** (one documented lag), and we measure the forward 5/10/20/60-day return on SPY, QQQ, IWM, DIA and GLD (yfinance daily total-return, 2005→2026). The Signal axis is **triple-screen vs a drift-matched random-entry baseline** (a Welch *t*) — the only honest test on an upward-drifting tape — plus a **screen-scramble placebo** that circularly shifts the weekly trend out of alignment while keeping each screen's marginal. Tradability charges costs on every trigger. A deterministic synthetic control with a *planted* post-alignment bounce proves the detector is live (edge 0 → *t* = −0.07; planted bounce → *t* = +8.29), so the flat real-tape result is a genuine "nothing there".

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the Triple Screen is, why stacking trend filters on a rising market always looks good, the triple-vs-random race, and the alignment scramble — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | mechanical three-screen entries, one-sample HAC *t* vs the beta trap, the random-entry Welch test, the screen-scramble placebo, per-ticker deltas, costs, and a synthetic planted-bounce control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`elder_triple_screen/`](elder_triple_screen/). Screen 1 (weekly MACD-hist slope) is shifted one day; entry is the next close (one lag). Basket is surviving liquid ETFs — but this is a single-instrument trend study, so the random-entry baseline neutralizes the drift/survivorship. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
