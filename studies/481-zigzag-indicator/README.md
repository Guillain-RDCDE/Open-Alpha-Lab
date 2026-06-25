# Study 481 — ZigZag Indicator ⚡

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the ZigZag turn forecast? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The "buy the confirmed up-leg" rule does **not** beat a drift-matched **random-entry** baseline at the horizons a turn lives on: up-leg − random = **−23.7 / +10.6 / +58.5 bps** at 5/10/20 days, with Welch *t* = **−1.04 / +0.34 / +1.33** (all *p* > 0.18). Only the **60-day** horizon clears the bar (Welch *t* = **+2.12**, *p* = 0.034) — and the placebo shows that is *not* the ZigZag's geometry. The big one-sample *t*'s (20d **+2.87**, 60d **+4.35**) are mostly **beta** — the upward drift every dip-buy inherits. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No residual turn edge once the free drift is removed; the lone 60-day blip is geometry-independent slow post-pullback drift you'd capture more cheaply by **holding the index**, and costs only deepen the hole. Nothing to scale. |
| **"Is the ZigZag tradable once you remove repaint?"** | ![Busted](https://img.shields.io/badge/ZigZag_tradable%3F-Busted-8b949e?style=flat-square) | Relabel which confirmation dates count as "lows" (relabelled-leg placebo) and the result barely moves: **65%** of nonsense-label runs match or beat the real up-leg (*p* = **0.645**). The ZigZag's up/down structure carries no information — strip the repaint and the turns stop being tradable. |

> **In one sentence:** The ZigZag looks uncanny because its lows sit exactly on the bottoms — but those lows **repaint** (only knowable after the bounce) and indices drift up; encode it mechanically (5% threshold, trade only **confirmed** legs, no peeking) and fire the "buy the turn" rule 428 times across 5 indices over 21 years, and it **fails to beat random** at 5/10/20 days (the one significant horizon, 60d, survives a random relabelling of the legs, *p* = 0.65): all tide and repaint, no tool.

## What we tested

We encode the tightest mechanical version a proponent would accept, with the **repaint removed**. A threshold ZigZag (5% reversal) connects alternating swing highs and lows; a swing low is only **confirmed** once price has rebounded 5% above it (the repaint/confirmation lag — never the future-peeking final pivot). At each confirmed low the ZigZag turns up, so a long fires, entered at the **next close** (one documented lag), and we measure the forward 5/10/20/60-day return on SPY, QQQ, IWM, DIA and GLD (yfinance daily total-return, 2005→2026). The Signal axis is **confirmed up-leg vs a drift-matched random-entry baseline** (a Welch *t*) — the only honest test on an upward-drifting tape — plus a **relabelled-leg geometry placebo** that randomises which confirmation dates count as "lows" while keeping the price marginal. Tradability charges costs on every signal. A deterministic synthetic control with a *planted* post-turn drift proves the detector is live (edge 0 → *t* ≈ 0; planted turn → *t* = +5.37), so the flat real-tape result is a genuine "nothing there".

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a ZigZag is, why it repaints, why a dip-buy on a rising market always looks good, the turn-vs-random race, and the leg relabelling — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | confirmed-leg ZigZag (no repaint), one-sample HAC *t* vs the beta trap, the random-entry Welch test, the relabelled-leg placebo, per-ticker deltas, costs, and a synthetic planted-turn control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`zigzag_indicator/`](zigzag_indicator/). Swings are a 5% threshold ZigZag; a swing low is usable only at its confirmation bar (price rebounded +5%, the repaint lag); entry is the next close (one lag). Basket is surviving liquid ETFs — but this is a single-instrument trend study, so the random-entry baseline neutralizes the drift/survivorship. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
