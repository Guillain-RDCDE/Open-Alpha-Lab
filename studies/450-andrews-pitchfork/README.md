# Study 450 — Andrews' Pitchfork 🔱

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the fork channel price? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The "buy the lower tine" rule does **not** beat a drift-matched **random-entry** baseline: touch − random = **−9.5 / −25.5 / −39.1 / +72.8 bps** at 5/10/20/60 days, and the touch-vs-random Welch *t* **never clears 2** (max **+1.77** at 60d, *p* = 0.078). The big one-sample *t*'s (20d **+3.93**, 60d **+6.47**) are **pure beta** — the upward drift every dip-buy inherits. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No residual edge once the free drift is removed; costs only deepen the hole. You'd capture the same drift more cheaply by **holding the index**. Nothing to scale. |
| **"Price respects the fork"?** | ![Busted](https://img.shields.io/badge/Respects_the_fork%3F-Busted-8b949e?style=flat-square) | Scramble the fork's geometry into nonsense (shuffled-pivot placebo) and the result barely moves: **61%** of nonsense forks match or beat the real one (*p* = **0.611**). The specific Andrews lines carry no information. |

> **In one sentence:** Andrews' pitchfork looks uncanny because indices drift up — encode it mechanically (confirmed-fractal pivots, no eyeballing) and fire the "buy the lower tine" rule 819 times across 5 indices over 21 years, and it **loses to buying on random days** at 5–20 days (and the geometry placebo leaves the result untouched, *p* = 0.61): all tide, no tool.

## What we tested

We encode the tightest mechanical version a proponent would accept. Swing pivots are **confirmed fractals** (a local extremum with *k* = 10 strictly-beaten bars each side, usable only 10 bars later — no look-ahead); at every bar we anchor a fork on the three latest confirmed pivots (median line P0→midpoint(P1,P2), tines parallel through P1 and P2); a long fires on the first close **below the lower tine**, entered at the **next close** (one documented lag), and we measure the forward 5/10/20/60-day return on SPY, QQQ, IWM, DIA and GLD (yfinance daily total-return, 2005→2026). The Signal axis is **touch vs a drift-matched random-entry baseline** (a Welch *t*) — the only honest test on an upward-drifting tape — plus a **shuffled-pivot geometry placebo** that destroys the lines while keeping the price marginal. Tradability charges costs on every touch. A deterministic synthetic control with a *planted* tine-bounce proves the detector is live (edge 0 → *t* ≈ 0; planted bounce → *t* = +3.44), so the flat real-tape result is a genuine "nothing there".

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a pitchfork is, why a dip-buy on a rising market always looks good, the touch-vs-random race, and the geometry scramble — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | mechanical forks, one-sample HAC *t* vs the beta trap, the random-entry Welch test, the shuffled-pivot placebo, per-ticker deltas, costs, and a synthetic planted-bounce control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`andrews_pitchfork/`](andrews_pitchfork/). Pivots are confirmed fractals (k = 10) with a 10-bar confirmation lag; entry is the next close (one lag). Basket is surviving liquid ETFs — but this is a single-instrument trend study, so the random-entry baseline neutralizes the drift/survivorship. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
