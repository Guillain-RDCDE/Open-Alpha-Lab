# Study 468 — Gartley / AB=CD Harmonic 🦋

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the D-point reverse? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The D-point long beats a drift-matched **random-entry** baseline at **only one** of four horizons: Welch *t* = −0.13 / +0.47 / +1.09 / **+2.40** at 5/10/20/60 days (60d *p* = **0.017**, Δ = +201 bps). A thin, long-only, long-horizon effect — three of four horizons are a coin flip. The big one-sample *t*'s (20d **+3.05**, 60d **+3.57**) are mostly drift. |
| **Tradability** — can you deploy it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | The edge lives at a single 60-day horizon, evaporates at the shorter holds, and **survives the geometry scramble** — i.e. it's generic "buy a confirmed swing low, wait three months" dip-buying, not harmonic structure. 194 trades over 21 years (~9/yr); nothing scalable, and you'd capture it more cheaply without drawing forks. |
| **"Do the Fibonacci ratios forecast the D-point turn?"** | ![Busted](https://img.shields.io/badge/Fibonacci_forecasts%3F-Busted-8b949e?style=flat-square) | Swap the Fibonacci grid (0.618/0.786) for **random** ratio targets and the result barely moves: **44%** (20d) / **59%** (60d) of nonsense grids match or beat the real Gartley grid (*p* = 0.44 / 0.59). The magic numbers carry no information. |

> **In one sentence:** A bullish Gartley XABCD looks prophetic because indices drift up and swing-lows mean-revert — encode it mechanically (confirmed-fractal pivots, no eyeballing) and fire the D-point long 194 times across 5 indices over 21 years, and you find a thin edge **only at 60 days** that **any random ratio grid reproduces** (placebo *p* = 0.44–0.59): there's a long-horizon dip-buy in there, but **the Fibonacci ratios are doing none of the work**.

## What we tested

We encode the tightest mechanical version a proponent would accept. Swing pivots are **confirmed
fractals** (a local extremum with *k* = 4 strictly-beaten bars each side, usable only 4 bars later
— no look-ahead); at every confirmed swing-low we scan recent pivots for a correctly-ordered,
alternating **X-A-B-C-D** whose retracement ratios satisfy the bullish-Gartley grid (B = 0.618·XA,
C = 0.382–0.886·AB, D = 0.786·XA within a tolerance); a long fires at the D-completion, entered at
the **next close** (one documented lag), and we measure the forward 5/10/20/60-day return on SPY,
QQQ, IWM, DIA and GLD (yfinance daily total-return, 2005→2026). The Signal axis is **D-point vs a
drift-matched random-entry baseline** (a Welch *t*) — the only honest test on an upward-drifting
tape — plus a **ratio-grid placebo** that swaps the Fibonacci targets for random ratios while
keeping the zig-zag machinery. Tradability charges costs on every trade. A deterministic synthetic
control with a *planted* Gartley bounce proves the detector is live (edge 0 → *t* = −0.10; planted
bounce → *t* = +60.6), so the busted ratio result is a genuine "the Fibonacci numbers aren't it".

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a Gartley pattern is, why a swing-low buy on a rising market always looks good, the D-vs-random race, and the Fibonacci-ratio scramble — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | mechanical XABCD scans, one-sample HAC *t* vs the beta trap, the random-entry Welch test, the ratio-grid placebo, per-ticker deltas, costs, and a synthetic planted-bounce control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`gartley_harmonic/`](gartley_harmonic/). Pivots are confirmed fractals (k = 4) with a 4-bar confirmation lag; entry is the next close (one lag). Basket is surviving liquid ETFs — but this is a single-instrument pattern study, so the random-entry baseline neutralizes the drift/survivorship. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
