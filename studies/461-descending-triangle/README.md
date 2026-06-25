# Study 461 — Descending Triangle 📐

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the break-down pay? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The "short the support break" rule does **not** beat a drift-matched **random-short** baseline: short − random = **+64.3 / −2.6 / −87.9 / −315.6 bps** at 5/10/20/60 days, and the only horizon that clears the *t* ≥ 2 bar clears it the **wrong way** (60-day Welch *t* = **−2.57**, *p* = 0.012 — the break-short is *significantly worse* than a random short). Booked as a short the rule **loses** money at 10/20/60 days. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Nothing to trade: the short bleeds the long-equity drift headwind, then loses *extra* vs a random short; costs only deepen the hole. The "measured-move target" never arrives because the break mostly reverses. Nothing to scale. |
| **"Does the break resolve in the textbook direction?"** | ![Busted](https://img.shields.io/badge/Resolves_textbook_direction%3F-Busted-8b949e?style=flat-square) | **It resolves the opposite way.** The 60-day win-rate of the bearish short is **23%** — the "bearish" break is followed by a *rally* three times in four. And scrambling the descending-highs geometry leaves the result intact: **69%** of nonsense triangles match or beat the real one (*p* = **0.692**). The defining geometry carries no information. |

> **In one sentence:** The descending triangle is sold as a textbook **bearish continuation** — flat floor, falling ceiling, breaks down — but encode it mechanically (confirmed-fractal pivots, descending highs + flat lows, no eyeballing) and short the support break **39 times** across 5 indices over 21 years, and the break **resolves UP**: the short loses at every horizon past 5 days, is *significantly worse* than shorting on random days at 60 days (*p* = 0.012, win-rate 23%), and the descending-highs geometry that names the pattern is not load-bearing (placebo *p* = 0.61). All chart-reading, no forecast.

## What we tested

We encode the tightest mechanical version a proponent would accept. Swing pivots are **confirmed fractals** (a local extremum with *k* = 5 strictly-beaten bars each side, usable only 5 bars later — no look-ahead); over a rolling window of the latest confirmed pivots we require the swing **highs to descend** (no higher high, last well below first) and the swing **lows to be flat** (spread inside a tolerance band) — a mechanical descending triangle with a horizontal support. A **short** fires on the first close **below that support** (the textbook break-down), entered at the **next close** (one documented lag), and we measure the forward 5/10/20/60-day **short** return on SPY, QQQ, IWM, DIA and GLD (yfinance daily total-return, 2005→2026). The Signal axis is **break vs a drift-matched random-short baseline** (a Welch *t*) — the only honest test on a drifting tape — plus a **scrambled-highs geometry placebo** that destroys the "descending highs" constraint while keeping the price marginal. Tradability charges costs on every break. A deterministic synthetic control with a *planted* break-down proves the detector is live (edge 0 → *t* ≈ 0, win 38%; planted break-down → +1103 bps, win 88%), so the wrong-way real-tape result is a genuine refutation, not a dead detector.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a descending triangle is, why a "bearish" break on a rising market is a trap, the short-vs-random race, and the geometry scramble — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | mechanical triangles, one-sample HAC *t* vs the drift headwind, the random-short Welch test, the scrambled-highs placebo, per-ticker deltas, costs, and a synthetic planted-break-down control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`descending_triangle/`](descending_triangle/). Pivots are confirmed fractals (k = 5) with a 5-bar confirmation lag; entry is the next close (one lag) and the trade is a SHORT (so positive = break-down paid). Basket is surviving liquid ETFs — but this is a single-instrument pattern study, so the random-short baseline neutralizes the drift/survivorship. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
