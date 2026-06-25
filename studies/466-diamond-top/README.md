# Study 466 — Diamond Top 💎

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the diamond call the turn? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The "short the breakdown" rule does **not** beat a drift-matched **random-short** baseline: break − random = **+74.0 / +45.7 / −108.5 / −273.6 bps** at 5/10/20/60 days, and the Welch *t* **never clears +2** (it's only +1.71 at 5d, *p* = 0.089, and at 60d it's **−2.55** — significant *the wrong way*). The supposed reversal short *loses* money at 20/60 days against the market's upward drift. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The short bleeds outright (20d **−151 bps**, 60d **−357 bps**) fighting the drift; costs only deepen the hole. The one flicker — a faint 5-day overshoot bounce — is sub-threshold and would be eaten by a fast round trip. Nothing to scale. |
| **"Does the diamond shape forecast a reversal?"** | ![Busted](https://img.shields.io/badge/Forecasts_a_reversal%3F-Busted-8b949e?style=flat-square) | Scramble the diamond's geometry into nonsense (shuffled-pivot placebo) and the result barely moves: **68%** of nonsense diamonds match or beat the real one (*p* = **0.679**). The broaden-then-narrow shape carries no information. |

> **In one sentence:** The diamond top — range broadens then narrows, the textbook "rare reversal" — looks dramatic, but encode it mechanically (confirmed-fractal pivots, no eyeballing) and fire the "short the breakdown" rule 125 times across 5 indices over 21 years, and the short **loses** at 20/60 days (worse than shorting random days, Welch *t* = −2.55 at 60d *the wrong way*) while the geometry placebo leaves the result untouched (*p* = 0.68): the diamond marks a pause, not a top.

## What we tested

We encode the tightest mechanical version a proponent would accept. Swing pivots are **confirmed fractals** (a local extremum with *k* = 5 strictly-beaten bars each side, usable only 5 bars later — no look-ahead); over the 6 most-recent alternating pivots we require the swing amplitudes to **broaden** (rise to a peak) then **narrow** (fall) — a diamond — formed after an advance; a **short** fires on the first close **below** the narrowing-apex floor, entered at the **next close** (one documented lag), and we measure the forward 5/10/20/60-day return of the short on SPY, QQQ, IWM, DIA and GLD (yfinance daily total-return, 2005→2026). The Signal axis is **break vs a drift-matched random-short baseline** (a Welch *t*) — the only honest test for a short on an upward-drifting tape — plus a **shuffled-pivot geometry placebo** that destroys the diamond while keeping the price marginal. Tradability charges costs on every break. A deterministic synthetic control with a *planted* diamond-top reversal proves the detector is live (edge 0 → *t* ≈ 0.8, no false positive; planted reversal → *t* = +5.06, win 77%), so the flat/negative real-tape result is a genuine "nothing there".

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a diamond top is, why a short on a rising market keeps losing, the break-vs-random-short race, and the geometry scramble — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | mechanical diamonds, one-sample HAC *t* vs the drift trap, the random-short Welch test, the shuffled-pivot placebo, per-ticker deltas, costs, and a synthetic planted-reversal control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`diamond_top/`](diamond_top/). Pivots are confirmed fractals (k = 5) with a 5-bar confirmation lag; diamonds span the 6 latest alternating pivots; entry is the next close (one lag), traded as a short. Basket is surviving liquid ETFs — but this is a single-instrument pattern study, so the random-short baseline neutralizes the drift/survivorship. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
