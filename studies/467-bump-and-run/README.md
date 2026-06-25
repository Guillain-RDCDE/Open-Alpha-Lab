# Study 467 — Bump-and-Run Reversal 🪤

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the bump-and-run break short pay? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The "short the trendline break" rule does **not** beat a drift-matched **random short** baseline — it does **worse**: break − random = **−60.9 / −71.2 / −123.9 / −131.8 bps** at 5/10/20/60 days, and the break-vs-random Welch *t* is **negative**, clearing −2 at 3 of 4 horizons (**−2.50 / −2.00 / −2.54 / −1.72**). Price tends to keep *rising* after the break — the opposite of the advertised reversal. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The short loses in absolute terms (negative short-beta) **and** loses to a random short (no residual skill); costs only deepen the hole. Nothing to scale. |
| **"Does the bump-then-break forecast a reversal?"** | ![Busted](https://img.shields.io/badge/Forecasts_a_reversal%3F-Busted-8b949e?style=flat-square) | Scramble the bump-and-run shape into nonsense (shuffled-window placebo) and the result barely moves: **36%** of order-destroyed tapes match or beat the real one (*p* = **0.359**). The specific lead-in / bump / break geometry carries no information. |

> **In one sentence:** The bump-and-run reversal looks like a textbook short — a quiet lead-in, a speculative bump, a break back below the line — but encode it mechanically (trailing-fit lead-in, ≥2× bump, downcross break, no eyeballing) and fire the short 162 times across 5 indices over 21 years, and it **loses to shorting on random days** (Welch *t* ≤ −2 at 5/10/20d): price keeps climbing after the break, the placebo leaves the result untouched (*p* = 0.36), and the planted-reversal control proves the detector is live — so the break forecasts nothing.

## What we tested

We encode the tightest mechanical version a Bulkowski proponent would accept. The **lead-in** is a least-squares trendline on a trailing 60-bar window with a *gently positive* slope (a calm up-trend, not a ramp), fit only on past bars. A **bump** is confirmed when the close surges to ≥ **2×** the lead-in's own above-line height *and* the bump just peaked (a recent rollover, so stale geometry can't re-fire). A **short** fires on the first close that *downcrosses* the extended lead-in line, entered at the **next close** (one documented lag), and we measure the forward 5/10/20/60-day return of the short on SPY, QQQ, IWM, DIA and GLD (yfinance daily total-return, 2005→2026). The Signal axis is **break vs a drift-matched random *short*** baseline (a Welch *t*) — the only honest test for a short on an upward-drifting tape — plus a **shuffled-window placebo** that destroys the bump-and-run ordering while keeping the price marginal. Tradability charges costs on every break. A deterministic synthetic control with a *planted* post-bump reversal proves the detector is live (edge 0 → *t* = +1.86, below the bar; planted reversal → *t* = +3.45, win 69%), so the negative real-tape result is a genuine "nothing there" — the break does not forecast the reversal it advertises.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a bump-and-run is, why shorting a rising market always looks bad, the break-vs-random-short race, and the shape scramble — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | mechanical lead-in/bump/break, one-sample HAC *t* vs the short-beta trap, the random-short Welch test, the shuffled-window placebo, per-ticker deltas, costs, and a synthetic planted-reversal control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`bump_and_run/`](bump_and_run/). Lead-in is a trailing least-squares trendline (gentle positive slope); the bump is ≥2× the lead-in height with a recent peak; entry is the next close after a downcross break (one lag). Basket is surviving liquid ETFs — but this is a single-instrument pattern study, so the random-short baseline neutralizes the drift/survivorship. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
