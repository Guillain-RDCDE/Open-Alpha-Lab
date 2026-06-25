# Study 499 — Trendline-Break 📉

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the break forecast a drop? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The "short the break" rule does **not** beat a drift-matched **random-short** baseline — it is *significantly worse*: break − random = **−20.0 / −47.2 / −52.6 / −126.0 bps** at 5/10/20/60 days, and the break-vs-random Welch *t* is **negative** throughout (down to **−2.84** at 60d, *p* = 0.005). A close below the rising trendline is, on average, a **fresh local low that bounces up** — the break points the **wrong way**. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The break-short bleeds; the mirror *fade* (long the breakdown) only re-buys the index drift you already own. No residual edge once the drift is removed, and costs only deepen the hole. Nothing to scale. |
| **"Does the trendline break forecast?"** | ![Busted](https://img.shields.io/badge/Break_forecasts%3F-Busted-8b949e?style=flat-square) | Scramble the line's slope into nonsense (shuffled-slope placebo) and the result barely moves: **63%** of nonsense lines match or beat the real one (*p* = **0.631**). The specific least-squares trendline carries no information. |

> **In one sentence:** the trendline break — charting's most-taught reversal signal — looks authoritative because you draw it after the lows form; encode it mechanically (3-low OLS fit on confirmed fractals, no eyeballing) and fire the "close-below-the-line short" **617 times** across 5 indices over 21 years, and it **loses to shorting on random days** at every horizon (Welch *t* negative, the break shorts straight into a bounce), while the geometry placebo leaves the result untouched (*p* = 0.63): all artefact, no edge — and pointing the wrong way.

## What we tested

We encode the tightest mechanical version a proponent would accept. Swing lows are **confirmed fractals** (a local minimum with *k* = 10 strictly-higher bars each side, usable only 10 bars later — no look-ahead); at every bar we **least-squares-fit a rising trendline** through the three latest confirmed lows; a signal fires on the first close **below the line**, entered at the **next close** (one documented lag). The folklore is bearish ("support broke → it falls"), so we read it as a **short** (sign-flipped: a positive number means the break correctly forecast a drop) and measure the forward 5/10/20/60-day return on SPY, QQQ, IWM, DIA and GLD (yfinance daily total-return, 2005→2026). The Signal axis is **break vs a drift-matched random-short baseline** (a Welch *t*) — the only honest test when a short fights an up-drifting tape — plus a **shuffled-slope geometry placebo** that destroys the line while keeping the price marginal. Tradability charges costs on every break. A deterministic synthetic control with a *planted* post-break bounce proves the detector is live (edge 0 → *t* ≈ 0; planted bounce read as a fade → *t* = +9.01), so the adverse real-tape result is a genuine "the folklore is wrong" — and tells us the bankable effect at a break is reversion **up**, not the breakdown the lore claims.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a trendline break is, why shorting it loses to random, the false-breakdown bounce, and the slope scramble — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | mechanical OLS trendlines, one-sample HAC *t* vs the drift trap, the random-short Welch test, the shuffled-slope placebo, per-ticker deltas, costs, and a synthetic planted-bounce control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`trendline_break/`](trendline_break/). Swing lows are confirmed fractals (k = 10) with a 10-bar confirmation lag; the line is a 3-low OLS fit kept only when rising; entry is the next close (one lag). Basket is surviving liquid ETFs — but this is a single-instrument trend study, so the random-short baseline neutralizes the drift/survivorship. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
