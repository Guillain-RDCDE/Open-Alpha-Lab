# Study 456 — Belt-Hold 🥋

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the belt-hold beat random? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | Vs a drift-matched **random-entry** baseline the belt-hold is a coin-flip at 5/10/20 days (Δ = **−1.7 / +9.3 / +32.8 bps**, Welch *t* ≤ 1.15) and only **marginally** beats random at 60 days (Δ = **+93.9 bps**, Welch *t* = **+2.03**, *p* = 0.042). One horizon out of four, right at the bar, concentrated in a single name — enough to deny a clean "None", far short of "Real". The big one-sample *t*'s (20d **+3.68**, 60d **+5.86**) are mostly beta. |
| **Tradability** — can you deploy it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | The only vs-random edge is slow (60-day), leans on **SPY** (+97 bps) while flipping **negative** on QQQ (−34 bps), and is really the prior-downtrend context. Costs eat a thin result; nothing robust to scale. |
| **"Does the opening-at-extreme reverse?"** | ![Busted](https://img.shields.io/badge/Opening_extreme_reverses%3F-Busted-8b949e?style=flat-square) | Keep the downtrend filter but scramble the candle shape (buy *any* random downtrend bar) and the result is intact: **43%** of shape-blind draws match or beat the real belt-hold (*p* = **0.433**). The open-at-the-low geometry carries no information — the residual is the downtrend, not the candle. |

> **In one sentence:** A bullish belt-hold (opens at its low, no lower wick; closes well up; after a downtrend) looks like a clean reversal — but encode it mechanically and fire it 666 times across 5 indices over 21 years, and it's a coin-flip vs buying on random days at 5–20 days, nudges ahead only at 60 days (Welch *t* = 2.03, barely), and a candle-shape scramble leaves it untouched (*p* = 0.43): the thin residual is the downtrend context, not the open-at-the-extreme.

## What we tested

We encode the tightest mechanical version a proponent would accept. A **bullish belt-hold** flags on the close of *t* when the bar has a **white body**, the **open sits within 10% of the low** (open ≈ low — the open-at-the-extreme premise), a **tall body** (≥60% of the bar's high-low range), and a real **prior downtrend** (close below the close 10 bars earlier). A long fires on the belt-hold close, entered at the **next close** (one documented lag), and we measure the forward 5/10/20/60-day return on SPY, QQQ, IWM, DIA and GLD (yfinance daily total-return, 2005→2026). The Signal axis is **belt-hold vs a drift-matched random-entry baseline** (a Welch *t*) — the only honest test on an upward-drifting tape — plus a **shape-scramble placebo** that keeps the downtrend filter but destroys the candle geometry. Tradability charges costs on every signal. A deterministic synthetic control with a *planted* belt-hold reversal proves the detector is live (edge 0 → *t* = +0.27; planted reversal → *t* = +13.14), so the marginal real-tape result is a genuine reading.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a belt-hold is, why a dip-buy on a rising market always looks good, the belt-hold-vs-random race, and the candle-shape scramble — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | mechanical belt-hold flags, one-sample HAC *t* vs the beta trap, the random-entry Welch test, the shape-scramble placebo, per-ticker deltas, costs, and a synthetic planted-reversal control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`belt_hold/`](belt_hold/). The belt-hold is read on the close of t (open within 10% of the low, body ≥60% of range, prior 10-bar downtrend); entry is the next close (one lag). Basket is surviving liquid ETFs — but this is a single-instrument candlestick study, so the random-entry baseline neutralizes the drift/survivorship. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
