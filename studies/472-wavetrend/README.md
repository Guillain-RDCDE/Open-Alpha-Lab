# Study 472 — WaveTrend (LazyBear) 🌊

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the WaveTrend cross beat drift? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | A single lucky random-baseline seed (=7) threw Welch *t* > 2 at all four horizons — but the honest bar is the **seed-AVERAGED** Welch *t*. Over 30 seeds it clears 2 **only at 10–20 days** (mean **2.74 / 2.37**); at 5d (**1.97**) and 60d (**1.29**) it fails. The zero-noise unconditional-drift test agrees (significant only at 10d **+2.77** / 20d **+2.56**, not 5d **+1.89** or 60d **+1.38**). A faint, narrow edge — not the across-the-board green the lucky seed suggested. |
| **Tradability** — can you deploy it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | Only **115 trades** in 21 years (~1 per ETF per year); over 30 seeds the 20d Welch *t* spans **+0.95 → +3.72** and only ~70% of seeds clear 2. The edge lives in a thin 10–20-day band and wobbles with the random-baseline seed. Not bankable. |
| **"Does the WT cross forecast?"** | ![Mixed](https://img.shields.io/badge/Does_the_WT_cross_forecast%3F-Mixed-dab617?style=flat-square) | The geometry placebo (SPY, 20d) rejects the scrambled-wave null — only **1.1%** of scrambled waves match the real one (*p* = **0.011**) — so *where the signal exists* the wave is load-bearing. But it forecasts only in the 10–20-day window, so the broad folklore claim (a high-probability buy at *every* horizon) is **not** confirmed. |

> **In one sentence:** WaveTrend's oversold cross-up *looked* like the rare oscillator that beats buying on random days — but that rested on a single lucky baseline seed; averaged over 30 seeds the edge survives at *t* ≥ 2 **only at 10–20 days** and collapses at 5 and 60, so the signal is merely **Weak** (not Real) and the folklore thesis **Mixed** (not Confirmed) — the geometry placebo (*p* = 0.011) confirms the wave matters where the signal exists, but on **115 trades in 21 years** in a thin 10–20-day band the tradability stamp stays **Fragile**.

## What we tested

We encode the tightest mechanical version a proponent would accept. WaveTrend (LazyBear) is built from the typical price HLC3: `esa = EMA(tp, 10)`, `d = EMA(|tp−esa|, 10)`, `ci = (tp−esa)/(0.015·d)`, `WT1 = EMA(ci, 21)`, `WT2 = SMA(WT1, 4)` — all **causal** (no look-ahead). A long fires when **WT1 crosses up through WT2 while WT1 was oversold** (below −60) on the prior bar, entered at the **next close** (one documented lag); we measure the forward 5/10/20/60-day return on SPY, QQQ, IWM, DIA and GLD (yfinance daily total-return, 2005→2026). The Signal axis is **cross vs a drift-matched random-entry baseline** (a Welch *t*) — the only honest test on an upward-drifting tape — but a *single* random seed can fluke a *t* > 2, so the decisive number is the **seed-AVERAGED Welch *t* over 30 baseline seeds** (cf. Study 452, which caught the identical trap). We add a **scrambled-signal geometry placebo** that permutes WT1's increments while keeping its marginal, and a zero-sampling-noise unconditional-drift baseline. A deterministic synthetic control with a *planted* WaveTrend bounce proves the detector is honest (edge 0 → *t* = +1.25, no false positive; planted bounce → *t* = +4.89). The upshot: averaged over seeds the edge is real **only at 10–20 days** — a faint, narrow signal (**Weak**), with the geometry confirmed where it exists (**Mixed**) but the across-the-board folklore refuted.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what WaveTrend is, why most dip-buys are just drift, the cross-vs-random race (and why one lucky seed over-stated it), and the wave-scramble check — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | causal WaveTrend, one-sample HAC *t* vs the **seed-averaged** random-baseline Welch test (the decisive bar), the scrambled-signal placebo, per-ticker deltas, the 30-seed sweep + unconditional-drift caveat, costs, and a synthetic planted-bounce control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`wavetrend/`](wavetrend/). Lines are causal EMAs/SMA of HLC3 (n1=10, n2=21, signal=4, oversold=−60); entry is the next close (one lag). Basket is surviving liquid ETFs — but this is a single-instrument timing study, so the random-entry baseline neutralizes the drift/survivorship. A **Weak × Fragile × Mixed** result: the cross beats random only in a thin 10–20-day band once the Welch *t* is averaged over baseline seeds; what a single lucky seed made look Real is, on rigorous re-testing, a faint amber. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
