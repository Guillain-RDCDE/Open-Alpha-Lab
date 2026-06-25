# Study 494 — Bullish Percent Index 📊

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does washed-out breadth call bottoms? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The "BPI crosses up out of oversold → buy SPY" rule does **not** beat a drift-matched **random-entry** baseline: cross − random = **−27.2 / +3.2 / −45.5 / −1.6 bps** at 5/10/20/60 days, and the cross-vs-random Welch *t* **never clears 2** (max **+0.08** at 10d, *p* = 0.93). Even the one-sample *t* barely registers (20d **+0.53**) — the oversold cross fires too rarely (159 trades) to ride much drift. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The cross-minus-random delta is **negative in all 5 names** at 20 days; no residual edge once the free drift is removed, and costs only deepen the hole. You'd capture the same drift more cheaply by **holding the index**. Nothing to scale. |
| **"Does the BPI forecast turns?"** | ![Busted](https://img.shields.io/badge/Forecasts_turns%3F-Busted-8b949e?style=flat-square) | Block-shuffle the BPI series in time (scrambled-breadth placebo) and the result barely moves: **97%** of randomly re-timed breadth series match or beat the real one (*p* = **0.974**). The specific breadth path carries no information. |

> **In one sentence:** the Bullish Percent Index looks like a market thermometer that calls tops and bottoms — encode it mechanically (% of the basket above its 50-day SMA, buy the up-cross out of oversold, no eyeballing) and fire it 159 times on SPY over 21 years, and it **loses to buying on random days** at every horizon (and re-timing the breadth at random leaves the result untouched, *p* = 0.97): a breadth gauge, not a forecast.

## What we tested

We encode the tightest mechanical version a proponent would accept. **BPI** is the percentage of a liquid ETF basket (SPY, QQQ, IWM, DIA) trading **above its 50-day SMA** — a transparent, causal proxy for the classic Point & Figure buy-signal count (stated as a proxy that caps the test; true BPI counts every member of a full exchange). A long fires on the bar where BPI **crosses up through 30** (Cohen's reversal out of oversold), entered at the **next close** (one documented lag), and we measure the forward 5/10/20/60-day return on SPY (yfinance daily total-return, 2005→2026). The Signal axis is **cross vs a drift-matched random-entry baseline** (a Welch *t*) — the only honest test on an upward-drifting tape — plus a **scrambled-breadth placebo** that block-shuffles the BPI in time, destroying its alignment with price while keeping the marginal. Tradability charges costs on every cross. A deterministic synthetic control with a *planted* oversold-bounce proves the detector is live (edge 0 → *t* = +0.08; planted bounce → *t* = +3.38), so the flat real-tape result is a genuine "nothing there".

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the BPI is, why a breadth dip-buy on a rising market always looks good, the cross-vs-random race, and the breadth scramble — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the breadth oscillator, one-sample HAC *t* vs the beta trap, the random-entry Welch test, the scrambled-breadth placebo, per-instrument deltas, costs, and a synthetic planted-bounce control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`bullish_percent_index/`](bullish_percent_index/). BPI = % of basket above its 50-day SMA (causal, no look-ahead); entry is the next close (one lag). Breadth is a coarse proxy for true exchange breadth and caps the test — but the random-entry baseline neutralizes the drift it inherits. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
