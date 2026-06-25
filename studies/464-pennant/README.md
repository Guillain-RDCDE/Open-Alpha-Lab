# Study 464 — Pennant 🚩

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the breakout beat random? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The "buy the breakout in the pole direction" rule does **not** beat a drift-matched **random-entry** baseline (matched long/short mix): pennant − random = **−29.6 / +28.1 / +48.1 / −66.1 bps** at 5/10/20/60 days (sign-flipping noise), and the breakout-vs-random Welch *t* **never clears 2** (max **+0.79** at 20d, *p* = 0.433). The only one-sample *t*'s that approach 2 (20d **+1.96**, 60d **+2.10**) are **beta** — the rule is net long ~80% of the time. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No residual edge once the free drift is removed; costs only deepen the hole. With ~5 breakouts per name per decade, there is nothing to scale even if there were an edge. You'd capture the same drift more cheaply by **holding the index**. |
| **"Does the pennant forecast continuation?"** | ![Busted](https://img.shields.io/badge/Forecasts_continuation%3F-Busted-8b949e?style=flat-square) | Scramble the traded **direction** on the same breakout dates (direction placebo) and the result is intact: **91%** of coin-flip-direction draws match or beat trading *with* the pole (*p* = **0.910**). "Continuation" carries no information. |

> **In one sentence:** The pennant looks like a continuation pattern because indices drift up and the rule is mostly long — encode it mechanically (steep pole, contracting triangle, breakout in the pole direction, no eyeballing) and fire it 109 times across 5 indices over 21 years, and it **does not beat entering on random days** (deltas flip sign across horizons, Welch *t* maxes at +0.79); scramble which way you trade and **91%** of coin-flips do as well (*p* = 0.91): all tide, no thrust.

## What we tested

We encode the tightest mechanical version a proponent would accept. A **pole** is a cumulative move over 8 bars exceeding 1.0 × (rolling σ × √8); the **converging body** is the next 12 bars whose recent half-range is below 85% of its earlier half-range (a contracting symmetrical triangle) with a small net move; the **breakout** fires when the close escapes the body range **in the pole direction**, entered at the **next close** (one documented lag, no look-ahead). We trade long if the pole was up, short if down, and measure the forward 5/10/20/60-day **pole-direction** return on SPY, QQQ, IWM, DIA and GLD (yfinance daily total-return, 2005→2026). The Signal axis is **breakout vs a drift-matched random-entry baseline** *with a matched long/short mix* (a Welch *t*) — the only honest test on an upward-drifting tape — plus a **direction-scramble placebo** that keeps the breakout dates but randomizes the traded direction, the direct test of the *continuation* thesis. Tradability charges costs on every breakout. A deterministic synthetic control with a *planted* pole→pause→continuation proves the detector is live (edge 0 → *t* ≈ 0; planted continuation → *t* = +10.30, win 90%), so the flat real-tape result is a genuine "nothing there". **No volume leg** — daily ETF tapes lack clean volume, so we test price geometry only and say so.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a pennant is, why a net-long breakout on a rising market always looks good, the breakout-vs-random race, and the direction scramble — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | mechanical poles + contracting triangles, one-sample HAC *t* vs the beta trap, the random-entry Welch test, the direction-scramble placebo, per-ticker deltas, costs, and a synthetic planted-continuation control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`pennant/`](pennant/). Pole + converging body are read on bars ≤ t; the breakout is detected on the close of t; entry is the next close (one lag). Basket is surviving liquid ETFs — but this is a single-instrument pattern study, so the random-entry baseline (with matched long/short mix) neutralizes the drift/survivorship. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
