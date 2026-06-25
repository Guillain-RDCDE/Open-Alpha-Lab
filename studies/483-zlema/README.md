# Study 483 — Zero-Lag EMA ⚡

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the zero-lag filter forecast? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The "long while price > ZLEMA" rule does **not** beat a drift-matched **random-entry** baseline: ZLEMA − random = **−9.6 / −9.5 / −21.2 / −14.6 bps** at 5/10/20/60 days, and the Welch *t* is **negative at every horizon** (20d **−1.57**). The big one-sample *t*'s (20d **+8.09**) are **pure beta** — the upward drift every long-in-uptrend rule inherits. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No residual edge once the free drift is removed; costs only deepen the hole. You'd capture the same drift more cheaply by **holding the index**. Nothing to scale. |
| **"Does removing MA lag buy anything?"** | ![Busted](https://img.shields.io/badge/Removing_lag_buys_edge%3F-Busted-8b949e?style=flat-square) | Head-to-head with the plain EMA it claims to beat, ZLEMA − EMA = **−4.2 / +0.9 / +15.4 / +14.2 bps** (a dead heat); the de-lag placebo leaves the result intact (*p* = **0.363**); and on a *planted* trend the **plain EMA banks more** than the de-lagged line. The "zero lag" is cosmetic. |

> **In one sentence:** Ehlers' zero-lag EMA hugs price more tightly than a plain EMA, but encode the "long while price > ZLEMA" rule mechanically and fire it 2,811 times across 5 indices over 21 years, and it **loses to buying on random days** at every horizon, **ties the boring EMA** it was meant to beat, and survives a de-lag scramble untouched (*p* = 0.36): the timeliness is decoration, the payoff is all tide.

## What we tested

We encode the tightest mechanical version a proponent would accept. The **ZLEMA** is causal —
`EMA_L(close + (close − close[lag]))`, `lag = (L−1)//2`, `L = 20` — using only past closes. The
folklore rule is "**long while price > ZLEMA**"; we sample that long-state every 5 bars
(non-overlap), enter at the **next close** (one documented lag), and measure the forward
5/10/20/60-day return on SPY, QQQ, IWM, DIA and GLD (yfinance daily total-return, 2005→2026). The
Signal axis is **filter vs a drift-matched random-entry baseline** (a Welch *t*) — the only honest
test on an upward-drifting tape. The thesis axis is the **plain-EMA head-to-head** (identical rule,
identical length, only the line differs) plus a **shuffled-offset placebo** that permutes the
de-lag term while keeping its marginal — the direct test of "is the zero-lag correction doing
anything?". Tradability charges costs on every trade. A deterministic synthetic control with a
*planted* persistent trend proves the detector is live (edge 0 → *t* ≈ 0; planted trend → *t* =
+4.94, and the plain EMA banks even more at *t* = +7.52), so the flat real-tape result is a genuine
"nothing there".

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a zero-lag EMA is, why a long-in-uptrend rule on a rising market always looks good, the filter-vs-random race, and the plain-EMA dead heat — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | causal ZLEMA, one-sample HAC *t* vs the beta trap, the random-entry Welch test, the plain-EMA head-to-head, the shuffled-offset placebo, per-ticker deltas, the whippy-upcross foil, costs, and a synthetic planted-trend control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`zlema/`](zlema/). ZLEMA is causal (length 20, lag 9); the `price > ZLEMA` long-state is read on the close and entered at the next close (one lag). Basket is surviving liquid ETFs — but this is a single-instrument trend study, so the random-entry baseline neutralizes the drift/survivorship. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
