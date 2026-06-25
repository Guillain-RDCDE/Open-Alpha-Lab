# Study 491 — McClellan Oscillator 📊

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the breadth-cross forecast SPY? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The McClellan up-cross-from-negative does **not** beat a drift-matched **random-entry** baseline — it is *worse* at **every** horizon: trigger − random = **−15.0 / −26.4 / −71.8 / −100.6 bps** at 5/10/20/60 days, and the trigger-vs-random Welch *t* is **significantly negative** (−2.55 at 20d, −2.20 at 60d). The only positive-looking number, the 60d one-sample *t* = **+3.01**, is **pure beta** — the upward drift every long entry inherits. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No residual edge once the free drift is removed; the rule actually *gives back* drift by entering at worse-than-random times, and costs only deepen the hole. You'd capture the same drift more cheaply by **holding the index**. Nothing to scale. |
| **"Does breadth momentum forecast the index?"** | ![Busted](https://img.shields.io/badge/Forecasts_the_index%3F-Busted-8b949e?style=flat-square) | Shuffle the net-advances series **in time** (kill the breadth-momentum structure, keep the marginal) and the result is untouched: **99%** of time-shuffled-breadth oscillators match or beat the real one (*p* = **0.988**). The EMA19−EMA39 geometry carries no information. |

> **In one sentence:** The McClellan oscillator looks like a breadth crystal ball because indices drift up — encode the textbook "buy the up-cross from negative" trigger mechanically and fire it 528 times on SPY over 21 years, and it **loses to buying on random days** at *every* horizon (Welch *t* significantly negative at 20–60d), while the time-shuffle placebo leaves the result intact (*p* = 0.99): all tide, no tool.

## What we tested

We encode the tightest mechanical version a proponent would accept. The oscillator is **EMA19(net-advances) − EMA39(net-advances)**, computed causally on a daily breadth proxy (net advances = members up minus members down across a small liquid-ETF basket — a coarse stand-in for true NYSE exchange breadth, stated loudly in the [docs](docs/results.md) and *capping* the test). A long fires on the first close where the oscillator **crosses up through zero from a non-positive value** (the textbook bull trigger), entered at the **next close** (one documented lag), and we measure the forward 5/10/20/60-day return on SPY (yfinance daily total-return, 2005→2026). The Signal axis is **trigger vs a drift-matched random-entry baseline** (a Welch *t*) — the only honest test on an upward-drifting tape — plus a **shuffled-breadth placebo** that destroys the indicator's temporal structure while keeping its marginal. Tradability charges costs on every trigger. A deterministic synthetic control with a *planted* post-cross bounce proves the detector is live (edge 0 → *t* ≈ 0; planted bounce → *t* = +18.84), so the dead-flat real-tape result is a genuine "nothing there".

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the McClellan oscillator is, why a breadth-cross on a rising market always looks good, the trigger-vs-random race, and the time-shuffle scramble — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | causal EMA19−EMA39 breadth osc, one-sample HAC *t* vs the beta trap, the random-entry Welch test, the shuffled-breadth placebo, per-ticker deltas, costs, and a synthetic planted-bounce control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`mcclellan_oscillator/`](mcclellan_oscillator/). Oscillator is EMA19−EMA39 of net advances (causal); the up-cross is read on close of *t*, entry is the next close (one lag). Breadth is a **proxy** built from a surviving liquid-ETF basket — a coarse stand-in for true exchange breadth — but this is a single-instrument timing study, so the random-entry baseline neutralizes the drift/survivorship. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
