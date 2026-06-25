# Study 459 — Hikkake Pattern 🪤

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the trap forecast the reversal? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The direction-signed hikkake does **not** beat a drift/**exposure-matched random** baseline: trap − random = **−3.4 / −13.1 / −22.0 / −72.7 bps** at 5/10/20/60 days, and the trap-vs-random Welch *t* is **never positive** (60d **−2.23**, *p* = 0.026 — in the *wrong* direction for believers). The negative one-sample *t*'s (20d **−2.57**) are just a **short-heavy rule fading the index drift**, not a forecast. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The signed return is **negative gross and net** (−34 bps at 20d); costs only deepen the hole. A short-heavy rule on an up-drifting tape fights the tide. Nothing to scale. |
| **"Does the false-breakout trap forecast?"** | ![Busted](https://img.shields.io/badge/Trap_forecasts%3F-Busted-8b949e?style=flat-square) | Randomly **flip** which way each trap points and the result is unchanged: **98%** of direction-scrambled hikkakes match or beat the real one (*p* = **0.978**). The trap's one piece of content — its direction — carries no information. |

> **In one sentence:** The hikkake looks clever because you only label it *after* the snap-back — encode it mechanically (inside bar → false break → snap-back, no eyeballing) and fire it **1 438 times** across 5 indices over 21 years, and it makes **negative money**, *loses* to the same trades on random days at every horizon, and its defining direction is a coin flip (flip it → *p* = 0.98): a post-hoc chart label, not a forecast.

## What we tested

We encode the tightest mechanical version a proponent would accept. An **inside bar** is `high_i < high_{i-1}` and `low_i > low_{i-1}`; within a 3-bar window a close must **break** beyond that range and then **snap back** through it (the false-breakout trap). We trade the reversal — long after a failed downside break, short after a failed upside break — entered at the **next close** (one documented lag), and measure the direction-signed forward 5/10/20/60-day return on SPY, QQQ, IWM, DIA and GLD (yfinance daily total-return, 2005→2026). Because the signal is **mixed long/short** (548 long / 890 short), the Signal axis is **trap vs a random baseline matched on the same long/short mix** (a Welch *t*) — the only honest test on a drifting tape — plus a **scrambled-direction placebo** that keeps every trap date but flips its direction, the direct test of "does *which way the trap points* matter?". Tradability charges costs on every trade. A deterministic synthetic control with a *planted* trap-reversal proves the detector is live (edge 0 → *t* = −1.42, below the bar; planted reversal → *t* = +8.89), so the negative real-tape result is a genuine "nothing there".

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a hikkake is, why a short-heavy rule on a rising market looks bad, the trap-vs-random race, and the direction-flip — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | mechanical inside-bar traps, one-sample HAC *t* vs the exposure trap, the matched random-entry Welch test, the scrambled-direction placebo, per-ticker deltas, costs, and a synthetic planted-trap control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`hikkake_pattern/`](hikkake_pattern/). Trap = inside bar + false break + snap-back within a 3-bar window; entry is the next close (one lag), return signed by direction. Basket is surviving liquid ETFs — but this is a single-instrument pattern study, so the exposure-matched random baseline neutralizes the drift/survivorship. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
