# Study 06 — Clockwork-Vol ⏰

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is there a real fixed-period cycle? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | At every period the tweet names — VIX **40d** & **80d**, stocks' 20-week (**100d**), 1-year (**250d**), 4-year (**1000d**) — the periodogram peak sits **inside** the red-noise envelope (p = **0.998 / 0.9995 / 0.994 / 0.9995 / 0.765**): red noise fakes peaks *taller* than the VIX's almost every time. The only thing clearing the 99% envelope is a ~**15.6-session** wiggle (ratio 1.10), not the claimed clocks. |
| **Tradability** — does timing it pay? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Walk-forward, the projected cycle calls the next move at **49–51%** (a coin; p = 0.745 / 0.720 / 0.089 against a 999-rotation random-phase null). The tradeable expression (long the S&P while the VIX cycle is projected to fall — a harness that **banks a planted cycle**, so its silence is the market's) earns **Sharpe 0.30** — *below* buy-and-hold's **0.56** and *below* the random-phase null's mean **0.37** (p = **0.88**): diluted beta from 59% exposure, not timing. One beat-7 rescue is reported honestly: hard-wiring the literal **80-day** clock clears its own null raw (skill 0.533, p ≈ 0.026; trade Sharpe **0.69** vs 0.56 buy-and-hold, p ≈ 0.01) — but it's a single hand-picked period in an uncorrected ~5×3×4×4 grid of looks, contradicted by its own spectrum (no 80d peak, p ≈ 0.9995), both legs cut from one fit on one tape: a fragile fragment to pre-register, not an edge ([extensions](docs/extensions.md)). |
| **A fixed clock?** — does the period even hold? | ![Not supported](https://img.shields.io/badge/Not_supported-8b949e?style=flat-square) | The "dominant cycle" wanders from **83 to 333 sessions** (mean 227, σ **82**) across rolling 4-year windows — it has to be re-drawn every few months. A period that won't sit still isn't a clock; it's a curve-fit. |

> **In one sentence:** the VIX's tidy 40-/80-day "cycles" don't clear what AR(1) red noise invents on its own, their period won't hold still, and a walk-forward forecast built on them is a coin flip that loses to buy-and-hold — one fixed-80-day variant does edge past its null raw (p ≈ 0.03, at face value and uncorrected for the grid of variants tried) but contradicts its own spectrum and stands on a single market, so what looks like a clock remains the eye reading rhythm into persistent noise.

## What we tested

A [cycles-analysis thread](https://x.com/Namzes_G) reads the VIX the way an astronomer reads an orbit — as a stack of **fixed-period cycles** you can project forward: an 80-day cycle with a nested 40-day one, projected to date the next vol low and the matching stock 20-week-cycle low weeks ahead. The claim takes the VIX's genuine mean-reversion and makes it *precise and datable* — not "vol tends to come back down" but "the next low is due on roughly *this* day, because the clock says so." If real, it would be one of the cleanest market-timing edges in existence — so we held the claimed periods to the red-noise floor.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the story + the stakes, plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the full method: periodogram, red-noise envelope, stability, walk-forward, trade |

The real run lives in [docs/results.md](docs/results.md); the worked beat-7 rescues (fixed periods, amplitude gate, stock clocks) in [docs/extensions.md](docs/extensions.md); reproduce offline with [examples/](examples/).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
