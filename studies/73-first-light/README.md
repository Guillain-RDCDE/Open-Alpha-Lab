# Study 73 — First-Light

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | ORB gross **+2.18 bps/trade**, HAC *t* = **+0.40** across SPY/QQQ/IWM/TQQQ (n=240); beats a coin by +2.85 bps but far below |*t*| ≥ 2; bootstrap Sharpe CI [−1.26, +1.67]. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The thin +2.18 bps gross is fully absorbed by ~1–2 bps round-trip cost; the TQQQ version adds ~50 bps/day leverage drag that turns it clearly negative. |
| **Beats a coin?** | ![Not_Confirmed](https://img.shields.io/badge/Beats_a_coin%3F-Not__Confirmed-8b949e?style=flat-square) | +2.85 bps above a random-direction control on the same entries, but statistically invisible in 60 days of data; literature support (Zarattini & Aiolfi 2023) relies on leverage and unrealistic costs. |

> **In one sentence:** the 5-minute ORB shows a directionally positive but statistically unconfirmed gross edge on the real 60-day tape (+2.18 bps, *t* = +0.40) — the breakout is doing something right, but the data window is too short to call it signal and costs absorb the mean long before confirmation.

## What we tested

Zarattini & Aiolfi (2023, SSRN) report spectacular back-tested returns on QQQ/TQQQ using the Opening Range Breakout: take the first 5-minute (or 15-minute) bar of the RTH session as the range, go long on a close above the high and short below the low, hold with a stop of 1× range-size and exit flat at 16:00. We steelman this as a clean directional claim and test it honestly: with a **random-direction control** on the same entry timestamps, a **cost sweep** at the natural turnover of ~1 trade/day, and an explicit **TQQQ leverage-drag** calculation. The 5-min ORB is the canonical variant; the 15-min variant is included for comparison. A deterministic synthetic tape with a tunable opening-drift knob serves as the positive control, confirming the engine recovers an edge when one is planted.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the recipe, what the opening range is supposed to capture, the coin test in plain English, why TQQQ leverage isn't free |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-ticker HAC *t*, bootstrap Sharpe CI, cost sweep, leverage-drag model, the synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`first_light/`](first_light/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
