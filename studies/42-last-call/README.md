# Study 42 — Last-Call 🕛

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do returns cluster at the turn of the month? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | Yes, strongly. The four [-1,+3] days (~19% of all days) earned **11.0 bp/day** vs **1.9 bp/day** the rest of the time on the S&P 500 (price index), 1950–2026 — Welch t **5.1**. One of the most replicated calendar facts there is. |
| **Tradability** — can you trade the window? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | No — even on an rf-consistent footing. Holding exactly the window on SPY, cash leg credited at the T-bill, makes **5.5%/yr vs buy-and-hold's 10.8%**, and its excess-of-cash Sharpe (**0.41** gross) still trails buy-and-hold's **0.51**; 12 round-trips/yr of costs widen the gap. |
| **"Still alive?"** | ![Faded](https://img.shields.io/badge/Faded-8b949e?style=flat-square) | The premium fell from **11.7 bp/day pre-2008 (t = 6.3)** to **0.6 bp/day since (t = 0.1)** — and the decline itself is significant (change t = **2.3**). |

> **In one sentence:** the turn-of-the-month effect is genuinely *real* and large — and a perfect trap: a window-only book compounds half of buy-and-hold and trails it on excess-of-cash Sharpe even before costs (you're in cash four-fifths of the time), and the premium has faded to statistical zero since ~2008.

## What we tested

A textbook calendar anomaly: equity returns concentrate in a four-day window straddling month-end (Lakonishok & Smidt 1988; McConnell & Xu 2008; [paperswithbacktest](https://github.com/paperswithbacktest/awesome-systematic-trading) lists it at Sharpe `0.305`). We confirm the effect on the S&P 500 price index back to **1950** (mean return on turn-of-the-month days vs the rest, with a t-stat), then ask the question that decides it: **does *trading* the window beat just holding the index, after costs?** The window is calendar-known ex ante, so the book holds exactly the [-1,+3] days (no signal lag), its cash leg earns the T-bill, the Sharpe race is run excess-of-cash on both sides, and the decay gets a formal sub-period test rather than a bare assertion. The offline control is a synthetic daily world with an injected turn-of-the-month bump (and a null).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "11 bp a day!" is real and still a losing trade, and how an effect can be true and dead at once |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the TOM/non-TOM split with its t-stat, the window-vs-buy-and-hold race with the cash leg credited (raw and excess-of-cash Sharpe), the time-in-market illusion, the tested sub-period decay |

The fingerprinted real-data run (S&P 500 1950–2026 fp `54fd9e780da2` / SPY 1993–2026 fp `d824e220dbca`) is in [docs/results.md](docs/results.md). Reproduce via [examples/verify.py](examples/verify.py) (`--fetch` to download); the offline machinery proof runs on the synthetic world in [last_call/data.py](last_call/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
