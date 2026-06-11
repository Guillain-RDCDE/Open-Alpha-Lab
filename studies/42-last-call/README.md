# Study 42 — Last-Call 🕛

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do returns cluster at the turn of the month? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | Yes, strongly. The four [-1,+3] days (~19% of all days) earned **11.0 bp/day** vs **1.9 bp/day** the rest of the time on the S&P 500, 1950–2026 — Welch t **5.1**. One of the most replicated calendar facts there is. |
| **Tradability** — can you trade the window? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | No. A long-the-window-else-cash rule on SPY makes **4.3%/yr (Sharpe 0.55)** vs buy-and-hold's **10.8% (0.64)** — you forfeit 60% of the return to sit in cash 81% of the time, and 12 round-trips/yr of costs widen the gap. |
| **"Still alive?"** | ![Faded](https://img.shields.io/badge/Faded-8b949e?style=flat-square) | The premium fell from **13.8 bp/day (1950–87)** to **4.8 bp/day (2008–on)**, now indistinguishable from an ordinary day's **4.1 bp**. |

> **In one sentence:** the turn-of-the-month effect is genuinely *real* and large — and a perfect trap: a window-only book underperforms simply holding the index (you're in cash four-fifths of the time), costs eat the thin edge, and the premium has faded to an ordinary day's return since ~2008.

## What we tested

A textbook calendar anomaly: equity returns concentrate in a four-day window straddling month-end (Lakonishok & Smidt 1988; McConnell & Xu 2008; [paperswithbacktest](https://github.com/paperswithbacktest/awesome-systematic-trading) lists it at Sharpe `0.305`). We confirm the effect on the S&P 500 back to **1950** (mean return on turn-of-the-month days vs the rest, with a t-stat), then ask the question that decides it: **does *trading* the window beat just holding the index, after costs?** We build the long-window-else-cash rule on SPY across cost levels, and split the sample into thirds to track the decay. The offline control is a synthetic daily world with an injected turn-of-the-month bump (and a null).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "11 bp a day!" is real and still a losing trade, and how an effect can be true and dead at once |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the TOM/non-TOM split with its t-stat, the window-vs-buy-and-hold race after costs, the time-in-market illusion, the sub-period decay |

The fingerprinted real-data run (S&P 500 1950–2026 / SPY 1993–2026, fp `54fd9e780da2`) is in [docs/results.md](docs/results.md). Reproduce via [examples/verify.py](examples/verify.py) (`--fetch` to download); the offline machinery proof runs on the synthetic world in [last_call/data.py](last_call/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
