# Study 441 — Camarilla Pivots 🧱

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do L3/H3 get respected? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | First-touch reversion off the Camarilla reversal levels is **−0.017%** over ~1h (one-sample *t* = **−0.44**, hit-rate **48%**). The HAC *t* of **−2.59** is significant only with the **wrong sign** (mild *continuation*, not reversion). No respect-the-level edge. |
| **Tradability** — could you trade it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Net of a **2 bps** round-trip the reversion is **−0.057%** per touch — on the **tightest-spread names there are**. There was never a gross edge to scale. |
| **"The line is special"?** | ![Busted](https://img.shields.io/badge/Line_is_special%3F-Busted-8b949e?style=flat-square) | Random control lines at the **same distance** revert **+0.250%** — *better* than L3/H3 (real − control = **−0.267%**, P[level beats random] ≈ **0**). The synthetic control proves the test *would* catch a real level; there just isn't one. |

> **In one sentence:** the famous Camarilla reversal levels L3/H3 are arithmetic on yesterday's range, and on 5-minute bars for the five most liquid US names they get respected **no more than a line drawn at random** — first-touch reversion is −0.017% (*t* = −0.44), a random control line actually reverts *better* (+0.250%), and a 2 bps round-trip pushes it negative; a synthetic positive control confirms the test would light up (*t* = 6.86) if any such respect existed.

## What we tested

We rebuild the Camarilla claim as a clean intraday **event study**. Per session we compute the Camarilla ladder from the **prior** day's high/low/close (`H3/L3 = C ± R·1.1/4`, `H4/L4 = C ± R·1.1/2`, pivot `P = (H+L+C)/3`) — known at the open, no look-ahead — then find the **first** 5-minute bar that touches **L3** (long) or **H3** (short), enter the **next** bar (one-bar execution lag), hold ~1 hour, and measure the forward reversion **toward the central pivot**. The Signal axis tests that reversion against zero with a one-sample *t* and an event-clustered **HAC** *t*; the decisive **myth-check** runs the identical rule on **27,442 random control lines** at the same distance and bootstraps the real − random difference; Tradability charges a round-trip spread. A deterministic synthetic panel with a *planted* respect-the-level pull confirms the engine is faithful (zero edge can't fake significance, a real one lights up). **Loud caveat:** free intraday history is ~60 days, so this is 5 names over ~59 sessions — a thin power budget — but the random control already *beats* the real level, so the verdict is not a power problem.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what Camarilla pivots are, why "price bounced at L3" isn't evidence, the random-line comparison in plain language, and why the spread finishes it off |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | first-touch reversion event study, one-sample + HAC *t*, the random-control bootstrap (the headline), hold/per-name robustness, costs, and a synthetic planted-respect power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`camarilla_pivots/`](camarilla_pivots/). Real tape is yfinance **5-minute** price-only bars (SPY/QQQ/AAPL/MSFT/NVDA), 2026-03-30 → 2026-06-23, in-progress final session dropped. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
