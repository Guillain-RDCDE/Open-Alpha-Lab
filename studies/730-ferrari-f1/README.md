# Study 730 — Ferrari-F1 🏁

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does RACE pop the Monday after a Ferrari win? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The day(0) "win pop" is a coin flip: **+0.276%**, one-sample *t* = **+1.18**, hit rate **54.2%** (Wilson [35.1%, 72.1%]), random-Monday placebo **p = 0.214**. No immediate-reaction cut clears *t* ≥ 2. |
| **Tradability** — could a fan bank it? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Enter the first close *after* the win (zero look-ahead), hold a week: **+0.478%** net (*t* = 0.88), placebo **p = 0.274** — a random-window draw. There is no edge to size. |
| **Fan-halo, or championship fundamentals?** | ![Misattributed](https://img.shields.io/badge/Misattributed-8b949e?style=flat-square) | The *one* cut past both bars — the 2017-18 **title-contender** 1-week drift, **+2.47%**, *t* = **2.73**, placebo **p = 0.017** — vanishes (goes **−0.50%**) for the pure sporadic wins. What little moves RACE tracks a live championship, not fan mood. |

> **In one sentence:** across all 24 Ferrari Grand Prix wins since the 2015 NYSE IPO, RACE's Monday reaction is statistically a random Monday (+0.28%, *t* = 1.18, placebo *p* = 0.21), the honest zero-look-ahead trade nets a random-window +0.48% (*t* = 0.88), and the only signal that survives — a 1-week drift — lives entirely in the two seasons when a win updated a live title fight and *reverses* for the one-off wins that are pure fan sentiment, so the brand-halo story is the wrong label for the little that's there.

## What we tested

Ferrari (`RACE`) is the rare listed company whose brand *is* a Formula 1 team, so the
folklore — the retail-and-motorsport-media cousin of Edmans, García & Norli's (2007)
sports-sentiment effect — says a Grand Prix win gives the stock a tifosi brand-halo pop.
We hardcode every Ferrari F1 **victory** in the RACE-listed era (24 wins, 2017→2024 —
Ferrari were winless in 2016, 2020, 2021 and 2025), run an event study on RACE's abnormal
return vs `SPY` around each win with one documented no-look-ahead lag (the race runs on a
non-trading Sunday, so day(-1) = last close before, day(0) = first close after), test it
with a one-sample *t* across the independent race events, a Wilson hit rate, a
multi-seed random-calendar placebo and a costed capture timer, and split the wins into
title-contender vs sporadic seasons to ask whether any effect is fan sentiment or
championship fundamentals. **As-of 2026-06-30.**

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, why the "brand *is* the team" makes it tempting, the coin-flip win pop, and the twist that the only whisper is about the title race, not the tifosi |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the one-sample-*t* battery, the random-calendar placebo, the day(0)-vs-week anatomy, the contender-vs-sporadic Welch split that reattributes the effect, the costed capture, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`ferrari_f1/`](ferrari_f1/). The win calendar is hardcoded from STATS F1 /
Formula1.com; `RACE` and `SPY` are fetched via yfinance (total-return adjusted). Benchmark
choice (SPY, not an auto/luxury peer) and the 1-week back-to-back overlap are both named
in the docs, not swept aside. **Not investment advice** — research & education. See
[LICENSE](../../LICENSE).*
