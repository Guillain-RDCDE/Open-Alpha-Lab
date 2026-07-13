# Study 741 — Cicada-Brood 🦗

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — are periodical-cicada emergence springs special for the S&P 500? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Cicada-spring abnormal CAR **+0.39%**, one-sample *t* = **+0.37** across 24 emergence years; random-year placebo **p = 0.17**; Welch vs non-cicada springs *t* = **+0.87**; no event-anatomy offset clears \|*t*\| ≥ 2. The eye-catching 75% up-rate (and the 5/5 "famous broods" cut) are small-sample/selection illusions on top of equities' ordinary drift. |
| **Tradability** — can you deploy the (perfectly foreseeable) calendar trade? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Long-SPY-over-the-cicada-spring overlay: the +2.25 *t*-vs-zero is pure two-month equity **beta**, not an edge — the honest **excess over the every-spring baseline is +60 bps, *t* = +0.56** (gross), fading to +40 bps net of costs. A signal known decades ahead that still buys you nothing. |
| **Cicada indicator?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | A brood emerges *somewhere* in **24 of the 30** years 1996–2025 — "a cicada year" is very nearly the whole calendar. There is, by construction, nothing to find, and the rigorous apparatus finds exactly that. |

> **In one sentence:** across 24 periodical-cicada emergence years since 1996 the S&P 500's cicada-spring return is a statistically-nothing +0.39% abnormal (*t* = 0.37, placebo *p* = 0.17), a "trade" you could have scheduled decades in advance earns only the beta you'd get holding any spring, and — the punchline — a brood emerges in most years anyway, so this is a purpose-built demonstration that a famous fixed calendar produces exactly the noise you'd expect.

## What we tested

There is no serious "cicada indicator" — this is a **deliberately silly spurious-pattern
demo**, built to show the desk's event-study apparatus returning an honest *nothing*. We
steelman the strongest version of a market-almanac-style calendar signal (in the family
of the Super Bowl indicator and "Sell in May") using the one calendar that is genuinely
fixed and famous: the 13-/17-year periodical cicada (*Magicicada*) brood emergences,
hardcoded from the University of Connecticut / Cooley brood chart (**30 mapped
emergences, 24 distinct years, 1996→2025**). We run an event study on **total-return SPY**
around each emergence's May–June window — constant-mean abnormal CAR, a one-sample *t*
across independent years, a Wilson up-rate, a 20-seed random-year placebo (same window,
so the season is controlled), a Welch cicada-vs-quiet-spring contrast — and a costed
overlay graded on its **excess over the every-spring baseline** (alpha, not beta),
noting the study's cute one-line execution convention: **zero look-ahead, because the
emergence year was on the calendar since the last emergence 17 years earlier.** A
deterministic synthetic tape with a *planted* spring bump is the positive control.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a fixed, famous calendar *feels* like it should mean something, what the tape actually shows, and how a 75%-up-rate and a "5/5 famous broods" streak are the noise selection manufactures |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the one-sample-*t* battery across independent years, the season-controlled random-year placebo, the event anatomy, the beta-vs-alpha timer (excess over the every-spring baseline), and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`cicada_brood/`](cicada_brood/). The brood calendar is hardcoded from the
UConn/Cooley *Magicicada* brood chart (cross-checked against USFS/USDA sources); SPY is
fetched via yfinance as a total-return series (no proxy — SPY is the real, tradable
instrument). No survivorship (the whole-market index; a fixed astronomical schedule).
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
