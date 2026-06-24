# Study 404 — Shooting Star ☄️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the star call the top? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Shorting the shooting star (small body, long upper wick, after an uptrend) earns a **−6.1 bps** edge over the short base rate at 5 days, HAC *t* = **−1.12**, label-shuffle *p* = **0.87**. No horizon (1/3/5/10d), no trend filter, and no name carries a positive short edge past **+2** — the lone *t* > 2 among 26 names is textbook multiple testing. Win-rate ~**45%**, below a coin. **Survivorship** named on the Signal axis (for a bearish claim the tilt cuts *against* the short). |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The gross edge is already ≤ 0, and the star is a **short** — the costly side: spread twice **plus** borrow per day. Net at 5 days ≈ **−26.1 bps/event**. There is no positive break-even cost. |
| **"Calls the top?"** | ![Busted](https://img.shields.io/badge/Calls_the_top%3F-Busted-8b949e?style=flat-square) | Pool every star and short it: edge **−3.9 bps** at HAC *t* = **−2.15** (significant, *wrong sign*). The long upper wick, if anything, predicts **continuation up** — the opposite of the "exhaustion / reversal" story. |

> **In one sentence:** the shooting star's long upper wick *looks* like a top, but on 26 US large-caps + SPY (decades of daily bars) shorting it earns no edge over the base rate at any horizon (5-day edge −6.1 bps, HAC *t* = −1.12, placebo *p* = 0.87), the "fade only strong rallies" refinement doesn't rescue it, and pooling the geometry the short actually loses *significantly* (*t* = −2.15) — the wick leans toward continuation, not reversal.

## What we tested

We detect the **shooting-star geometry** by precise OHLC rules — upper wick ≥ 2× the body, body near the bottom of the range, body ≤ 35% of the day's range — on a fixed basket of **26 liquid US large-caps + SPY** (yfinance daily, un-adjusted, cache-first; full history, longest back to 1962). We split each occurrence by **prior trend** (the *shooting star* is the shape after an uptrend, the bearish claim) and trade it **short**: enter at the *next* close (one execution lag), hold 1/3/5/10 days, and measure the conditional short return against each name's own **short base rate**. The Signal axis tests the pooled edge with a one-sample **HAC *t*** and a per-name **label-shuffle placebo**; Tradability charges one-way costs ×2 **plus short borrow**. A myth-check asks whether a **trend-strength filter** ("only fade strong rallies") changes the answer, and a deterministic **synthetic positive control** (a *planted* post-star decline) confirms the engine can detect a real top when one exists. Survivorship (the basket is names still trading in 2026) is named on the Signal axis.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a shooting star is, why the long upper wick *looks* like exhaustion, why on large-caps it usually isn't a top, and why shorting it costs you — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the exact-geometry detector, trend split, short-side forward 1/3/5/10-day returns vs base rate, HAC *t* + label-shuffle placebo, per-name breakdown, the trend-filter myth-check, cost+borrow landscape, and the synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`shooting_star/`](shooting_star/). Un-adjusted daily OHLC; forward returns are **price-only**. Basket is **survivors** — named on the Signal axis. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
