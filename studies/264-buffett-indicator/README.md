# Study 264 — Buffett-Indicator

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | The slope has the **right (negative) sign** and a deep valuation-predictability literature behind it, but on the real tape the HAC *t* is just **−0.92** (R² < 1%; n = 55 annual obs). It clears neither the *t* ≥ 2 bar nor the higher hurdle valuation ratios now face — price-only forward returns. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | A real-time "leave when expensive" rule (expanding median) sits in cash **76%** of the years and earns **1.3%/yr** vs **8.1%/yr** buy-and-hold (excess *t* = **−3.76**). Market-cap/GDP has been above its own history for ~30 years while the market compounded anyway. |

> **In one sentence:** the Buffett Indicator is a *thermometer, not a timer* — it points the right way (cheap years out-return expensive years by ~2 pp), but a real-time rule built on it kept you out of a generation-long bull market, and at the one-year horizon the signal is statistically indistinguishable from zero.

## What we tested

The Buffett Indicator (Buffett & Loomis, *Fortune* 2001) is total US market
capitalisation divided by GDP — "probably the best single measure of where
valuations stand." We hardcode the year-end indicator (1971–2025, market cap /
GDP in % of GDP) in `data.py`, join it to S&P 500 (`^GSPC`) **price** calendar-
year returns, and ask whether the *level* of the indicator at year-end Y
forecasts the return in year Y+1 (one full year of execution lag). We run a
predictive regression with a Newey-West HAC *t*-stat, a cheap-vs-expensive
tercile sort, and — the honest test — a **real-time, expanding-median** timing
strategy (no hindsight thresholds) against buy-and-hold. A synthetic positive
control confirms the engine recovers a planted valuation→return link when one
exists; the real tape shows the one-year signal is too weak to trade.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, the rising-ratio chart, cheap-vs-expensive years, why "leave the party" fails in plain English |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | predictive regression + HAC *t*, in-sample vs real-time (expanding median), the timing wipeout, the synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`buffett_indicator/`](buffett_indicator/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
