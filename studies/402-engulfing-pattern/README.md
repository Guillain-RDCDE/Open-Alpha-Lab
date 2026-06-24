# Study 402 — Engulfing Pattern 🕯️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does it predict the next day? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The signed engulfing return (long bullish / short bearish) is **significantly negative** at every horizon — **−0.05% at 1 day (HAC *t* = −3.79)**, worsening to **−0.17% at 5 days (*t* = −5.74)**. Win-rate **48%**, placebo *p* = **1.000**. The pattern doesn't predict the reversal — it predicts the **opposite**. |
| **Tradability** — could you trade it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The edge is **negative before costs**, so there is nothing to harvest; the round trip + short borrow only widen the loss to **−0.15% / −0.29%** per event. No capacity question even arises. |
| **A reliable reversal?** | ![Busted](https://img.shields.io/badge/Reliable_reversal%3F-Busted-8b949e?style=flat-square) | The **bearish** leg keeps *rising* (5-day *t* = **−10.46**); the **bullish** leg's small gain is pure beta — it **underperforms** just holding for a day (+0.46% vs +0.53% unconditional at 10d). The textbook trend/volume filters don't flip it. |

> **In one sentence:** the most-taught reversal candle is busted on liquid US large-caps — signed long after bullish and short after bearish, it earns a *significantly negative* −0.05% the next day (HAC *t* = −3.79) and only gets worse with horizon, a sub-50% win-rate where the bearish leg keeps climbing and the bullish leg merely rides the market's drift; the harness's synthetic control proves it *would* see a real reversal if one existed.

## What we tested

We detect every **engulfing** candle (the precise real-body rule: opposite colours, the current
body strictly larger and fully containing the prior one) across a fixed **30-name** basket — 29
long-listed liquid US large-caps + **SPY**, yfinance daily 2005→2026, cache-first. Confirming at the
close, we **enter the next open** (one execution lag) and measure the forward **1 / 3 / 5 / 10-day**
return *signed by the pattern direction* (long after bullish, short after bearish) against the
unconditional base rate, with a Newey-West **HAC t**, a label-shuffle placebo, realistic costs, and a
trend/volume **myth-check**. A deterministic synthetic control with a *planted* day-after reversal
confirms the engine can bank a real edge and that a random walk can't fake one. Survivorship (the
basket is names still trading in 2026) is named on the Signal axis — and it works *for* the claim,
yet the pattern still anti-predicts.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what an engulfing candle is, what "reversal" should look like, why the bearish version keeps rising and the bullish version is just the market drifting up — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the real-body detector, forward 1/3/5/10-day signed returns, HAC *t* vs zero + a coin-sign placebo, leg split, costs, the trend/volume myth-check, and a synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`engulfing_pattern/`](engulfing_pattern/). Bars are total-return adjusted (`auto_adjust=True`); the signed return is long after bullish / short after bearish. Basket is **survivors** — named on the Signal axis. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
