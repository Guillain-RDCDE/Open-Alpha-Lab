# Study 696 — Double-Bottom 📉🇼

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the breakout carry information? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Across **560** confirmed double-bottom breakouts on SPY + 29 large-caps (21.5y), the forward excess over each name's own base rate never clears **t ≥ 2** in the pattern's favor at any horizon — the closest approach (10 days) is **−1.91**, the wrong sign. The random-date placebo is never beaten (*p* = 0.50–0.64). The classic measured-move target hits **80.7%** of the time — but a magnitude-matched random-walk placebo already hits **79.7%** (*z* = 0.57). A robustness sweep finds no tolerance where the edge turns reliably positive. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | A "long timer" (hold to target-or-timeout) shows **+2.94% gross/trade** at *t* = 7.80 vs zero — a slam dunk *until* raced against a holding-period-matched base rate, where the excess collapses to **+0.125% at HAC *t* = 0.32**. No deployable edge; 5–10 bps costs only worsen the already-flat headline horizons. |
| **"Target hit rate beats a coin flip of the same size?"** | ![Busted](https://img.shields.io/badge/Target_hit_rate%3F-Busted-8b949e?style=flat-square) | The 80.7% measured-move hit rate looks striking alone — but a magnitude-matched placebo (same target distance, random entry) already clears it 79.7% of the time (*z* = 0.57). The "accuracy" is a function of the target being a modest distance on a rising basket, not of the W-shape predicting anything. |

> **In one sentence:** a clean, objective detector for the textbook "two troughs at one level, buy
> the breakout" figure finds 560 double bottoms across two decades of large-caps, but buying the
> confirmed breakout never beats the stock's own drift at *t* ≥ 2 at any horizon, the celebrated
> measured-move target hits about as often as a random walk asked to travel the same distance, and
> the "long timer" P&L that looks like +7.8 *t* against zero shrinks to statistical nothing (*t* =
> 0.32) once raced fairly against the market's own up-drift over the same holding period.

## What we tested

Chart figures are **partly subjective**, so we wrote down the closest **mechanical** definition
we could and said so: two swing-pivot lows within a tolerance of one price level, separated by a
**genuine** intervening rally to a "neckline" peak (not a flat shelf), then a **confirmed close
above the neckline** as the entry — the classic W. Running it on a fixed **30-name large-cap
basket + SPY** (the same basket as siblings 415-triple-top-bottom and 695-inverse-head-shoulders;
yfinance daily auto-adjusted OHLC, 2005 → 2026-06-30, as-of 2026-06-30), we measure the forward
**5/10/20/40-day** return after each breakout, **net of each name's own base rate** — entering one
day after the breakout (no look-ahead). The Signal axis tests the pooled excess with a one-sample
and HAC *t* and a **same-tape random-date placebo**; a robustness sweep checks the tolerance isn't
doing the work. Two extra arbiters go beyond a plain forward-return study: the classic
**measured-move target** (trough-to-neckline height, projected from the neckline) is tested for a
hit rate against a **magnitude-matched placebo**, and a **"long timer"** — hold to
target-or-timeout — is raced against a **holding-period-matched** base rate, net of 5/10 bps
costs. A deterministic synthetic control with *planted* double bottoms confirms the harness banks
a real edge (placebo *p* = 0.009) and refuses a null across 20 seeds even though a single-seed
naive *t* can read misleadingly high. Survivorship (a surviving-names basket, which tilts *for*
the figure) is named on the Signal axis. **Dedup:** [189-double-top](../189-double-top/) already
detects both double-tops *and* double-bottoms but on a fixed-horizon-vs-random-placebo protocol
with no measured-move or long-timer test; [415-triple-top-bottom](../415-triple-top-bottom/) is
the three-tap version; [695-inverse-head-shoulders](../695-inverse-head-shoulders/) is the
three-trough, asymmetric-head version; [694-matching-low](../694-matching-low/) is the two-*candle*
micro version of the same "tested a level twice" idea. None of them run this study's
base-rate-neutral excess + measured-move + long-timer bar on the two-trough figure.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a double bottom is, a real detected example drawn by the code, why the target hit rate is a magic trick, why the "long timer" looks great until you ask the fair question — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the mechanical detector, forward 5/10/20/40-day excess over base rate, one-sample + HAC *t*, a same-tape random-date placebo, a detector-strictness sweep, the magnitude-matched measured-move test, the holding-period-matched long timer, and a synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`double_bottom/`](double_bottom/). Detector is one mechanical definition of a
partly-subjective figure — said loudly on the Signal axis. Basket is **survivors** (tilts *for*
the figure) — named on the Signal axis. **Not investment advice** — research & education. See
[LICENSE](../../LICENSE).*
