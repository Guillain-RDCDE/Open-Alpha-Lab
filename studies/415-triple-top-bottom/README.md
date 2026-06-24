# Study 415 — Triple Top & Bottom 📉

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the breakout carry information? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Across **356** confirmed triple-bottom breakouts on SPY + 29 large-caps (21.4y), the forward excess over base rate peaks at **+0.34% at 5 days** — but at one-sample *t* = **1.75** (HAC **1.82**), *under* the **t ≥ 2** bar, with a random-date placebo it can't beat (**p = 0.37**). It **decays to −0.69% by 40 days** and **evaporates** under the detector-strictness sweep. Indistinguishable from the tape's own up-drift. (Survivorship tilts *for* the figure — and it still fails.) |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Net of 5 bps/leg the 5-day excess is **+0.24%** (inside the placebo cloud) and the recommended multi-week hold is **−0.79%**. No deployable edge, nothing to scale. |
| **"Reliable reversal"?** | ![Busted](https://img.shields.io/badge/Reliable_reversal%3F-Busted-8b949e?style=flat-square) | The bearish **triple top** short is **net-negative** (**−1.38% at 20d**, **−1.94% at 40d**): the figure does **not** reverse symmetrically. The long side only looked alive because it borrows market drift; the short side fights it and loses. |

> **In one sentence:** a clean, objective detector for the textbook "three failures at one level" figure finds plenty of triple bottoms across two decades of large-caps, but buying the confirmed breakout beats the stock's own drift by a noisy +0.34% at best (*t* = 1.75, placebo *p* = 0.37), decays negative by 40 days, vanishes when you change the tolerance, and the bearish triple-top twin loses outright — so the pattern is a shape the eye loves and the tape ignores.

## What we tested

Chart figures are **partly subjective**, so we wrote down the closest **mechanical** definition we could and said so: three swing pivots (lows for a triple bottom, highs for a triple top) clustered within a tolerance of one price level, separated by genuine bounces, then a **confirmed close through the neckline** as the entry. Running it on a fixed **30-name large-cap basket + SPY** (yfinance daily auto-adjusted OHLC, 2005 → 2026-05-29, as-of 2026-05-31), we measure the forward **5/10/20/40-day** return after each breakout, in the trade's intended direction, **net of each name's own base rate** — entering one day after the breakout (no look-ahead). The Signal axis tests the pooled excess with a one-sample and HAC *t* and a **same-tape random-date placebo** (the honest control for market up-drift); Tradability charges a round trip; the myth-check runs the bearish **triple top** as a short to ask whether the figure reverses symmetrically. A deterministic synthetic control with *planted* triple bottoms confirms the harness banks a real edge (placebo *p* = 0.02) and refuses a null (placebo *p* = 0.83). Survivorship (a surviving-names basket, which tilts *for* the figure) is named on the Signal axis.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a triple top/bottom is, a real detected example drawn by the code, why "three failures" feels like a wall but isn't, why the long side borrows market drift, and why the bearish twin loses — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the mechanical detector, forward 5/10/20/40-day excess over base rate, one-sample + HAC *t*, a same-tape random-date placebo, a detector-strictness sweep, the triple-top myth check, and a synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`triple_top_bottom/`](triple_top_bottom/). Detector is one mechanical definition of a partly-subjective figure — said loudly on the Signal axis. Basket is **survivors** (tilts *for* the figure) — named on the Signal axis. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
