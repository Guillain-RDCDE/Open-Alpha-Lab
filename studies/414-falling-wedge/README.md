# Study 414 — Falling Wedge 📐

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the up-break edge exist? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The up-break excess-over-base-rate clears a naive bar at the slow horizons (**+1.26% at 20 days, one-sample *t* = 2.19, HAC *t* = 2.26**), but the same-tape **placebo *p* = 0.30** (random dates win ~3-in-10), the strictness sweep keeps *p* at ~0.26–0.34, and a **zero-edge synthetic control finds no positive edge** (placebo *p* = 0.885). At the **breakout day itself the excess is negative** (5-day *t* = −2.57). The number is real; its attribution to the figure is not. **WEAK, not REAL.** |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | What you'd "trade" is the names' own down-then-up mean reversion — exactly what random dates capture. The breakout-day rule **loses** at 5 days; the SPY-only version rests on **2** events in 21 years; costs are a footnote because there's no placebo-clean edge to tax. Nothing to deploy. |
| **"Bullish — breaks up specifically"?** | ![Busted](https://img.shields.io/badge/Bullish%3F-Busted-8b949e?style=flat-square) | The **down**-break of the *identical* figure also drifts up (+1.80% / +3.79% at 20/40 days, *t* = 2.00 / 2.97) — and **beats** the up-break at 40 days. Break up or break down, the stock drifts up the same. The direction the wedge claims to predict carries no information. |

> **In one sentence:** the falling wedge's bullish reputation survives as a *tape artefact*, not an edge — the up-break's 20-day excess clears a naive *t* = 2.19 but a same-tape placebo wins three times in ten, the breakout day itself loses money, the bearish *down*-break of the same shape drifts up just as hard, and on SPY there are only two wedges in 21 years — so it lands **Weak × Mirage**, with the "breaks up specifically" premise **Busted**.

## What we tested

A falling wedge is two **downward-sloping, converging** trendlines that folklore calls **bullish**:
selling exhausts as the range narrows, price breaks **above** the upper line, and it runs. Chart
figures are partly subjective, so we encode the closest **mechanical** definition — a run of
*descending* swing highs fit by a line, the intervening swing lows fit by a second line, **both
slopes negative**, the upper line **steeper** (highs fall faster → convergence), the band
**narrowing** ≥25% toward an apex, and a confirmed close above the extrapolated upper line — across
**SPY + 29 US large-caps** (21.4 years of auto-adjusted daily OHLC). We enter the close **one day
after** the breakout (no look-ahead) and measure the forward 5/10/20/40-day return as an **excess
over each name's own base rate**, arbitrated by a one-sample/HAC *t*, a **same-tape label-shuffle
placebo**, a 10-bps round trip, a **down-break symmetry** myth-check, and a deterministic synthetic
positive control. A survivors basket (named on the Signal axis) tilts the test *for* the figure —
and it still fails.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a falling wedge is, why "buy the upside break" sounds bullish, why the breakout day actually loses, and the killer fact that the *bearish* break drifts up too — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the mechanical detector, excess-over-base-rate by horizon, one-sample/HAC *t* vs a same-tape placebo, the strictness sweep, the down-break symmetry test, the SPY hook, and a synthetic faithful-engine / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`falling_wedge/`](falling_wedge/). Mechanical detector = descending swing-high line + swing-low line, both slopes negative, upper steeper, band narrows ≥25%, confirmed close above the rim. Basket is **survivors** — named on the Signal axis. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
