# Study 684 — Inverted Hammer 🔻

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a long upper wick after a downtrend mark a floor? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Across **4,895** inverted-hammer occurrences (small body, long upper wick, after a downtrend) on **26** large-caps + SPY (1962→2026), the best forward edge over the base rate is **+0.095% at 3 days, HAC *t* = +1.79** — under the **t ≥ 2** bar. **Bonferroni-corrected** across the 1/3/5/10-day family, no horizon's adjusted *p* clears 0.05 (best 0.082). Only **2/26** names individually cross \|*t*\| > 2 — chance level. The harness detects a *planted* floor at *t* = 12.8, so the flat reading is genuine. (Survivorship tilts *toward* a bounce → conservative.) |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Net of a 5-bps round trip the edge is **negative or a rounding error at every horizon** (−0.06% to +0.04%); break-even sits under ~2 bps one-way. There is no robust gross edge to defend. |
| **Deeper washout rescues the floor?** | ![Busted](https://img.shields.io/badge/Deeper_washout_rescues_it%3F-Busted-8b949e?style=flat-square) | A shorter trend lookback or a ≥5% prior-decline filter can each nudge *t* past 2 on a thinned sample — but doubling the washout threshold **collapses** it back to *t* = 1.02, and a *longer, purer* wick (the "textbook" inverted hammer) makes it **worse** (*t* ≈ 0.9). Non-monotonic, filter-snooped, not a dose-response. |

> **In one sentence:** the inverted hammer's long upper wick is real *intraday* information, but as a *forward* bullish signal it's a mirage — the trend-conditional floor never clears *t* = 2 (best +0.095% at 3 days, *t* = 1.79), survives no Bonferroni correction across the four horizons tested, turns negative net of ordinary costs, and the only ways to "rescue" it with a filter contradict each other — while the same harness nails a *planted* floor at *t* = 12.8.

## What we tested

We detect the inverted-hammer candle by precise OHLC rules — small body at the bottom of the
range, upper wick ≥ 2× the body, little lower shadow — across a fixed **26-name** liquid
large-cap basket + SPY (yfinance **un-adjusted, price-only** daily bars, cache-first, the
same panel as the sibling candlestick studies). For each occurrence after a **downtrend**
(the bullish claim) we measure the **forward 1/3/5/10-day return**, entered at the **next
close** (one documented execution lag), against that name's own **unconditional base rate**,
and test it with a HAC one-sample *t*, a label-shuffle placebo, and a **Bonferroni**
correction across the four simultaneous horizons. Tradability charges a 5-bps round trip; the
myth-check axis asks whether a trend-window or washout-depth filter rescues it. A
deterministic synthetic control with a *planted* bounce proves the harness can detect a floor
when one exists, and we cross-check the same geometry traded long after an **uptrend** — the
"wrong side" of the trend split (the direct look-alike of sibling study
[404-shooting-star](../404-shooting-star/)) — to see whether the trend split actually
discriminates a floor from generic post-wick drift. Survivorship (surviving-names basket,
biased *toward* a bounce for this bullish claim) is named on the Signal axis.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what an inverted hammer *is*, why the long upper wick *feels* like sellers giving up, what "beating the base rate" means, and why the floor isn't there — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the OHLC detector, forward 1/3/5/10-day edge vs base rate, HAC *t* + label-shuffle placebo + Bonferroni, the trend-split contrast, the filter-snoop myth-check, costs, and a synthetic planted-floor power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`inverted_hammer/`](inverted_hammer/). Un-adjusted OHLC (the candle shape needs printed levels) → forward returns are **price-only**. Basket is **survivors** — named on the Signal axis. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
