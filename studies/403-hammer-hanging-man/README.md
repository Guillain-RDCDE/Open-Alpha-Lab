# Study 403 — Hammer & Hanging Man 🔨

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a long lower wick mark a floor? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Across **11,816** hammer-shaped bars on **26** large-caps + SPY (1962→2026), the bullish **hammer**'s best forward edge over its own base rate is **+0.05% at 3 days, HAC *t* = +0.92** (placebo *p* = 0.13) — and *negative* at 1 day. Nowhere near **t ≥ 2**. The only "real" stat is the **hanging man**'s faint *under*-performance (3-day *t* = −2.63) — wrong sign for a floor, untradeable. The harness detects a *planted* floor at *t* = 13.5, so the flat reading is genuine. (Survivorship tilts *toward* a bounce → conservative.) |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Net of a 5-bps round trip the edge is **negative at every horizon and on every side** (−0.05% to −0.20%). There is no positive gross edge to defend. |
| **Reliable reversal?** | ![Busted](https://img.shields.io/badge/Reliable_reversal%3F-Busted-8b949e?style=flat-square) | The same shape marks neither a floor (hammer) nor a shortable top (hanging man). The only way to nudge *t* toward 2 is to **snoop the trend window** (lookback 5 → *t* = 1.94; lookback 20 → 0), and a *longer, "purer" wick* — the stronger signal the folklore prizes — makes it **worse** (*t* ≈ 0.05). |

> **In one sentence:** the hammer's long lower wick is real *intraday* information, but as a *forward* signal it's a mirage — the bullish floor never clears *t* = 2 (best +0.05% at 3 days, *t* = 0.92), it's negative after costs everywhere, the hanging man only weakly *under*-performs (the wrong direction for a tradable edge), and any apparent edge is a snooped-filter artefact — while the same harness nails a *planted* floor at *t* = 13.5.

## What we tested

We detect the hammer / hanging-man candle by precise OHLC rules — small body at the top of the
range, lower wick ≥ 2× the body, little upper shadow — across a fixed **26-name** liquid
large-cap basket + SPY (yfinance **un-adjusted, price-only** daily bars, cache-first). For each
occurrence we measure the **forward 1/3/5/10-day return**, entered at the **next close** (one
documented execution lag), against that name's **unconditional base rate**, then split the
*same shape* by **prior trend** into the bullish hammer (after a downtrend) and the bearish
hanging man (after an uptrend). Signal is a HAC one-sample *t* on the conditional edge plus a
label-shuffle placebo; Tradability charges a 5-bps round trip; the myth-check asks whether any
trend/wick filter rescues it. A deterministic synthetic control with a *planted* bounce proves
the harness can detect a floor when one exists. Survivorship (surviving-names basket, biased
*toward* a bounce) is named on the Signal axis.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a hammer *is*, why a long lower wick *feels* like a floor, what "beating the base rate" means, and why the floor isn't there — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the OHLC detector, forward 1/3/5/10-day edge vs base rate, HAC *t* + label-shuffle placebo, the trend split (hammer vs hanging man), the filter-snoop myth-check, costs, and a synthetic planted-floor power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`hammer_hanging_man/`](hammer_hanging_man/). Un-adjusted OHLC (the candle shape needs printed levels) → forward returns are **price-only**. Basket is **survivors** — named on the Signal axis. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
