# Study 95 - Holiday-Cheer 🦃

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) - see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** - is the effect statistically real? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | **Real on the long tape, faded since.** Over 1950+ (^GSPC price-only) the pre-holiday day earned **6.1x** the average day (18.9 vs 3.1 bps), HAC *t* = **+5.05**, win-rate **64%** vs 53%. But that is carried by the early sample: pre-1990 *t* = **+7.25**, post-1990 *t* = **+1.96**, and on modern total-return SPY alone *t* = **+0.98**. |
| **Tradability** - does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The day is wonderful (per-invested-day Sharpe **3.57**) but there are only **~9 a year**. A pre-holiday-only book compounds to **1.5%/yr vs 8.3%** for buy-and-hold (^GSPC PO) - it captures far too few days to matter, and the modern day no longer reliably pays. |
| **Faded since 1990?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | Split at Ariel's 1990 publication: the gap fell from **25.1 bps** to **5.0 bps**. Block-bootstrap of the *difference*: decay **20.1 bps**, **P(decay>0) = 0.999**, 95% CI [6.8, 32.8] bps. |

> **In one sentence:** the pre-holiday effect was **real and enormous** before it was published - 6x the normal day across 1950+ - but it **faded sharply after 1990**, and with only ~9 pre-holiday days a year a buy-the-day-before book earns a fraction of just holding the index, so the famous edge is a **mirage** to trade today.

## What we tested

The oldest calendar anomaly in the book, stated at full strength: *"stocks reliably drift up the trading day **before** every market holiday - several times the normal daily average - so buy the day before every holiday"* (Ariel 1990, *High Stock Returns before Holidays*; Lakonishok & Smidt 1988). We take it literally. The clever part is the calendar: we **derive the US market holidays offline from the trading index itself** - any weekday absent from the trading dates (and not a weekend) is a day the exchange was shut, and the prior trading day is "pre-holiday." No external calendar, no drift. We then measure pre-holiday vs every other day (HAC *t*, Wilson win-rate intervals) on **SPY total-return** (1993+, the fair tape) and **^GSPC price-only** (1950+, the long sample), run a **pre-holiday-only** book against buy-and-hold, and split **pre/post-1990** with a block-bootstrap test of the decay. A deterministic synthetic tape with a *planted* pre-holiday bump (must be detected) and a flat i.i.d. tape (must not) is the harness's positive/negative control.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the rule, the 6x pre-holiday bar chart, why the day is great but rare, the cumulative pre-holiday-only line that can't keep up, the fade since 1990 |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC *t* on the gap, Wilson win-rate intervals, the pre/post-1990 block-bootstrap decay test, capacity arithmetic, price-only vs total-return |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`holiday_cheer/`](holiday_cheer/). **Not investment advice** - research & education. See [LICENSE](../../LICENSE).*
