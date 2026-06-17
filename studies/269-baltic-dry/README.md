# Study 269 -- Baltic-Dry

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Full-sample regression of next-month S&P return on 3M BDI momentum clears the bar (HAC *t* = **+2.82**), but it is almost entirely the 2008-09 co-crash: drop the crisis and *t* falls to **+1.35**. The slope is *negative* in 1985-1999 and the sign hit-rate (54.6%) is **below** the base rate (64.7%). Not robust. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The long-when-BDI-rising timing overlay **loses 3.3%/yr to buy-and-hold** (*t* = -1.90) with a lower Sharpe; its only edge is a marginally shallower drawdown. No exploitable signal. |
| **Curated-series bias** | ![Named](https://img.shields.io/badge/Curated--series-8b949e?style=flat-square) | The BDI has no clean free feed; the real tape is a hindsight-curated monthly reconstruction, so the in-sample fit is flattered. Real numbers are indicative. |

> **In one sentence:** the freight index "leads stocks" only in the sense that it crashed *at the same time* as equities in 2008 -- strip that one episode and the signal is gone, the sign-predictability is worse than a coin that always says "up", and the tradable overlay gives back a third of the equity premium, making this a *weak/mirage* piece of macro folklore.

## The claim

> *Does the Baltic Dry shipping index lead the stock market?*

## What we tested

At each month-end we form a 3-month BDI momentum signal (the log change of the
freight index) and ask it to predict the *next* month's S&P 500 return -- one
full month of execution lag. We run (a) a predictive regression with a HAC
*t*-stat on the slope as the inference bar, (b) a tradable long-when-BDI-rising
timing overlay (long the S&P next month if freight is rising, else cash) charged
one-way costs and raced gross/net against buy-and-hold, (c) a sub-period
breakdown and an ex-2008/09 robustness cut, and (d) a sign hit-rate versus the
unconditional base rate. A deterministic synthetic positive control (a planted
``beta`` lead-lag) confirms the regression recovers a real relationship when one
exists. The BDI is a curated monthly reconstruction (no clean free feed), named
on the Signal axis; price-only S&P returns on both legs.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the freight-as-demand-thermometer story, why the headline *t* is a 2008 mirage, and the timing overlay losing to buy-and-hold in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC regression, ex-crisis and sub-period slopes, timing gross/net vs buy-and-hold, sign hit-rate, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`baltic_dry/`](baltic_dry/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
