# Study 263 -- Insider-Buying

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Predictive HAC *t* = **+0.49** (next-month return on standardised buy/sell ratio); the high-minus-low tercile is the **wrong sign** (-3%/yr); no sub-period clears *t* = 2. The insider "buy-the-dip" reflex is **contemporaneous** (ratio vs same-month return = -0.08), not predictive. The buy/sell ratio is a **curated proxy**, which caps the axis at WEAK regardless -- the realised stats put it at NONE. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | A long/flat overlay returns **+3.4%/yr** net vs **+10.2%/yr** for buy-and-hold; it sits in cash through more than half the sample and misses the upside. No vehicle, no edge. |
| **Busted?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The folklore conflates a real coincident reflex (insiders buy on weakness) with a market-timing rule. The one-month-ahead aggregate forecast is a coin flip. |

> **In one sentence:** aggregate insider buying spikes near bottoms -- but by the time you can see the spike, the down move has already happened, and the next month is a coin flip, so the timing overlay loses badly to buy-and-hold.

## The claim

> *Does aggregate insider buying (Form 4) forecast returns?*

## What we tested

We pair a **curated** monthly aggregate insider buy/sell ratio (open-market Form-4
purchases / sales, 2003-2025, anchored to documented buying spikes in 2009 and 2020
and selling tops in 2007 and 2021) with S&P 500 (^GSPC) **price-only** monthly
returns. We test whether a HIGH ratio forecasts the NEXT month via (a) a Newey-West
HAC predictive regression, (b) a low/mid/high tercile sort, and (c) a long/flat
timing overlay (long when the ratio is above its trailing 24-month median, else flat;
cost x NAV on every switch) raced against buy-and-hold. A deterministic synthetic
positive control confirms the regression fires (t > 3) when a contrarian effect is
planted. **The buy/sell ratio is a curated proxy, not a live Form-4 feed -- the
Signal axis is capped at WEAK for that reason, and the realised t-stat puts it at
NONE.**

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the contrarian story, why insiders buy at bottoms, the coincident-vs-predictive distinction, the overlay that loses to buy-and-hold |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC predictive regression, tercile sort, sub-period decay, timing overlay net of costs vs buy-and-hold, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`insider_buying/`](insider_buying/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
