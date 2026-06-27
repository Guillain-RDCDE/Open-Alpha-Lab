# Study 506 -- Industry-Momentum

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## The claim

> Moskowitz & Grinblatt (1999) argued the famous stock-momentum effect is *mostly an industry
> effect* -- winner industries keep winning. Does sorting sectors beat sorting stocks on the
> modern ETF tape?

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- do winner *industries* keep winning? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The 11-sector-ETF 12-1 long-short earns **+0.90%/yr** gross, HAC *t* = **+0.315**, label-shuffle placebo *p* = **0.348** -- indistinguishable from a coin, and no `top_k` cut (1-4) clears *t* = 2. **Survivorship is named here on the signal axis**: the *sector ETFs are survivorship-free*, but the single-name comparison basket is names *still trading in 2026*, so its loser short is an upper bound (opt-in guard: feed a delisting-complete panel to lift it -- we cannot from yfinance). |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Best industry net return **+0.14%/yr** (after 5bps/leg + 50bps borrow), with a **-45%** drawdown and **22%/mo** turnover; both legs earn ~the market (winner sectors +11.1%, loser +10.2%, SPY +11.7%). Nothing to trade. |
| **Industry beats single-name?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | Moskowitz-Grinblatt's signature claim is that the *industry* sort dominates and subsumes single-name momentum. On this tape the single-name book is **larger** (**+2.46%/yr**, *t* = +0.566) than the industry book (**+0.90%/yr**) -- and both are noise. The headline does not reproduce. |

> **In one sentence:** the canonical industry-momentum trade -- buy past-winner sectors, short past-loser sectors -- earns a coin-flip **+0.90%/yr** at HAC *t* = 0.315 (placebo *p* = 0.35) on 256 months of the 11 SPDR sector ETFs, never beats the single-name sort it was supposed to dominate, and is a survivorship-free Mirage net of costs.

## What we tested

Moskowitz & Grinblatt (1999): each month, rank assets by their trailing 12-month return *skipping
the most recent month* (the classic "12-1", which dodges short-term reversal), go long the top
winners and short the bottom losers, equal-weight, dollar-neutral, hold one month. We prove the
apparatus on a deterministic synthetic sector panel with a *baked-in* persistent industry-momentum
component (and a no-momentum null that earns nothing, seed-averaged over 20 seeds), then **race**
two real books: the **11 SPDR sector ETFs** (industry momentum, survivorship-free) against a
**40-name large-cap survivor basket** (single-name momentum), yfinance daily adjusted-close,
2004-2026, 256 holding months. One execution lag (form on the month-end close, hold the next month
-- no same-bar fill, no look-ahead); costs of 5bps/leg/rebalance plus a 50bps/yr short borrow; a
500-draw label-shuffle placebo null; survivorship named on the **signal** axis as an upper-bound
caveat on the single-name leg.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | industry momentum in plain language, why a long bull tape makes every sector a winner, the synthetic control, the industry-vs-stock race, and the honest verdict |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the 12-1 signal, the long-short construction, HAC inference, the label-shuffle placebo, the industry-vs-single-name race, the top_k sweep, year-by-year crashes, gross-vs-net |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`industry_momentum/`](industry_momentum/). Sector ETFs are survivorship-free; the single-name basket is **survivors** -- named on the Signal axis. **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
