# Study 505 -- Left-Tail-Momentum

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## The claim

> The stocks with the worst recent crash keep underperforming -- does the left tail have momentum?

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the continuation real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The ABDG (2020) continuation **inverts** on a 48-name large-cap survivor basket: the crashed-tail leg *out-earns* the safe leg (+37.8% vs +12.5%/yr), so long-safe/short-crashed loses **-25.4%/yr**, one-sample *t* = **-3.44**. The signal is real and robust (placebo p = 0.000), but it is **reversal**, not the claimed momentum -- on the hypothesis as written, NONE. **Survivorship-biased basket (names trading in 2026): the worst-tail names that delisted are absent -- exactly the short-leg names ABDG needs.** |
| **Tradability** -- survives costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | A -96.1% max-drawdown short book, Sharpe **-1.03**, whose short leg is the decade's winners. Costs are tiny (turnover ~12%/mo) and do not move the wrong-sign spread. Nothing to trade in the stated direction. |
| **Does the left tail keep losing?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | On survivors a deep recent crash was a **buy**, not a sell -- the left tail rebounds. The continuation loses in 11 of 12 calendar years; the lone positive year (2022) is the one bear market. |

> **In one sentence:** the Atilgan-Bali-Demirtas-Gunaydin (2020) left-tail-momentum premium -- "the worst-crashed stocks keep crashing" -- runs in **reverse** on a tradable large-cap survivor basket, where recently-crashed names snap back (long-safe/short-crashed loses -25.4%/yr at *t* = -3.44), because the very names the short leg needs (the crashers that delist) are exactly the ones survivorship removes.

## What we tested

ABDG (2020): each month, sort the basket on trailing **left-tail risk** -- the 5%/1% VaR or the
single worst daily return over the trailing year. Go **long the safe tail** (least-negative VaR)
and **short the crashed tail** (most-negative VaR), enter the close **one trading day after** the
signal is public (one execution lag, no same-bar fill), hold the next calendar month, equal-weight
each leg. Panel: 48 large-cap S&P 500 survivors, yfinance daily adjusted-close 2014-2026
(135 monthly observations). Inference: one-sample *t* on the monthly spread (house bar |t| >= 2),
a 200-draw placebo label-shuffle null, costs = 5 bps/leg x turnover + 50 bps/yr short borrow, a
flavour/lookback sweep, and a 20-seed synthetic positive control proving the engine is faithful.
The basket is **survivorship-biased** -- we name it; the absent delisted crashers are precisely the
short-leg names, so the verdict is a lower bound on how badly the naive continuation trade does.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the under-reaction mechanism in plain language, the synthetic positive control, the real long-short that runs backwards, the survivor-rebound story, the honest verdict |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the VaR signal, seed-robust synthetic control, placebo null, flavour/lookback sweep, leg returns, equity curve and -96% drawdown, year-by-year breakdown |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`left_tail_momentum/`](left_tail_momentum/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
