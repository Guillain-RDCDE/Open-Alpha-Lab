# Study 532 -- Firm-Age-Anomaly

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## The claim

> Do mature, long-established firms quietly out-earn young, recently-listed ones -- the "new-list" effect?

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the firm-age premium statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The claimed old-minus-young premium **does not appear** -- the real-tape spread is significantly *negative* (**-18.2%/yr**, HAC *t* = **-4.66**; clean post-2012 *t* = **-2.07**; label-shuffle placebo **p = 0.010**). Young firms *beat* old ones. The sign is reversed; literature support cannot rescue it. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | As specified (long old, short young) it bleeds: Sharpe **-0.775** (net **-0.819** after 10 bps + 100 bps borrow), max drawdown **-100%** (effective total loss by compounding), hit rate 38.8%. Turnover is ~0.02, so costs are not the culprit -- the *direction* is. |
| **Is the survivor basket the whole story?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | Yes. The basket holds only young firms that *survived* to 2026 -- the winners (META, the NVDA-era IPOs). Their failed peers, which make the new-list effect work, are gone from the tape. The reversed sign is survivorship made visible, not an edge. |

> **In one sentence:** the firm-age / new-list premium (mature firms beat young ones) is well-documented in the literature, but on a survivor basket that mixes 1985-floor old-economy names with 2012-2021 IPOs the spread flips hard negative -- young survivors win by 18%/yr at *t* = -4.66 -- which is not a tradable edge but a textbook portrait of survivorship bias inverting an anomaly.

## What we tested

Fama-French (2004) and Jiang-Lee-Zhang (2005) document that young, recently-listed firms
underperform mature firms. We proxy firm age from each name's **first available price date**,
sort a 39-name large-cap basket spanning four IPO decades into age terciles each month, go long
the oldest tercile and short the youngest (old-minus-young), enter at the close **one day after**
the ranking is public, and hold equal-weight to the next rebalance. We charge 10 bps one-way x
turnover plus 100 bps/yr borrow on the short leg, run a firm-age **label-shuffle placebo**
(p = 0.010), and prove the engine with a deterministic synthetic positive control whose mean and
HAC *t* rise monotonically with a planted premium (seed-robust over 25 seeds). yfinance daily
adjusted-close, 1985-2026, 490 monthly observations. The basket is **survivorship-biased** -- we
name it, and it is the entire reason the sign reverses.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the new-list claim in plain language, the first-price-date age proxy, the synthetic control, the real spread, and why survivorship flips the sign |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | age-dispersion over time, the old-minus-young long-short, HAC *t*, label-shuffle placebo, cost/turnover, post-2012 subsample, equity curve and drawdown |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`firm_age_anomaly/`](firm_age_anomaly/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
