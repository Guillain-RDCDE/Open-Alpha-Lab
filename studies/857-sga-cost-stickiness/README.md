# Study 857 — SG&A Cost Stickiness 📎

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do sticky-cost firms under-earn? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | A monthly tercile long-short (long the leanest cost-discipline, short the stickiest) earns **−28.6 bps/mo (−3.4%/yr gross)** — the **wrong sign** vs the claim and insignificant: Newey-West *t* = **−0.45** (one-sample −0.57), and it isn't even sign-stable (a staleness-120 variant flips to **+0.66**). The pooled event drift is **flat and sign-inconsistent** (long-short *t* ∈ [−0.32, +0.57], placebo *p* ≈ 0.25-0.62), and the two eras carry **opposite-signed** insignificant means (+14 vs −78 bps). No robust effect in either direction. |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | It's the wrong sign and insignificant **before** costs; net of 20 bps + 100 bps borrow it is −4.8%/yr, NW *t* = **−0.63**, Sharpe **−0.27**. Nothing to trade. |
| **Is SG&A actually sticky?** | ![Confirmed](https://img.shields.io/badge/Sticky_costs%3F-Confirmed-8b949e?style=flat-square) | **Yes — the ABJ effect replicates.** Pooled, SG&A rises **+0.65%** per 1% sales gain but falls only **+0.55%** per 1% sales loss (mean β₂ = **−0.097**). The phenomenon is real; it just doesn't sort returns *or* forecast weaker future ROA (tercile spread −0.14 pp, correlation +0.01). |

> **In one sentence:** cost stickiness is a **genuine accounting regularity** — SG&A really
> does cling on the way down — but the firm-level stickiness estimate, ranked point-in-time,
> **neither predicts stock returns (NW *t* = −0.45, wrong sign) nor weaker future profitability
> (spread −0.14 pp)**: a real behaviour of costs, no alpha in the stock.

## What we tested

Anderson, Banker & Janakiraman (2003): SG&A costs are *sticky* — they rise more when sales rise
than they fall when sales fall, because managers delay cutting discretionary overhead into a
downturn. The trading pitch: a firm whose SG&A stays sticky into a sales decline has **weaker
operating discipline**, so **sticky-cost firms should under-earn**. We take it literally on **33
large US filers that report `SellingGeneralAndAdministrativeExpense` on EDGAR** (25 of them ever
identify the effect), estimating each firm's ABJ β₂ **point-in-time on an expanding window of
only-public quarterly filings** (zero look-ahead), 2015→2026. We rank on `disc = −stickiness`
(leanest minus stickiest), hold one month forward with one execution lag, and grade the monthly
long-short on an autocorrelation-robust **Newey-West *t***, cross-checked by a pooled event drift
+ label-shuffle placebo, an era split, a future-ROA regression (does stickiness forecast weaker
earnings?), and a 10-seed synthetic control. Costs are one-way × NAV × turnover with the short
leg paying borrow. **Coverage is thin, cyclical-tilted and noisy** — only firms that actually
*decline* identify β₂ — and we say so throughout.
**Dedup:** [524-operating-leverage](../524-operating-leverage/) is the *magnitude* of the
cost-to-sales elasticity (fixed vs variable), not its up-vs-down **asymmetry**;
[200-roe-quality](../200-roe-quality/) and [122-gross-profitability](../122-gross-profitability/)
rank on profitability *levels*; [749-layoff-drift](../749-layoff-drift/) trades an announced
cost-cut *event*. None estimates the **asymmetric SG&A-to-sales response** itself — this study
does. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "sticky costs" is a real, intuitive accounting fact, why that doesn't make it a stock signal, and what "real behaviour, no alpha" means |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the point-in-time ABJ estimator, the calendar-time Newey-West long-short, the pooled event drift + placebo + monotonicity, the era split, the future-ROA regression, the cost/borrow timer, and the 10-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`sga_stickiness/`](sga_stickiness/). EDGAR XBRL `companyconcept` (SG&A, revenue, net
income, assets) + yfinance adjusted closes; a **current-survivors** basket — survivorship named
on the Signal axis. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
