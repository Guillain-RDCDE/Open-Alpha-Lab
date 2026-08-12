# Study 855 — Accrual Quality (Dechow-Dichev) 📊

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does accrual quality predict forward returns? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | A monthly tercile long-short (buy high-quality / short low-quality, where *quality = −DD residual vol*) earns an economically-zero and **faintly wrong-signed −3.6 bps/mo (−0.44%/yr gross)** — Newey-West *t* = **−0.18** (one-sample −0.19). The pooled event drift is **negative and non-monotone at every horizon** (long-short *t* −0.58 → −0.91, placebo *p* 0.73-0.84), the working-capital variant agrees (NW *t* = −0.47), and neither pre/post-2018 era certifies. The stated "buy quality, it's underpriced" claim fails outright; the sliver of tilt that exists leans the *opposite* (Francis-2005 risk-premium) way. |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | It was wrong-signed and insignificant **before** costs; net of 20 bps + 100 bps borrow it is −1.8%/yr, NW *t* = **−0.73**, Sharpe **−0.20**. You would pay to hold it. |
| **Does it flag earnings quality?** | ![Confirmed](https://img.shields.io/badge/Earnings_persistence%3F-Confirmed-8b949e?style=flat-square) | **Yes — decisively.** Good-quality (low residual-vol) names have far more persistent earnings (next-quarter ROA slope **+0.87 vs +0.27**) and ~**5× lower** earnings volatility (0.53% vs 2.44% of assets). The Dechow-Dichev construct measures exactly what it claims — the market just already prices the fundamental. |

> **In one sentence:** the Dechow-Dichev residual-volatility measure is a *real* gauge of
> earnings quality — good-quality earnings persist (+0.87 vs +0.27) and are ~5× less volatile —
> but ranking stocks on it earns **nothing**: the long-high-quality/short-low-quality spread is
> −3.6 bps/mo (NW *t* = −0.18), if anything the wrong sign, with a flat non-monotone event drift.
> **Real accounting information about the firm, no alpha in the stock.**

## What we tested

The Dechow-Dichev accrual-quality idea, stated the way its trading believers state it: *"earnings
whose accruals don't map into cash flows are low-quality and discounted — so buy the high-quality
names and short the low-quality ones."* We approximate a name's accrual quality as the **standard
deviation of the residual** from regressing its `(NI−CFO)/avg-assets` on lag/current/lead
`CFO/avg-assets` over a rolling **12-quarter** window (a *high* residual vol = poor quality), on
**45 deep-history US non-financial filers** (42 clear the data bar) whose EDGAR fundamentals
(`NetIncomeLoss`, `NetCashProvidedByUsedInOperatingActivities`, `Assets`, receivables, inventory)
we pull and rank **point-in-time on the 10-Q/10-K filing date** — the DD lead-CFO term never
peeks past the filing (zero look-ahead). We split the claim in two: an earnings-persistence check
(good vs poor quality) and a monthly tercile **long-short** held one month forward (the return
claim), graded on an autocorrelation-robust **Newey-West *t***, cross-checked by a pooled event
drift + label-shuffle placebo, an era split, a working-capital-accrual variant, and a 12-seed
synthetic control. Costs are one-way × NAV × turnover with the short leg paying borrow.
**Coverage is thin and uneven** — the rolling window needs ~4 years before a first signal and the
cross-section grows from ≈26 (2012) to ≈40 (2024+) — and we say so throughout.
**Dedup:** [231-sloan-accruals](../231-sloan-accruals/) ranks on the accrual *level/sign*;
[522-percent-operating-accruals](../522-percent-operating-accruals/) on accruals scaled by
earnings; [539-cash-flow-volatility](../539-cash-flow-volatility/) on raw cash-flow vol;
[52-smoke-screen](../52-smoke-screen/) on discretionary-accrual manipulation. None ranks on the
**Dechow-Dichev residual volatility** (the reliability of the accrual-to-cash mapping) — this
study does. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "accrual quality" really does separate steady earners from noisy ones, why the stocks *don't* care, and what "real information, no alpha" means |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the calendar-time Newey-West long-short, the pooled event drift + placebo + monotonicity, the era split, the WC-accrual variant, the earnings-persistence mechanism, the cost/borrow timer, and the 12-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`accrual_quality/`](accrual_quality/). EDGAR XBRL `companyconcept` (net income, operating
cash flow, assets, receivables, inventory — quarterly flows reconstructed from the YTD cumulative
chain) + yfinance adjusted closes; a **current-survivors** basket — survivorship named on the
Signal axis. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
