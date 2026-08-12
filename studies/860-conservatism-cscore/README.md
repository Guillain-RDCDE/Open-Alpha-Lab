# Study 860 — Accounting Conservatism (C-Score) 🛡️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a high C-score predict forward returns? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | A monthly tercile long-short (long the most conservative / short the least) earns a **flat +4.0 bps/mo (+0.5%/yr gross), Newey-West *t* = +0.22** — \$1 → \$1.005 over 17 years — and it isn't even robustly right-signed: the pooled event drift runs **negative** at every horizon (−0.07% / −0.68% / −1.36%, placebo *p* up to 0.92), the staleness-120 variant is −9.4 bps (NW *t* = −0.43), the NOA-scaled signal a dead +0.02 *t*, and the sign flips across eras. No conservatism return premium on this tape. |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Zero gross edge to begin with; net of 20 bps + 100 bps borrow the book is **−1.5%/yr** (NW *t* = −0.70, Sharpe −0.16). Nothing to trade. |
| **Is the accounting actually conservative?** | ![Suggestive](https://img.shields.io/badge/Conservative%3F-Suggestive-8b949e?style=flat-square) | The Basu (1997) asymmetric-timeliness coefficient is **positive but faint** ($b_3$ = +0.025, *t* = +1.67, R² = 0.5%): earnings load a little more on bad news than good, so the panel is *mildly* conservative — a real-but-weak property that a coarse XBRL reserve proxy captures poorly and that buys no return edge. |

> **In one sentence:** a simplified Penman-Zhang / Basu conservatism score — estimated reserves
> over assets, ranked long-conservative / short-aggressive — is a **clean null on returns**
> (NW *t* = +0.22, if anything mildly *wrong*-signed), even though the panel does show a faint
> Basu bad-news asymmetry: **a real (weak) accounting phenomenon, no alpha in the stock.**

## What we tested

The Penman-Zhang / Basu idea, stated the way its believers state it: *"conservative accounting
recognises bad news fast and defers good, burying hidden reserves that understate net operating
assets — so firms carrying a big reserve cushion hold un-booked value the market under-prices;
buy the most conservative names."* We take it literally on **37 US non-financial large filers
that tag reserve/allowance accounts on EDGAR** (allowance for doubtful accounts + inventory
valuation reserve + deferred-tax valuation allowance, summed, ÷ assets), 2008→2026, ranked
**point-in-time on the 10-Q/10-K filing date** (zero look-ahead). We split the claim in two: a
pooled **Basu asymmetric-timeliness** regression (is the accounting even conservative?) and a
monthly tercile **long-short** held one month forward (the return claim), graded on an
autocorrelation-robust **Newey-West *t***, cross-checked by a pooled event drift + label-shuffle
placebo, an era split, a NOA-scaled variant, and a 12-seed synthetic control. Costs are one-way ×
NAV × turnover with the short leg paying borrow. **The reserve proxy is coarse and the panel thin
and uneven** — XBRL exposes only a subset of Penman-Zhang's reserves (no LIFO / R&D reserve), the
cross-section grows from ≈20 to ≈31 names, and only 42% of events carry the NOA denominator — and
we say so throughout.
**Dedup:** [229-beneish-m-score](../229-beneish-m-score/) detects the *opposite* posture
(income-inflating manipulation); [232-mohanram-g-score](../232-mohanram-g-score/) is a growth-firm
fundamental composite; [855-accrual-quality](../855-accrual-quality/) measures how well accruals
map to cash, not the level of hidden reserves; [52-smoke-screen](../52-smoke-screen/) is the
method-demo cousin of a good-story-no-reward null. None ranks on **reserve-intensity conservatism**
— this study does. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why hidden reserves *sound* like buried treasure, why the most-conservative stocks don't out-return the least, and what "real accounting, no alpha" means |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the calendar-time Newey-West long-short, the pooled event drift + placebo + monotonicity, the era split, the NOA-scaled variant, the Basu asymmetric-timeliness regression, the cost/borrow timer, and the 12-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`conservatism/`](conservatism/). EDGAR XBRL `companyconcept` (reserve/allowance
accounts, assets, cash, liabilities, debt, net income) + yfinance adjusted closes; a
**current-survivors** basket — survivorship named on the Signal axis. **Not investment advice** —
research & education. See [LICENSE](../../LICENSE).*
