# Study 856 — Book-Tax Differences 🧮

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a large book-tax gap predict lower future returns? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | A monthly tercile long-short (long low-BTD "clean" names, short high-BTD "aggressive" ones) is **right-signed** at **+20.1 bps/mo (+2.4%/yr gross)** — the Hanlon direction — but **never clears the bar**: Newey-West *t* = **+0.74** (one-sample +0.75); the strongest variant (the year-on-year *change* in the gap) tops out at NW *t* = **+1.79**. Worse, the pooled event drift leans the **other** way (low-BTD names slightly *under*-perform, long-short −0.2% to −2.5%, all insignificant, placebo *p* > 0.5) and both pre/post-2018 eras are insignificant. Genuinely **mixed and uncertified**. |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | It fails **before** any costs are charged (gross NW *t* < 1). Turnover is low (~0.05/mo, a slow annual balance-sheet signal) so 20 bps + 100 bps borrow only trims it to +1.2%/yr — but NW *t* = **+0.36**, Sharpe **0.09**. No paycheck for a spread you can't distinguish from luck. |
| **Does it mark less-persistent earnings?** | ![Not detected](https://img.shields.io/badge/Persistence%3F-Not_detected-8b949e?style=flat-square) | **No — flat.** Hanlon's actual headline is about earnings *persistence*, and on this basket it is a non-event: high-BTD earnings persist **+0.752** vs low-BTD **+0.750** (interaction *t* = **+0.03**). Even the mechanism that anchors the claim does not show up among large, clean filers. |

> **In one sentence:** a famous accounting red flag — book income far above the income the tax
> bill implies — that, taken to a clean panel of 39 large-cap survivors, delivers **neither** a
> certifiable return spread (NW *t* = +0.74, and the event drift is if anything mildly the *wrong*
> way) **nor** its signature earnings-persistence effect (high-BTD vs low-BTD persistence: +0.752
> vs +0.750): on blue chips the flag is a non-event.

## What we tested

Hanlon's (2005) book-tax-difference red flag, stated the way its believers state it: *"when book
income sits far above the taxable income the tax expense implies, earnings are propped up and won't
persist — so short the big-gap names and own the clean ones."* We take it literally on **42 large
US filers** (**39** survive the XBRL history filter), computing **BTD = (Pretax − IncomeTaxExpense /
statutory-rate) / Assets** from annual 10-Ks, with a **time-varying statutory rate** (35 % pre-2018,
21 % after the TCJA), ranked **point-in-time on the 10-K filing date** (zero look-ahead). We split
the claim: a monthly tercile **long-short** held one month forward (the return claim), graded on an
autocorrelation-robust **Newey-West *t***, cross-checked by a pooled event drift + label-shuffle
placebo, an era split, a change-in-gap variant, and a 12-seed synthetic control; **plus** a direct
test of the **earnings-persistence** mechanism (an interaction regression of next-year ROA on
this-year ROA across BTD terciles). Costs are one-way × NAV × turnover with the short leg paying
borrow. **Coverage is thin and skewed to the wrong tail** — these are large-cap survivors, where
book-tax gaps are smallest and Hanlon's effect weakest — and we say so throughout.
**Dedup:** [568-effective-tax-rate](../568-effective-tax-rate/) ranks on the tax *rate* level, not
the book-tax *gap*; [231-sloan-accruals](../231-sloan-accruals/) is the broader total-accruals
anomaly of which BTD is a tax-specific slice; [229-beneish-m-score](../229-beneish-m-score/) blends
eight manipulation signals, where BTD is tested here alone. None ranks on the statutory-grossed-up
**book-minus-tax income difference** — this study does. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a "book-tax difference" even is, why book income routinely tops the taxable figure, and why on blue chips the red flag predicts neither returns nor fragile earnings |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the calendar-time Newey-West long-short, the pooled event drift + placebo + monotonicity, the era split, the change variant, the earnings-persistence interaction, the cost/borrow timer, and the 12-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`book_tax_diff/`](book_tax_diff/). EDGAR XBRL `companyconcept` (pretax income, income-tax
expense, assets) + yfinance adjusted closes; a **current-survivors** basket — survivorship named on
the Signal axis. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
