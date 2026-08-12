# Study 859 — Return-on-Invested-Capital Premium 🏭

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does ROIC sort future returns? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | A monthly tercile long-short on ROIC (long high / short low) earns **+13.5 bps/mo (+1.6%/yr gross)** — a near-perfect null: Newey-West *t* = **+0.40** (one-sample +0.48). The best specs (staleness-120 +1.12, ROIC-*change* +1.14) don't approach 2; the pooled event drift is **flat and turns negative at 2 quarters** (−1.68%, non-monotone terciles, placebo *p* ≈ 0.4-0.8); and the era split **flips sign** (−5.6 bps pre-2018, +32.2 bps after). Right-signed by a hair, real by no measure. |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | It fails **before** any costs (gross NW *t* < 0.5). Turnover is tiny (~0.06/mo), so 20 bps + 100 bps borrow barely moves it — to +0.33%/yr, NW *t* = **+0.08**, Sharpe **0.02**. Nothing to trade. |
| **Does ROIC add anything over ROE / gross profitability?** | ![No](https://img.shields.io/badge/Adds_over_ROE%2FGP%3F-No-8b949e?style=flat-square) | **Immaterial.** On this survivor mega-cap panel **none** of the quality signals certify; ROIC is only the *least bad* (the sole one that stays right-signed — plain ROE **−1.36** and gross profitability **−1.59** come out *negative*), and it ranks **+0.75** with ROE. Largely the same bet, leverage scrubbed off, no certifiable edge. |

> **In one sentence:** ROIC — unlevered NOPAT over invested capital, the "quality compounder"
> number — earns a **statistically-zero** long-short on 32 large US survivors (NW *t* = +0.40,
> flat and even sign-flipping across horizons and eras), and it **adds nothing** over plain ROE or
> gross profitability, which are actually *negative* here: **the quality premium is absent among
> mega-cap survivors, and stripping leverage out of the ratio doesn't resurrect it.**

## What we tested

The value-investor / quality-factor staple: *"ROE is polluted by leverage — divide **unlevered**
operating profit (NOPAT) by **all** the capital deployed (debt + equity − cash) and you get ROIC,
the clean gauge of a business's return on capital; buy the high-ROIC compounders, short the
low-ROIC capital-destroyers."* We take it literally on **44 large US non-financial filers** (32
yield a valid ROIC) from EDGAR (`OperatingIncomeLoss` → TTM NOPAT, `StockholdersEquity`,
`LongTermDebtNoncurrent`, `CashAndCashEquivalentsAtCarryingValue`), 2009→2026, ranked
**point-in-time on the 10-Q/10-K filing date** (zero look-ahead). The primary is a monthly tercile
**long-short** held one month forward, graded on an autocorrelation-robust **Newey-West *t***,
cross-checked by a pooled event drift + label-shuffle placebo, an era split, a ROIC-change variant,
and a 12-seed synthetic control. We also run the **same** long-short on plain ROE and gross
profitability to answer the headline question — *does the unlevered refinement add anything?*
(A flat 21% tax rate is a common scalar, so it leaves the ranking untouched — proved in the tests.)
Costs are one-way × NAV × turnover with the short leg paying borrow. **Coverage is thin and
uneven** (avg cross-section 16.5; XBRL starts ~2009, some names re-registered under new CIKs) and
the basket is **current survivors** — survivorship named on the Signal axis. **Dedup:**
[200-roe-quality](../../200-roe-quality/) is the *levered* return-on-equity; [122-gross-profitability](../../122-gross-profitability/)
is GrossProfit/Assets with no capital-structure or tax treatment; [242-quality-minus-junk](../../242-quality-minus-junk/)
is the AQR *composite*; [521-cash-based-operating-profitability](../../521-cash-based-operating-profitability/)
strips accruals from the numerator and scales by assets. None ranks on **NOPAT ÷ invested capital**
— this study does. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why ROIC is the "quality compounder" number, why it doesn't sort mega-cap survivors, and why stripping leverage out of ROE changes nothing here |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the calendar-time Newey-West long-short, the pooled event drift + placebo + monotonicity, the era split, the ROIC-vs-ROE-vs-GP contrast, the cost/borrow timer, and the 12-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`roic_premium/`](roic_premium/). EDGAR XBRL `companyconcept` (operating income, equity,
long-term debt, cash) + yfinance adjusted closes; a **current-survivors** mega-cap basket —
survivorship named on the Signal axis. **Not investment advice** — research & education. See
[LICENSE](../../LICENSE).*
