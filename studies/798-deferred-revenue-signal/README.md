# Study 798 — Deferred-Revenue Signal 📥

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does deferred-revenue growth predict forward returns? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | A monthly tercile long-short (buy the fastest-growing deferred revenue, short the slowest) earns a right-signed **+56.2 bps/mo (+6.7%/yr gross)** — but it **never clears the bar**: Newey-West *t* = **+0.94** (one-sample +1.04), the asset-scaled variant tops out at NW *t* = **+1.63**, the pooled event drift is **flat and non-monotone** (long-short *t* ≈ +0.2, label-shuffle placebo *p* ≈ 0.4), and both pre/post-2019 eras are right-signed but insignificant. Literature-supported (Sloan 1996; Prakash-Sinha 2013), this tape can't certify it. |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | It fails **before** any costs are charged (gross NW *t* < 1). Turnover is low (~0.10/mo, a slow balance-sheet signal) so 20 bps + 100 bps borrow only trims it to +5.2%/yr — but NW *t* = **+0.73**, Sharpe **0.20**. No paycheck for a spread you can't distinguish from luck. |
| **Does it actually lead future sales?** | ![Confirmed](https://img.shields.io/badge/Leads_sales%3F-Confirmed-8b949e?style=flat-square) | **Yes — decisively.** This quarter's deferred-revenue growth predicts *next* quarter's revenue growth with correlation **+0.67**; the fastest-deferred-growth third goes on to grow sales **+32 percentage points** faster than the slowest. The accounting lead is real and mechanical — the market just already prices it. |

> **In one sentence:** deferred revenue is a genuine crystal ball for a SaaS firm's *sales*
> — top-minus-bottom third is **+32 pp** of future revenue growth, correlation +0.67 — but
> that lead is a public balance-sheet number the market prices on filing day, so ranking
> stocks on it earns a right-signed-but-**uncertifiable** long-short (NW *t* = +0.94, flat
> event drift): **real information about the company, no alpha in the stock.**

## What we tested

The SaaS-investor staple, stated the way its believers state it: *"deferred revenue is bookings
that haven't hit the income statement yet — if it's swelling, revenue is coming, so buy the
names growing it fastest."* We take it literally on **40 subscription/SaaS-type US names that
report a current deferred-revenue / contract-liability balance on EDGAR** (`DeferredRevenueCurrent`
→ the ASC-606 successor `ContractWithCustomerLiabilityCurrent`), 2009→2026, ranked **point-in-time
on the 10-Q/10-K filing date** (zero look-ahead). We split the claim in two: a pooled regression
of *next-quarter revenue growth* on the signal (the accounting lead) and a monthly tercile
**long-short** held one month forward (the return claim), graded on an autocorrelation-robust
**Newey-West *t***, cross-checked by a pooled event drift + label-shuffle placebo, an era split,
an asset-scaled variant, and a 12-seed synthetic control. Costs are one-way × NAV × turnover with
the short leg paying borrow. **Coverage is thin and uneven** — the cross-section grows from ≈9
names (2010) to ≈27 (mid-2020s) as the SaaS cohort IPO's — and we say so throughout.
**Dedup:** [199-sales-growth](../199-sales-growth/) ranks on the *recognised* top-line number;
[534-revenue-surprise-drift](../534-revenue-surprise-drift/) tests the drift after a *revenue
surprise*; [799-order-backlog-drift](../799-order-backlog-drift/) uses *order backlog / RPO*, one
step earlier in the cash cycle. None ranks on the **deferred-revenue balance** itself — this
study does. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why deferred revenue really is next year's revenue in plain sight, why the stocks *don't* follow, and what "real information, no alpha" means |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the calendar-time Newey-West long-short, the pooled event drift + placebo + monotonicity, the era split, the leads-sales regression, the cost/borrow timer, and the 12-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`deferred_revenue/`](deferred_revenue/). EDGAR XBRL `companyconcept` (deferred revenue,
revenue, assets) + yfinance adjusted closes; a **current-survivors** basket — survivorship named
on the Signal axis. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
