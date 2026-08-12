# Study 858 — Margin ÷ Inventory Divergence 📦

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the margin/inventory contradiction sort returns? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | A monthly tercile long-short (long the *coherent* names, short the *contradictory* ones) earns a **slightly wrong-signed null**: Newey-West *t* = **−0.18** (one-sample −0.20), and it isn't even sign-stable — the staleness-120 variant flips to +0.09, the era split runs **+18.6 bps (pre-2016) then −23.8 bps (post)**, and the pooled event drift is right-signed but **flat, non-monotone, placebo-insignificant** (*t* ≤ +1.34, *p* ≈ 0.06–0.20). No spec approaches *t* = 2. |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Negative gross, negative net (20 bps + 100 bps borrow → −2.76%/yr, NW *t* = −0.51, Sharpe −0.14). Nothing to trade. |
| **Does it lead the fundamentals?** | ![No](https://img.shields.io/badge/Leads_margin%3F-No-8b949e?style=flat-square) | The divergence does **not** predict next year's gross-margin change (correlation **−0.03**, wrong-signed −0.56 pp tercile spread). Even the accounting mechanism that could have made it "real info, no alpha" is absent. |

> **In one sentence:** the Abarbanell–Bushee "rising margin + inventory outrunning sales =
> contradiction" signal is one of accounting's most famous, most data-mined edges — and on 40
> inventory-carrying US names, 2009→2026, it sorts neither **returns** (NW *t* = −0.18,
> sign-flipping across eras) **nor** the **fundamentals** it was built to forecast (next-year
> margin correlation −0.03): a clean null on both axes.

## What we tested

The fundamental-analysis staple from Lev–Thiagarajan (1993) and Abarbanell–Bushee (1997/1998),
stated the way its believers state it: *"if a firm's gross margin is climbing while its
inventory grows faster than its sales, something is wrong — the margin is unsustainable or the
inventory is about to be written down, so short it; reward the coherent names instead."* We
collapse the two classic signals into one number — **`divergence = (ΔGross-margin%) −
(inventory-growth − sales-growth)`** — on **40 inventory-carrying US names that report
Revenues, CostOfRevenue and InventoryNet on EDGAR** (retailers, manufacturers, staples,
apparel, hardware/semis, autos), 2009→2026, ranked **point-in-time on the 10-Q/10-K filing
date** (zero look-ahead). We split the claim in two: a monthly tercile **long-short** held one
month forward (the return claim), graded on an autocorrelation-robust **Newey-West *t***,
cross-checked by a pooled event drift + label-shuffle placebo, an era split, a classic
inventory-vs-sales-gap variant, and a 12-seed synthetic control — plus a pooled regression of
*next-year gross-margin change* on the signal (the accounting mechanism). Costs are one-way ×
NAV × turnover with the short leg paying borrow. **Coverage is thin and uneven** — the
quarterly-span filter drops fiscal-Q4 figures and the cross-section averages ≈17–20 names — and
we say so throughout.
**Dedup:** [529-inventory-growth](../529-inventory-growth/) ranks on the *level* of inventory
growth alone; [854-cash-conversion-cycle](../854-cash-conversion-cycle/) is the *working-capital
cycle* (days inventory + receivables − payables); [122-gross-profitability](../122-gross-profitability/)
is Novy-Marx *gross profits ÷ assets* (a level, not a change); [231-sloan-accruals](../231-sloan-accruals/)
is *total accruals* earnings quality. None ranks on the **margin-vs-inventory-vs-sales
divergence** itself — this study does. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the "contradiction" story is so seductive, why the stocks ignore it, and why even the accounting mechanism fails to show up |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the calendar-time Newey-West long-short, the pooled event drift + placebo + monotonicity, the sign-flipping era split, the leads-margin regression, the cost/borrow timer, and the 12-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`margin_inventory/`](margin_inventory/). EDGAR XBRL `companyconcept` (revenue, cost of
revenue, inventory) + yfinance adjusted closes; a **current-survivors** inventory-carrying
basket — survivorship named on the Signal axis. **Not investment advice** — research &
education. See [LICENSE](../../LICENSE).*
