# Study 853 — Days-Sales-Outstanding Buildup 🧾

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a rising DSO predict lower forward returns? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | A monthly tercile long-short (long the lean-DSO third, short the building-DSO third) earns a *right-signed but trivial* **+14.2 bps/mo (+1.7%/yr gross)** — and it is **statistically zero**: Newey-West *t* = **+0.53** (one-sample +0.55). It doesn't survive perturbation (staleness-120 NW *t* = **+0.16**, percentage-change variant **+0.27**), the pooled event drift is **flat and sign-flipping** (long-short *t* +0.01 → +0.11 → **−0.74** at 2 quarters, placebo *p* 0.44-0.81), and it is **dead after 2018** (*t* = +0.02). No coherent, robust effect. |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | It fails **before** costs (gross NW *t* ≈ 0.5) and turns **negative** after realistic friction: net of 20 bps + 100 bps borrow it is **−0.13%/yr**, NW *t* = **−0.04**, Sharpe **−0.01**. Nothing to trade. |
| **Does a building DSO warn on future sales?** | ![Faint](https://img.shields.io/badge/Warns_on_sales%3F-Faint-8b949e?style=flat-square) | **Barely.** DSO buildup is *correctly signed* against next-quarter sales (slope −0.0031) and the lean third grows sales **+3.3 pp** faster than the building third — but the correlation is a whisker (**−0.08**, R² 0.007). The channel-stuffing mechanism exists in direction, is negligible in size. |

> **In one sentence:** the Abarbanell-Bushee receivables red flag — rising Days Sales Outstanding
> as a warning of channel-stuffing and weak collections — is, on a liquid large-cap tape, a
> **near-perfect null on returns** (NW *t* = +0.53, dead post-2018, sign-flipping event drift) and
> only a **faint, correctly-signed whisper on sales** (+3.3 pp, correlation −0.08): **a famous
> forensic signal that a public ratio on blue chips prices away entirely.**

## What we tested

The forensic-accounting staple, stated the way its believers state it: *"when receivables grow
faster than sales, Days Sales Outstanding rises — that's channel-stuffing / aggressive revenue
recognition / weak collections, and it precedes disappointment, so short the building-DSO names and
favour the lean ones."* We take it literally on **38 large US filers with real trade receivables**
(a 48-name basket; financials excluded, DSO is undefined for them) that report both
`AccountsReceivableNetCurrent` (→ `ReceivablesNetCurrent`) and quarterly `Revenues`
(→ `RevenueFromContractWithCustomerExcludingAssessedTax`) on EDGAR, 2009→2026, ranked **point-in-time
on the 10-Q/10-K filing date** (zero look-ahead), with DSO = AR ÷ (annualised revenue ÷ 365). We
split the claim: a pooled regression of *next-quarter revenue growth* on the DSO change (the
channel-stuffing mechanism) and a monthly tercile **long-short** held one month forward (the return
claim), graded on an autocorrelation-robust **Newey-West *t***, cross-checked by a pooled event
drift + label-shuffle placebo, an era split at 2018, a percentage-change variant, and a 12-seed
synthetic control. Costs are one-way × NAV × turnover with the short leg paying borrow. **Coverage
is thin and uneven** — the matched cross-section *shrinks* from ≈27 names (pre-2013) to ≈12
(mid-2020s) as ASC-606 revenue-tag switches break the YoY match — and we say so throughout.
**Dedup:** [231-sloan-accruals](../231-sloan-accruals/) and
[522-percent-operating-accruals](../522-percent-operating-accruals/) rank on *aggregate accruals*;
[529-inventory-growth](../529-inventory-growth/) is the *inventory* working-capital red flag;
[855-accrual-quality](../855-accrual-quality/) grades *accrual reliability*. None ranks on the
**year-over-year change in Days Sales Outstanding** itself — this study does. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a rising invoice pile *sounds* like a red flag, why the stocks don't react, and how a signal can be a faint truth about the business and pure noise about the stock |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the calendar-time Newey-West long-short, the pooled event drift + placebo + monotonicity, the era split, the winsorised warns-on-sales regression, the cost/borrow timer, and the 12-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`dso_signal/`](dso_signal/). EDGAR XBRL `companyconcept` (receivables, revenue) +
yfinance adjusted closes; a **current-survivors** basket — survivorship named on the Signal axis.
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
