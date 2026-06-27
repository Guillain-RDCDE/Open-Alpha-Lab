# Study 529 — Inventory-Growth 📦

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## The claim

> *Do firms that pile up inventory go on to underperform the lean ones?*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do low-inventory-growth firms beat high ones? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | On the survivor basket the long-low/short-high hedge earns **+4.6%/yr at HAC *t* = 0.50**, clears the label-shuffle placebo with **p = 0.38**, and *flips sign* between the only two usable years (+22.9% then −13.7%). No certifiable signal. |
| **Tradability** — does the spread pay? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Nets **+4.0%/yr (t 0.44)** after 5 bps/leg + 50 bps borrow — indistinguishable from zero over n=2 years. Nothing to trade. The basket is survivor-biased, excluding the very over-builders the anomaly is built around. |
| **Data depth** — is the panel deep enough to even test it? | ![Insufficient](https://img.shields.io/badge/Data_depth-Insufficient-8b949e?style=flat-square) | No. Yahoo serves ~4–5 annual balance sheets per name → after the reporting lag and a complete-year return filter, only **2** hedge years survive. The anomaly is **untestable here, not disproven**. |

> **In one sentence:** Belo–Lin (2012) and Thomas–Zhang (2002) document a real low-minus-high inventory-growth premium on decades of Compustat data, but Yahoo's four-deep balance-sheet history leaves only two usable years on a 37-name survivor basket — the hedge (+4.6%/yr, *t* 0.50, placebo *p* 0.38) flips sign between them, so the effect is untestable here rather than confirmed or denied.

## What we tested

The **inventory-growth anomaly** (Belo & Lin 2012; Thomas & Zhang 2002): firms that build
inventory aggressively earn *low* future returns — a real-investment / over-extrapolation
story in the asset-growth family. We compute the signal exactly as the accounting literature
does — the inventory *change* scaled by lagged total assets,
**INVG_t = (Inventory_t − Inventory_{t-1}) / Total_Assets_{t-1}** — pull it from Yahoo
`balance_sheet` filings for a 40-name inventory-heavy survivor basket (retail, consumer
staples, industrials, autos), lag it one full year, and sort into quintiles. The tradable
expression is **long low-INVG, short high-INVG**, measured with a one-sample HAC *t*, a
label-shuffle placebo null, a random-portfolio control, and costs + short borrow. The
offline control is a deterministic synthetic panel with a dial-able inventory-growth premium
(and a null), averaged over 20 seeds to prove the engine is faithful.

The honest constraint is **data depth**: Yahoo's annual statements only go ~4 years deep,
so after the lag and a complete-year return filter only two hedge years remain — too few to
certify or refute the effect. *Distinct from [244 Asset-Growth](../244-asset-growth/) (total
assets, not inventory), [231 Sloan-Accruals](../231-sloan-accruals/) (working-capital
accruals) and [522 Percent-Operating-Accruals](../522-percent-operating-accruals/).*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what inventory growth is, why over-building is *supposed* to predict pain, and why four years of Yahoo data can't tell us |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the quintile sort, the hedge with one-sample HAC *t*, the label-shuffle placebo, costs + borrow, and the seed-robust synthetic positive control |

The fingerprinted real-data run (37-name basket, 2 usable hedge years, INVG fp `044108b907ab`)
is in [docs/results.md](docs/results.md); the offline machinery proof runs on the synthetic
panel in [`inventory_growth/data.py`](inventory_growth/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
