# Study 568 — Effective-Tax-Rate 🧾

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do low-effective-tax-rate firms earn different returns? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The long-low / short-high-ETR hedge earned **−3.3%/yr** over 17 years (HAC *t* **−0.91**) — the *wrong* sign for the quality story and indistinguishable from zero. **Placebo *p* = 0.42**, rank-**IC = −0.02** (*t* −0.46), the change-in-ETR signal is equally dead, and the sign **flips across windows** (+3.0% early, −4.2% late), never clearing |*t*| = 1.1. No robust *t* ≥ 2 on the real tape. Survivor basket. |
| **Tradability** — does the ETR sort pay? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | A 40-name survivor basket, annual rebalance, the short leg being high-ETR energy/industrials. It loses before costs (gross **−3.3%/yr**) and more after (net **−3.9%/yr** at 10 bps/leg + 50 bps borrow). Nothing to harvest. |

> **In one sentence:** the effective-tax-rate anomaly — low-ETR firms as either efficient
> quality (higher returns) or fragile loopholes (lower returns) — simply **does not appear** on a
> 40-name large-cap survivor basket over 2008-24: the low-minus-high-ETR hedge is a small,
> wrong-signed, insignificant −3.3%/yr (placebo *p* 0.42, IC ≈ 0, sign-unstable), because on blue
> chips the ETR sort mostly picks up *sector* and the survivor basket has already deleted the
> aggressive-tax firms that blew up.

## What we tested

The **effective-tax-rate return anomaly**: a firm's ETR (income-tax expense / pretax income) is
read two ways — the *quality / tax-avoidance premium* (low-ETR firms are efficient cash machines
the market underprices → **higher** returns) and the *red-flag / risk* story (a suspiciously low
ETR is a fragile loophole that reverses → **lower** returns). Because the literature contests the
*sign*, we build a **low-minus-high ETR** quintile sort on real EDGAR fundamentals (income-tax
expense, pretax income) + yfinance prices, and report its HAC *t*, a **label-shuffle placebo**
null, the rank-**information coefficient**, a **change-in-ETR** second signal, a **window-stability**
sweep, costs + a short-leg borrow, and a deterministic seed-robust synthetic positive control that
plants a low-ETR premium of *either* sign and proves the engine catches it. *Distinct from
[192 tax-day](../192-tax-day/) (a calendar seasonal), and from the quality/accrual sorts
[122 gross-profitability](../122-gross-profitability/), [200 ROE-quality](../200-roe-quality/),
[231 Sloan](../231-sloan-accruals/), [521](../521-cash-based-operating-profitability/) /
[522 accruals](../522-percent-operating-accruals/) — this study's sort key is the **tax line** itself.*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what an effective tax rate is, why "pays little tax" could mean *better* or *worse* stocks, and why on this basket it meant neither |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the quintile sort with a HAC *t*, the placebo null, the rank-IC, the change-in-ETR signal, the window sign-flip, costs + borrow, and the seed-robust synthetic positive control |

The fingerprinted real-data run (40 survivors, fiscal 2008 → 2024, ETR panel fp `74b7313496ca`) is
in [docs/results.md](docs/results.md); the offline machinery proof runs on the deterministic
synthetic world in [`effective_tax_rate/data.py`](effective_tax_rate/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`effective_tax_rate/`](effective_tax_rate/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
