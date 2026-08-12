# Study 910 — Managed-Distribution CEF 🎁

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the discount + payout beat the asset class? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The payout is **real**: the equal-weight CEF basket earns **+88 bps/mo excess-of-cash (HAC *t* = +2.63)**, bootstrap Sharpe CI **[0.12, 1.17]** clear of zero (PCEF +49 bps, *t* = +2.48) — total returns clear cash, so these are **not** the capital-destroying mirage of an mREIT. **But the claimed edge — beating the asset class — fails:** the excess-vs-excess Sharpe **trails SPY in every fund and every sub-era** (basket adv **−0.21**; −0.10 pre-2022, −0.36 post; PCEF −0.41), the CAPM alpha is ~0-to-negative (basket −1.6 %/yr *t* = −0.59; PCEF −3.3 %/yr *t* = −1.96), and β ≈ 1.0 (R² 0.73) exposes a **levered-beta income clone**. *Survivorship + ~11.5y basket history bias the magnitude upward.* |
| **Tradability** — can you bank it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Costs are **not** the killer — buy-and-hold, ~15 bps CEF spread, net ≈ gross (+88.0 bps). A **hidden levered equity beta** is: you get a **better Sharpe just holding SPY** (0.82 vs 0.61), and when the 2022 rate-hike regime made leverage expensive and blew out discounts the excess return collapsed to **+32 bps (*t* = 0.62)** with a −5.5 %/yr alpha and a −24 % drawdown year (RQI alone: −87 % max DD). A hidden beta erasing the edge is the textbook Mirage. |

> **In one sentence:** the "discount pull *plus* payout" double-carry is half-true — the payout is
> a genuine excess-of-cash return (not an mREIT-style capital shredder), but the asset-class *edge*
> is a levered-beta illusion that a plain SPY position beats risk-adjusted, and that the 2022 rate
> shock turned negative.

## What we tested

A closed-end fund at a persistent discount with a big managed distribution supposedly hands the
buyer the discount pull *and* the payout. We test the buyer's bottom line on liquid tape:
**PCEF** (the CEF-of-CEFs) and an equal-weight, monthly-rebalanced basket of four large
single-name CEFs — **PDI** (bond), **UTF** (infrastructure), **BST** (tech option-overwrite),
**RQI** (real estate) — vs **SPY**, everything **excess of BIL** (cash) so leverage can't hide in
the Sharpe. yfinance `auto_adjust=True` **total-return** closes mean every distribution is
reinvested, so the number we hold is the honest economic return regardless of how much was
labelled "distribution". Inference is **Newey-West HAC** (6 lags) on the excess-of-cash mean and
on the excess-vs-excess CAPM `r_ex(fund) = α + β·r_ex(SPY)` — β is the leverage/asset-class
exposure, α the structural pickup that survives beta — plus a moving-block bootstrap Sharpe CI, a
2022-rate-hike era cut, a calendar-year / max-drawdown table, and a costed net. A deterministic
synthetic world with a **planted, tunable net carry** and a **return-of-capital leak** proves the
estimator recovers the knob, that a pure-levered-beta null cannot fire, and that a 100 %-ROC
payout yields no alpha. **Dedup:** [367-closed-end-fund-discount](../367-closed-end-fund-discount/)
trades the **discount cross-section** (NAV-vs-price, which yfinance can't see) — we test the
**hold-the-payout total return** instead; [611-mreit-carry](../611-mreit-carry/) is the
leverage-financed-carry sibling this study checks these CEFs against;
[342-bdc-yield](../342-bdc-yield/) is the BDC private-lending cousin;
[616-muni-cef-tax-loss](../616-muni-cef-tax-loss/) is the seasonal tax-loss angle, not a carry
hold. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a managed-distribution CEF is, why a fat "yield" can be your own money handed back (return-of-capital), and why buying a dollar of assets for 90 cents doesn't mechanically beat the index — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the excess-of-cash Sharpe race vs SPY, HAC *t*'s, the excess-vs-excess CAPM β/α, the bootstrap Sharpe CI, the 2022 era cut, the cost math, and the planted-carry / return-of-capital-trap synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`md_cef/`](md_cef/). All Sharpes are excess-of-cash (both legs minus BIL); returns are
yfinance total-return (distributions reinvested); the discount/NAV series is **not** observed — we
test the buyer's total-return bottom line. **Not investment advice** — research & education. See
[LICENSE](../../LICENSE).*
