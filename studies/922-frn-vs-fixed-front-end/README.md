# Study 922 — Floating-Rate Front End 🪙

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the floating end out-pay the fixed end? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The direction is textbook and the sign never wavers — floaters beat 1-3y fixed by **+4.46 pp/yr** through the 2022-23 hikes, lose to it by **0.4-0.7 pp/yr** when rates sit still, and the rising-minus-falling contrast is positive in **12/12** classifier settings. **This tape certifies none of it.** The headline contrast is **+2.43 pp/yr, HAC *t* = +1.37**, and clears \|*t*\| = 2 in **0/12** settings; the unconditional USFR−BIL and USFR−SHY gaps are *t* = +0.68 and +0.96, and all three bootstrap CIs straddle zero. Duration arithmetic says real; twelve years containing **one** hiking cycle cannot say so. |
| **Tradability** — is it bankable? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | The floater-over-bills pickup is ≈ **+15 bps/yr**, stable in both eras and untouched by cost (5 bps round trip costs 20 bps over a one-year hold, 7 over three) — but that is a fee-sized decision, not an edge. The floater-over-SHY choice pays **only if you know which way rates are going**; as a dollar-neutral pair it is **negative past 50 bps of borrow**. The one durable win is a *risk* one: post-2018 the floaters delivered SHY's return with a **−0.4% worst drawdown against −5.7%**. |

> **In one sentence:** across a full hike-plateau-cut cycle the Treasury floaters (USFR, TFLO) out-earned both T-bills and 1-3y fixed paper — but by **16 to 44 bps a year**, entirely earned in one 17-month tightening window, with no unconditional *t* above **1.1**, no classifier setting clearing 2, and a duration give-up that still has not been paid back in the cutting cycle it was supposed to win.

## What we tested

Four front-end sleeves **held, not traded**: **USFR** and **TFLO** (Treasury floating-rate
notes, coupon resets weekly off the 13-week bill, ~0 duration) against **BIL** (1-3 month
bills) and **SHY** (1-3 year fixed, ~1.85 years of duration), on daily **total-return**
closes 2014-02-04 → 2026-06-30. HAC *t* on every pairwise daily difference, a regime cut on
the direction of **^IRX** (a price-only yield index, lagged one day — the study's single
execution lag), one HAC-OLS *contrast* test for the rate-direction flip, a drawdown table, a
launch-liquidity era cut, block-bootstrap CIs, and cost / borrow / cash-proxy / classifier
sweeps. **Dedup:** **[921-bill-ladder-vs-etf](../921-bill-ladder-vs-etf/)** asks the *fee*
question inside the bill sleeve; **925-short-rate-momentum-switch** (a sibling candidate,
not yet built) would *trade* front-end duration on a trend signal, where we deliberately
trade nothing;
**[885-ultra-short-credit-pickup](../885-ultra-short-credit-pickup/)** sources the same
"cash with a pickup" pitch from credit spread, not a floating Treasury coupon;
**[826-treasury-duration-bab](../826-treasury-duration-bab/)** is a levered cross-maturity
factor; **[16-storm-shy](../16-storm-shy/)** uses SHY as an equity-crisis hedge.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a floating coupon is, why 2022 was its year, what it costs you the rest of the time, the honest verdict |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the pairwise race, the HAC-OLS regime contrast, bootstrap CIs, cycle windows, borrow and cash-proxy sweeps, the live synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`frn_front/`](frn_front/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
