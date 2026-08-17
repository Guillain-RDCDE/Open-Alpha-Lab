# Study 945 — The Hidden Financing 💳

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the charge real and measurable? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | Strip the published fee from the regression intercept and both wrappers imply the *same* borrowing rate — **2.07%** vs **2.08%**, a **+0.68 pp** mark-up over ^IRX at HAC *t* = **+3.25 / +3.65** (the intercept's *t* = −13 tests only that the drag is non-zero, which the fee alone guarantees — it is **not** the bar and is not quoted as one). Bootstrap CI [+0.39, +1.08]; both rate regimes clear \|*t*\| = 2; the all-in charge, which needs no fee assumption, is **+1.57 / +1.14 pp** at *t* = **+7.5 / +6.1**. Caveats carried, not buried: what survives the fee is financing **plus** swap and reset frictions (an **upper bound** on interest); the pre-2018 half is only **+0.37 at *t* = 1.74**, so the robust finding is the **post-2018 ~+0.95**; and SSO/UPRO are **survivors** of a cohort that lost members, both ProShares, so the estimate is a **floor** and their agreement is a consistency check, not independent confirmation. |
| **Tradability** — is it bankable? | ![Investable](https://img.shields.io/badge/Investable-2ea44f?style=flat-square) | Not alpha — a **cost decision** that survives because the margin is wide, not because the estimate is sharp. All-in you pay **^IRX + 1.57 pp** per borrowed dollar at 2x and **+1.14 pp** at 3x; do-it-yourself breaks even at **^IRX + 1.39% / + 1.04%**, and never above **+1.75%** in any era or rate cut — against mainstream retail margin at bills + 4 to 6%, safe by a factor of three (**+2.6 to +9.9 %/yr**; those races' *t* = 18-39 are mechanical, same exposure both sides, and are **not** evidence of an edge). Limits named: a prime-broker tier (bills + 0.75%) beats the wrapper by −0.64%/yr, and **Reg T does not permit the 3x DIY arm in a retail margin account** at all. |

> **In one sentence:** A leveraged ETF is a margin loan you never see quoted — the tape puts the rate at **T-bills + ~0.68 pp** (upper bound: swap and reset frictions are in there too), floating one-for-one with the Fed, which makes the all-in bill **bills + 1.6 pp** at 2x and **bills + 1.1 pp** at 3x — far cheaper than any margin desk an ordinary account can open, dearer than a prime broker no ordinary account can.

## What we tested

Regress **SSO** (2x) and **UPRO** (3x) daily **total returns** on **SPY**'s: the slope is the
realised leverage (1.997, 2.989 — indistinguishable from 2 and 3), the intercept is the whole
drag. Add back SPY's own fee (index basis), strip the prospectus expense ratio, divide by the
`L−1` borrowed dollars → the **implied financing rate**, raced against **^IRX**. Rolling 252-day
estimates through the 2009-2026 rate cycle, HAC *t*, a block-bootstrap CI, an era cut, a
rate-regime cut, an expense-ratio sweep, and a costed race against a self-financed daily-reset
margin replication (one execution lag; no shorts, so no borrow fee; no margin call modelled,
which flatters the DIY arm). 2009-06-26 → 2026-06-30, **survivor** wrappers only.
**Dedup:** [61-slow-burn](../61-slow-burn/) and [100-melting-ice](../100-melting-ice/) price the
*volatility drag* of the daily reset and **assume** an all-in fee (~5%/yr) — 945 **measures** that
fee and splits it into expense ratio and interest; [30-house-edge](../30-house-edge/) assumes a
broker mark-up over T-bills, which this study supplies empirically;
[943-leverage-reset-frequency](../943-leverage-reset-frequency/) and
[944-optimal-leverage-realized](../944-optimal-leverage-realized/) study the reset *shape* and the
optimal *amount* of leverage, [593-hfea](../593-hfea-leveraged-6040/) and
[594-leverage-rotation-200sma](../594-leverage-rotation-200sma/) use the wrappers as building
blocks. None of them backs the **interest rate** out of the wrapper or prices it against a margin
account. **Assumptions labelled and swept:** prospectus expense ratios, SPY's fee, and the broker
margin spreads (^IRX is quoted on a **discount** basis, ~2-3 bp below the bond-equivalent yield —
against us, not for us).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the invisible loan, the rate it charges, why 3x is the cheaper borrow, and when your own broker is cheaper |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the HAC regression, the price-index trap, bootstrap CI, rolling pass-through, era/rate-regime cuts, the ER sweep, the break-even race, the live synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`lev_financing/`](lev_financing/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
