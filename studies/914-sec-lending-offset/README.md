# Study 914 — Securities-Lending Offset 📚

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the edge real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | On the three same-sponsor pairs the fee-adjusted residual is **+3.0 / +4.6 / +5.1 bp/yr** (\|*t*\| ≤ 1.04, every CI straddling zero); pooled **+3.7 bp/yr, *t* = +0.31** — and **−0.6 bp/yr (*t* = −0.05)** if the funds are charged their early-era fees instead. **The fee assumption owns the sign, so the study reports a magnitude bound and no direction:** every cheap leg's ER was cut during the sample (IVV 0.0945→0.03, IEMG 0.18→0.09), so pinning it at today's value inflates each residual by about its own size — sweep the cheap leg back to its earliest in-sample fee and all three turn negative (**−1.4 / −6.0 / −2.4 bp/yr**). What survives every scenario: \|residual\| ≲ **6 bp/yr**, CI edge ≈15 bp/yr. The two big residuals (IWM−IJR −114 bp, VEA−IEFA +49 bp) are index-composition wedges. Named limits: noise floor **20–147 bp/yr** against a **1–15 bp/yr** effect (underpowered by construction); the SPY/IVV control **confounds** lending with SPY's UIT dividend cash drag; all ten funds are **survivors**. |
| **Tradability** — is it bankable? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | There is no lending spread to bank. The **fee gap** is tradable — long IEMG / short EEM earns +47 bp/yr gross (Sharpe +0.38, *t* +1.69) — but it is **exactly zero at 50 bp/yr of borrow**, inside what an EM ETF actually costs to borrow. The one spread that pays (IJR/IWM, +110 bp) is the small-cap index quality premium wearing a 3.78% tracking error and *t* = +1.45. Nothing here clears the bar even before borrow. |

> **In one sentence:** Securities-lending revenue is real in the sponsors' annual reports and **too small to see in the tape** — the realised return gap between same-class ETF pairs is explained by the expense-ratio gap to within a handful of basis points, and since that handful is *smaller than the error in the fee input itself* (every cheap leg's fee was cut mid-sample, which flips the residual's sign), the only honest deliverable is a **magnitude bound** — ≲6 bp/yr per pair, ~25 bp/yr pooled — with no direction attached.

## What we tested

For five same-asset-class ETF pairs — **SPY/IVV, EEM/IEMG, EFA/IEFA, VEA/IEFA, IWM/IJR** —
the **fee-adjusted relative tracking residual**: annualised drift of the monthly log
total-return difference **plus** the expense-ratio gap. Zero means fees explained the whole
gap; positive means the dearer fund handed back more than its fee implied, which is what a
lending offset looks like from outside. HAC *t*, block-bootstrap CI, era cut, a
**two-sided expense-ratio sweep** — dear leg *and* cheap leg, plus the whole table re-run
under early-era / time-averaged / current fee schedules (ERs are a labelled PROXY, not
tape, and they decide the sign) — a daily-vs-monthly frequency check, pooled estimates with
and without the composition-wedge pairs, and a static dollar-neutral
**long-cheap/short-dear** trade with a **borrow sweep** (0→100 bp/yr, also an assumption;
no signal, so no execution lag is claimed). Total-return closes
(`auto_adjust=True`), as-of 2026-06-30. **Dedup:** distinct from **557-borrow-fee-signal**
(borrow fee as a *stock-return predictor*, not as fund revenue), **913-tracking-difference-persistence**
(does last year's best S&P tracker stay best — a persistence question on one asset class),
**920-total-cost-of-ownership** (fee vs spread break-even holding period), and
**378-etf-nav-premium** (a *pricing* wedge, not an *accrual* wedge). Fund-level lending
revenue is not public daily tape: everything here is an **inference from realised returns**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a fund's lending desk is, the one number that settles it, the fund that legally cannot lend, why the fee history decides the sign, why the tape could never have resolved this |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the estimand, five-pair table, SPY/IVV structural control and its confounds, the two-sided fee-history sweep, pooled residuals, era cut, frequency check, power arithmetic, borrow-swept pair trade, live synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`lending_offset/`](lending_offset/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
