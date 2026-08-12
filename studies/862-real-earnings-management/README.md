# Study 862 — Real Earnings Management 🎭

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a Roychowdhury REM proxy predict forward returns? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | A monthly tercile long-short on `rem = ab_prod − ab_disx` (long the overproducers/expense-cutters, short the clean firms) is a **flat null**: Newey-West *t* = **−0.16** (mean −9.2 bps/mo). It isn't even **sign-stable** — flip the staleness window (200→120 d) and it jumps to +75.3 bps (NW *t* = +1.82); split at 2015 and it goes **+69.5 bps pre / −62.8 bps post**. The pooled event drift is **flat, sign-flipping and non-monotone** (placebo *p* ≈ 0.5–0.9). No robustness, no stable sign. |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Nothing to trade: a (slightly negative) zero gross edge. Net of 20 bps + 100 bps borrow it is −1.05%/yr, NW *t* = **−0.16**, Sharpe **−0.06**. |
| **Does REM foreshadow an operating reversal?** | ![Not detected](https://img.shields.io/badge/Operating_reversal%3F-Not_detected-8b949e?style=flat-square) | The forward gross-margin change regresses on REM with a tiny, insignificant, **wrong-signed** slope (+0.057, *t* = +1.36, R² = 0.005). On this industry-pooled EDGAR panel the mechanism is simply not observable — a non-detection, honestly stamped, not a refutation of Roychowdhury's industry-year original. |

> **In one sentence:** Roychowdhury's real-earnings-management fingerprints (abnormally high
> production + abnormally low R&D/SG&A) are a beautiful *forensic* idea, but ranking large US
> manufacturers on the proxy and holding a month buys you **a coin-flip** — a Newey-West *t* of
> −0.16 whose very sign flips when you nudge the staleness window or cut the sample at 2015: **no
> return signal, and on this thin panel not even the operating reversal shows up.**

## What we tested

Roychowdhury (2006), taken to the tape: firms that need to hit a number by **real actions**
overproduce (building inventory so fixed overhead spreads thin and reported COGS falls) and **cut
discretionary spend** (R&D, SG&A), leaving an *abnormally high production cost* and an *abnormally
low discretionary expense* versus a normal-operations benchmark. We estimate Roychowdhury's two
normal-level regressions on **44 large US manufacturers / hardware / pharma / industrials** that
disclose the needed line items on EDGAR (`Revenues`, `CostOfRevenue`, `SG&A`,
`ResearchAndDevelopmentExpense`, `InventoryNet`, `Assets`) — **32** clear the ≥6-quarter usable
bar — take the residuals as `ab_prod` and `ab_disx`, and form `rem = ab_prod − ab_disx`, ranked
**point-in-time on the 10-Q/10-K filing date** (zero look-ahead). We split the claim in two: a
monthly tercile **long-short** held one month forward (the return claim), graded on an
autocorrelation-robust **Newey-West *t***, cross-checked by a pooled event drift + two-sided
label-shuffle placebo, an era split, a staleness cut, the two components separately, and a 12-seed
synthetic control; and a pooled regression of the **forward gross-margin change** on REM (the
operating-reversal mechanism). Costs are one-way × NAV × turnover with the short leg paying borrow.
**Coverage is thin and uneven** — Q4 flow tags are sparse, several names never report R&D, the
cross-section averages ≈17 — and the normal-model coefficients are pooled in-sample across
industries; we say so throughout. **Dedup:** [574-penny-beat](../574-penny-beat/) documents the
*just-beat* discontinuity this is one *mechanism* for; [229-beneish-m-score](../229-beneish-m-score/)
and [855-accrual-quality](../855-accrual-quality/) detect *accrual* manipulation, the channel REM
deliberately avoids; [525-r-and-d-intensity](../525-r-and-d-intensity/) ranks on the R&D *level*,
whereas we use R&D only inside the abnormal-discretionary-expense *residual*. None ranks on the
Roychowdhury REM proxy — this study does. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "managing earnings by real actions" means, why a good forensic tell is not a good trading signal, and how a signal that flips sign on a knob-twist gives itself away |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the two Roychowdhury normal-level regressions, the calendar-time Newey-West long-short, the sign-instability across staleness and eras, the pooled event drift + placebo + monotonicity, the gross-margin-reversal regression, the cost/borrow timer, and the 12-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`real_earn_mgmt/`](real_earn_mgmt/). EDGAR XBRL `companyconcept` (revenue, COGS, SG&A,
R&D, inventory, assets) + yfinance adjusted closes; a **current-survivors** manufacturer basket —
survivorship named on the Signal axis. **Not investment advice** — research & education. See
[LICENSE](../../LICENSE).*
