# Study 617 — Crash-Insurance-Cost ☂️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does crash insurance really bleed every year? | ![Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square) | Three ways on the real tape: TAIL's monthly drift **−56.5 bps/mo** (HAC *t* = **−2.17**), its alpha vs its own Treasury collateral **−7.69%/yr** (monthly HAC *t* = **−3.28**, negative in both crash-free sub-periods), and the buyer-side variance premium on 33 yrs of ^VIX+SPY: implied 20.98% vs realized 18.58%, RV−IV HAC *t* = **−3.12** (vol-points *t* = −9.42), buyer loses **5 months in 6**. Survivorship runs *against* the claim (TAIL is the surviving tail fund) — it clears anyway. |
| **Tradability** — is buying the insurance deployable? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The long side has no edge to deploy: a dollar at launch is **50 cents** today (net, total-return); every SPY/TAIL blend drags CAGR **−1.1 to −4.3 pp/yr** at 5 bps for a 1–7 pp drawdown trim; the one jackpot (COVID: **+28.5%** while SPY fell a third) was fully given back in **337 days**. The harvestable side is the *seller's* — see [92-easy-money](../92-easy-money/) and [63-free-fall](../63-free-fall/). |
| **"Did the 2020 jackpot leave anyone ahead?"** | ![Busted](https://img.shields.io/badge/2020_win_kept%3F-Busted-8b949e?style=flat-square) | **1 of 110** month-end cohorts is above water today (+0.10%, a 2025 entry that never saw the crash). 72% of cohorts *were* ahead mid-flight — almost all via the COVID spike — and every one gave it back; the at-inception buyer peaked at **+4.22%**. |

> **In one sentence:** the folk claim is *true and priced* — TAIL, the live buyable crash-insurance
> ETF, bleeds ≈**7.7%/yr** against its own collateral (monthly HAC *t* = −3.28) because its buyers
> are paying the same volatility risk premium the desk's short-vol studies harvest (implied 21.0
> vs realized 18.6 vol, *t* = −3.12 over 33 years), and even the perfect 2020 crash payoff was
> gone within 337 days — so the Signal is **Real** and the sign hands the money to the **seller**.

## What we tested

TAIL (Cambria Tail Risk ETF, 2017→, ~90% intermediate Treasuries + a laddered OTM SPX put budget,
0.59%/yr ER inside the NAV) is the cleanest live quote of what crash insurance *costs*. We measure
the bleed four ways on yfinance tapes (total-return for TAIL/IEF/SPY, ^VIX as a level): raw drift
since inception with HAC/Newey-West *t*; the **alpha of TAIL on IEF** — its own collateral — which
isolates the put sleeve + fee from bond duration (headline: monthly, both crash-free sub-periods
shown); the **2020 payoff episode** quantified end-to-end (gain, giveback date, what an inception
holder actually had); and the premium **named** via variance-swap arithmetic on real ^VIX+SPY
(prior month-end VIX² vs next month's realized variance — one clean one-month lag, labeled
model-derived). The third axis runs every month-end **entry cohort** to as-of: did anyone keep the
2020 win? Tradability charges 5 bps one-way × turnover on monthly-rebalanced SPY/TAIL blends. A
20-seed synthetic control (fair insurance vs planted bleed) proves the decomposition flags nothing
when insurance is fair and recovers a planted bleed exactly. Siblings, framed distinctly:
[92-easy-money](../92-easy-money/) and [63-free-fall](../63-free-fall/) tested *selling* this
premium (both Real); [86-tail-radar](../86-tail-radar/) tested the SKEW index as a crash *signal*
(None). Here: the **buyer's bill, on a live product**. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a fund that did exactly what its prospectus promised still halved your money, what the 2020 spike really paid, and who pockets the insurance premium — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC drift + alpha-vs-IEF decomposition (daily/monthly/sub-periods), the RV−IV variance-premium series, cohort accounting, blend costs, and the 20-seed fair-vs-planted-bleed control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`crash_insurance_cost/`](crash_insurance_cost/). TAIL buy-and-hold is static (no signal,
no lag); the variance-premium series uses strictly prior-month VIX (one documented one-month lag).
All TAIL numbers net of its 0.59%/yr ER; blends net of 5 bps one-way × turnover; total-return vs
level labeled everywhere. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
