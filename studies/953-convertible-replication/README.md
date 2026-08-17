# Study 953 — Replicating the Convert 🎭

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the wrapper add anything? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Out-of-sample residual alpha over a **frozen** SPY/QQQ/LQD/cash mix is **+1.80%/yr, HAC *t* = +0.58**, bootstrap CI **[−4.5%, +7.6%]** — zero, and not era-robust (+4.7%/yr in 2018-21, −0.8%/yr in 2022-26). Across **11 specifications** (4 replication kits × 7 in-sample boundaries) the largest \|*t*\| is **0.94**; ICVT gives **−2.16%/yr (*t* = −0.56)**. The convexity claim fails separately: up-capture **1.219** ≈ down-capture **1.215** (asymmetry **+0.004**), tail smile **−22 bps/month** (*t* = −0.50), Treynor-Mazuy γ = +0.19 (*t* = +0.25). **Survivorship:** CWB and ICVT are two *surviving* listed funds — the wrappers that closed are not in this tape, so the sample is tilted in the wrapper's favour and still comes up blank. |
| **Tradability** — is it bankable? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Nothing to bank either way. Long the fund: a coin-flip residual against a certain 0.40%/yr fee. Short it against the replica: a 2%/yr expectation inside a **7.9%/yr** monthly tracking error, skewed against you (**−6.8% in March 2020 alone**) — and the replication is a good *description*, not a tight clone (monthly R² 0.764). The frozen replica even edged the fund on excess Sharpe (**0.725** vs 0.662) at a shallower drawdown. |

> **In one sentence:** a static 37% SPY / 20% QQQ / 6% LQD / 37% cash mix, fitted before 2018 and never touched again, tracks the biggest convertible ETF well enough that the fund's remaining alpha is statistically zero — and the fund's payoff against that mix is a straight line, not the promised smile, because what a convertible wrapper actually sells you is 22% more beta in **both** directions.

## What we tested

The convertible pitch says the instrument is *not* decomposable: a bond floor plus an
equity call gives you a **convex** payoff you cannot build from parts. We take the
decomposition literally. CWB's daily **total-return** excess-of-cash series is fitted, by
**Sharpe-style constrained least squares** (weights non-negative, summing to at most one,
so the replica is a portfolio a human can hold and no borrow arises), to **SPY + QQQ + LQD**
with the remainder in **BIL** — fitted on **2009-2017 only**, then **frozen** and scored on
the **2018-2026 hold-out** with a Newey-West *t*, a block-bootstrap CI, an era cut, an
annual-refit variant, a cost/fee sweep and an ICVT cross-check. Convexity is then tested
*against the replica*: out-of-sample months sorted into terciles by SPY's excess return,
plus up-/down-capture and a Treynor-Mazuy curvature on the fund-minus-replica difference —
a **description** of the realised shape (breakpoints are full-window), never a trading rule.
One execution lag throughout (weights applied from the first day after the in-sample end;
the month-end drift is carried into the next day's return and traded out at *that* close).
Both judgement calls are swept, not defended: the **QQQ leg** is hindsight-flavoured, so the
kit is stripped back to SPY-only (which leaves a *bigger* residual, +3.0%/yr, *t* = +0.92),
and the 2017 boundary is walked across seven years. The replica's 10 bps/yr fee drag is a
labelled **PROXY** swept 0-20; the funds' own expense ratios are quoted from the issuers, not
measured. **Dedup:** [Study 339 (Convertible-Bonds)](../339-convertible-bonds/) tests
CWB's convexity against **SPY directly** and against a *beta-matched* two-asset SPY/AGG
blend, in-sample; 953 replaces that with a **fitted three-factor long-only replication**
frozen before a genuine hold-out, and asks the convexity question *against the replica*
rather than against the market. [Study 97 (Balancing-Act)](../97-balancing-act/) tests
the plain stock/bond mix itself; [Study 912 (Gold + Trend)](../912-gold-trend-managed/)
shares the inference spine but times an asset rather than replicating a fund.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the recipe three ETFs write for a convertible fund, why "more beta" is not "convexity", and the March-2020 month where the bond floor was not there |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the constrained fit, the frozen hold-out with HAC *t* and bootstrap CI, the tercile smile, Treynor-Mazuy γ, era cut, kit / boundary / cost / fee sweeps, annual refit, ICVT, and the live synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run (fp `5e5674563124`, as-of 2026-06-30): [docs/results.md](docs/results.md), via [examples/verify.py](examples/verify.py).

---

*Engine: [`quantlab/`](../../quantlab/) + [`convert_repl/`](convert_repl/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
