# Study 926 — T+1 ⏱️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the edge real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The overnight share of realised variance — the one thing a settlement change should move — shifts **+0.023 (HAC *t* = +1.03)** for EFA and **−0.017 (*t* = −0.73)** for EEM: two treated legs, opposite signs, bootstrap CIs straddling zero. The only two significant results are a total-vol measure driven by SPY's own compression (it **flips sign** at a one-year window) and the *domestic placebo* IWM. Decisive: across every outcome and leg, the **median arbitrary placebo date beats the real one** — 61 of 81 fake dates exceed EEM's headline *t* = 3.39, at a median of 6.26. |
| **Tradability** — is it bankable? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The month-end EFA/EEM-vs-SPY spread changes by **+7 bps per on-day, HAC *t* = +0.4 to +0.6** at the switch — zero gross, zero at every point of the cost × borrow sweep. The EEM spread's post-period Sharpe of +0.96 is not a T+1 effect: it paid +10.75 bps/on-day *before* the switch as well, on 100 on-days with its own *t* of +1.40. |

> **In one sentence:** The May-2024 move to T+1 settlement left **no fingerprint at all** in the daily open/close tape of SPY, IWM, EFA or EEM — the plumbing change moved who funds what overnight, not where price is discovered — and the two apparently significant results dissolve the moment you ask what an *arbitrary* date in 2022–2026 would have produced.

## What we tested

A difference-in-difference around **28 May 2024**, when US equities and ETFs moved from
T+2 to **T+1** while Europe and Asia stayed on T+2. **EFA** and **EEM** are the
settlement-mismatched legs (shares settle T+1, holdings settle T+2), **SPY** is the
control and **IWM** is a *domestic placebo-treated* leg. Four daily outcomes from the
exact night/day identity — overnight return, overnight |return|, the overnight share of
realised variance, and total |return| — over **symmetric 524-day windows**, with HAC *t*,
block-bootstrap CIs, a window-length sweep, 81–93 **placebo switch dates** per leg, an
excess-of-cash Sharpe race vs BIL, and one tradable turn-of-month long/short spread
(one execution lag, so the days actually held are the first four of each month; cost ×
borrow swept). **Dedup:** distinct from **01-overnight-anomaly** and
**788-overnight-intraday-tug-of-war** (which ask whether the overnight *level* is a
tradable return; we ask whether the *split itself* moved at a known institutional date),
**558-failures-to-deliver** (settlement *fails*, not the settlement *cycle*),
**605-vix-settlement-day** (a recurring monthly auction, not a one-off regime change), and
**89-turn-of-the-month** / **604-month-end-rebalancing-flows** (which measure the
turn-of-month effect itself; here it is only the venue for the DiD, traded as a
foreign-vs-domestic ETF spread).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what changed on 28 May 2024, why EFA "should" have felt it, the picture the tape actually shows, and the placebo test that settles it |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the DiD specification, HAC *t* and block-bootstrap CIs, the window-length sign flip, the placebo distribution, the costed overlay and the live synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`t_plus_one/`](t_plus_one/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
