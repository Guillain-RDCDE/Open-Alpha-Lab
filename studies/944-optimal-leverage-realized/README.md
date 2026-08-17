# Study 944 — How Much Leverage ⚖️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the edge real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | The growth curve is real and concave, peaking at **2.85×** (Kelly says 3.10×) — but its *location* is unidentified: the block-bootstrap CI for the argmax is the **whole grid [1.00, 3.00]**, the 5-year hindsight optimum sits at the floor in 24% of windows and at the cap in 54%, the realised optimum moves **2.60 → 3.00 on the sample's start date alone**, and the tradable ex-ante Kelly rule beats 1× by **+6.86%/yr at HAC *t* = +1.20** (CI spans zero). No cap or estimation window reaches \|*t*\| = 2 (best +1.74). |
| **Tradability** — is it bankable? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Excess Sharpe is **invariant in L by construction** (0.5752 at 1× *and* 3× gross of financing) and only falls once the spread is paid; the one time-varying arm that *can* move it adds **+0.030, CI [−0.180, +0.254]**. The growth is bought with a **−94.5%** drawdown at the realised optimum, and the era hand-off swings from **+1.73%/yr to −4.32%/yr against not levering at all** depending on where the sample starts. |

> **In one sentence:** Constant leverage on the index has a genuinely concave growth curve that peaked at **2.85×** over 2003-2026, but that peak is a hindsight artefact — its bootstrap confidence interval covers the entire 1×-3× grid, the five-year rolling answer oscillates between "never lever" and "max out", and moving the sample's start date by seven months is enough to flip the era hand-off from beating buy-and-hold to losing to it — so "optimal leverage" is a number that exists only after the fact, and only for the sample you happen to hold.

## What we tested

The constant multiple **L = 1.0 → 3.0**, **reset daily**, on **SPY** total-return closes:
`r_L = L·r_SPY − (L−1)·(^IRX + spread) − cost·turnover`. Borrow is financed at the
**^IRX** bill rate (act/360) plus a **50 bps/yr spread (PROXY**, swept 0-200); the reset
pays **1 bp one-way × NAV traded** (PROXY, swept 0-5). Excess returns subtract the same
bill rate on both sides. We map terminal wealth, excess Sharpe and max drawdown against
the theoretical Kelly `mu/sigma²`, then attack the peak's *stability* four ways — a
63-day block bootstrap of the argmax, a rolling 5-year hindsight optimum, an era cut with
a hand-off test, and a **start-date sweep** — plus a tradable ex-ante Kelly arm carrying
the study's **single execution lag** (estimate through *t*, act at *t+1*), tested on
**log**-return differences because growth compounds. SPY∩^IRX **2003-06-04 → 2026-06-30**;
the window omits the 2000-02 bust and the verdict is conditional on that — section (4) of
[results](docs/results.md) prices that conditionality rather than merely confessing it.
**Dedup:** the nearest neighbour is
**[157-kelly-sizing](../157-kelly-sizing/)** (which tests the walk-forward Kelly *rule*,
Weak/Fragile); 944 maps the *object* that rule chases — the full realised leverage surface
and the confidence interval of its argmax — and its ex-ante arm is an independent
replication of 157 on a different (financed, daily-reset, costed) engine. Distinct from
**[61-slow-burn](../61-slow-burn/)** / **[100-melting-ice](../100-melting-ice/)** (the 3×
*instrument*'s decay mechanics), **[593-hfea](../593-hfea-leveraged-6040/)** /
**[594-leverage-rotation-200sma](../594-leverage-rotation-200sma/)** (levered *portfolios*
and levered *timing*), and **[590-sharpe-hacking](../590-sharpe-hacking/)** (which shows
leverage cannot move Sharpe — here we measure that invariance exactly and rule the axis
out before the growth analysis begins).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why there *is* a best leverage, the peak nobody can find, the answer that changes when you change where you start counting, the drawdown you have to survive |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the g(L) curve vs Kelly, Sharpe invariance, the bootstrapped argmax CI, rolling optimum instability, the start-date sweep, the ex-ante race with log-growth HAC *t*, spread/cost/cap sweeps, the live synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`optimal_leverage/`](optimal_leverage/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
