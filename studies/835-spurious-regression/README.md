# Study 835 — Spurious Regression 🎭

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is there any real relation between the two series? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The two series are drawn **independent by construction**, so there is *nothing real to find*. Regressing one random walk on the other in **levels** rejects "no relation" in **85.0%** of 5,000 pairs (Wilson 95% CI [0.840, 0.860]) — a **17× oversized** test — with mean `|t|` **8.99** and mean R² **0.24**, all manufactured by the unit root. First-differencing collapses the rejection rate to **5.3%** (R² 0.004), and the same OLS on *stationary* series is correctly sized (**5.1%**), proving the inflation is nonstationarity, not OLS. A synthetic-only method demo — no real tape, so it can never earn `REAL`. |
| **Tradability** — can you trade the fake relation? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The spurious spread `y − βx` is *itself a random walk*, not mean-reverting. A costed, look-ahead-free pairs trade earns a gross edge **indistinguishable from zero** (*t* = **−1.23**, \|t\|<2) and loses net of any friction (net Sharpe −1.43 → −1.54 as cost rises). There is nothing to harvest. |
| **Do trending series manufacture false significance?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | *Yes.* Driftless walks reject **85%** of the time; add a shared trend and it jumps to **98%** with mean R² **0.66**. Worse, the spurious `|t|` scales with **√T**, so *more data makes it worse* (mean `|t|` **4.0 → 18.0** as n: 50 → 1000). Differencing and an Engle-Granger cointegration test (reject **0.05** on independent walks vs **1.00** on a genuinely cointegrated pair) both see through it. |

> **In one sentence:** regress one independent random walk on another and OLS will hand you a
> textbook "significant relation" — 85% of the time on driftless walks, 98% if they merely trend
> together — with a fat *t*-stat and a high R² that grow with the sample size, all pure artefact of
> nonstationarity; first-differencing (or a cointegration test) restores honest inference, and there
> is of course nothing there to trade.

## What we tested

Granger & Newbold (1974), **"Spurious Regressions in Econometrics"**: regress one **independent
random walk** on another and the ordinary-least-squares *t*-statistic rejects "no relation" far more
than 5% of the time, with an R² that can be high — purely because the two series are non-stationary
(each carries a unit root). We simulate **5,000 pairs of independent random walks (250 obs each, base
seed 835)**, run the textbook level OLS, and record the slope *t*-stat and R²; then we apply the two
cures — **first-differencing** (regress the changes) and an **Engle-Granger cointegration test** — plus
a **stationary-series size control** that isolates the pitfall to the unit root, a **sample-size sweep**
that shows the √T divergence, and a costed, look-ahead-free **pairs timer** for the tradability axis.
Everything is deterministic, offline, and synthetic-only (a method demo cannot certify "no relation"
from a real tape) — named on the **Signal** axis and capped at `None`. **Dedup:**
[346-multiple-testing](../346-multiple-testing/) inflates significance by testing **many** hypotheses
(selection), not by nonstationarity in a **single** regression; [348-curve-fitting](../348-curve-fitting/)
is about over-flexible models **memorising** noise, not a correctly specified OLS on unit-root data;
[343-data-mining-roulette](../343-data-mining-roulette/) mines a large signal space for lucky winners,
whereas here one regression on two independent walks is already grossly over-sized with **no** search.
As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | two coin-flip paths that have nothing to do with each other, why OLS calls them "related" anyway, why trending makes it worse, and why more data doesn't save you — in plain language, with a live mini-simulation |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the oversized level *t* and R², the differencing fix, the √T sample-size divergence, the stationary-series size control, the Engle-Granger cointegration positive control, and the costed pairs timer |

The fingerprinted headline run (simulation fp `73e2821b184c`, as-of 2026-06-30) is in
[docs/results.md](docs/results.md); sources & literature map in [docs/references.md](docs/references.md).
Reproduce every number with `python examples/verify.py`.

---

*Engine: [`spurious_regression/`](spurious_regression/). Deterministic seeded simulations — no network,
no real market data. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
