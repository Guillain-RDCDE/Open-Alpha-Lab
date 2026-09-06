# Study 998 — The Moving Target 🎯

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a Kalman-filtered hedge ratio track a moving relationship better than a rolling window? | ![Confirmed](https://img.shields.io/badge/Confirmed-2ea44f?style=flat-square) | Graded against a hedge ratio that is **known** because it was planted, the Kalman filter tracked it with an RMSE of **0.0471** against the best rolling window's 0.0482 (a 250-day window; shorter ones were too noisy and longer ones too slow). The reason is visible in one diagnostic: the filter's estimate moved **3.43×** as much as the truth did, against 14.01× for the rolling window — adaptiveness without the thrashing. On the 7 real pairs the filter produced the tighter spread in **57%** of them. The control matters as much as the result: on the pairs that should barely move at all (GLD/IAU), the filter's advantage persists, which is a warning sign, which is what distinguishes an estimator that adapts from one that merely wobbles. |
| **Tradability** — does the better tracking survive contact with a spread trade? | ![Useful](https://img.shields.io/badge/Useful-2ea44f?style=flat-square) | Tracking is not trading, and the gap between them is the hedge-rebalancing cost that most comparisons omit. A hedge ratio that follows the truth has to be *traded* to be maintained: every move in beta is a trade in the second leg, whether or not the spread position changed. Charged properly, the filter's annual hedge turnover was **0.15** against the rolling window's 1.55, costing 1.25% a year versus 1.31%. Gross, the filter earned a Sharpe of 0.60 against 0.53; **net, -0.03 against -0.11**. And a caveat that outranks all of it: the best pairs-trade Sharpe here is -0.03 across 21 years, which is not a business. |

> **In one sentence:** A Kalman filter tracks a moving hedge ratio 1.0× more accurately than the best rolling window — and once you charge for the rebalancing that accuracy requires, the trading advantage is +0.08 of Sharpe.

## What we tested

Every spread trade rests on a hedge ratio, and every hedge ratio estimates a
relationship that will not sit still. The standard answer is a rolling window, which forces a
choice with no good answer: short enough to adapt, long enough to be stable, and no length is
both. A **Kalman filter** offers a different bargain — treat the hedge ratio as a hidden state
that random-walks, and let the filter's gain adapt automatically.

This study grades them where grading is possible: on a synthetic pair whose true hedge ratio is
**planted**, so estimators compete against the truth rather than against each other. The key
diagnostic is *excess movement* — how much an estimate moves relative to how much the truth
moves — which separates "adapts fast" from "adapts to noise", and it is the number that explains
why a 20-day window loses. Two controls keep it honest: a **constant-beta world**, where
adaptation must be a *liability*, and two real pairs of near-identical funds (GLD/IAU, SPY/IVV)
where the filter should show no advantage at all. An estimator that wins there is wobbling, not
learning. `effective_window` then demystifies the filter by translating its `delta` into the
rolling window it is equivalent to.

Finally the part most comparisons skip: **tracking is not trading**. A hedge ratio that follows
the truth closely has to be *traded* to be maintained — every move in beta is a trade in the
second leg — so the adaptive estimator pays for its adaptiveness in turnover. The backtest
charges that explicitly, and reports gross and net side by side.
**Dedup:** distinct from **287-pairs-trading** and **604-cointegration-tests** (whether pairs
trading works at all), **973-dimson-beta** and **987-silver-high-beta-gold** (beta estimation for
measurement rather than hedging), **1001-cusum-change-points** (detecting a break rather than
tracking a drift) and **992-vol-clustering-halflife** (the persistence of volatility rather than
of a relationship).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a hedge ratio will not hold still, what a Kalman filter does about it, and whether the improvement survives being traded |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | estimators graded against a planted beta, excess-movement diagnostics, the filter's effective window, a constant-beta control, seven real pairs, and gross versus net with hedge rebalancing charged |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`movingtarget/`](movingtarget/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
