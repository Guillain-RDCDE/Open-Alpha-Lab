# Study 1011 — The Half-Life of an Edge ⏱

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — how fast does a signal's predictive content actually decay? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Yes, and the differences are large. Measuring the rank information coefficient at every horizon from one day to a year across 50 names, the fitted half-lives ran from **57 days** (reversal_5d) to **114 days** (momentum_12m) — a factor of 2.0 between signals whose headline ICs differ by far less. That is the number that should appear beside a backtest and almost never does. Two measurement points are worth making. The fit is to the **marginal** IC, not the cumulative one: a cumulative IC can keep rising simply because the horizon is longer, which flatters every decay profile. And the half-life is estimated with real uncertainty — bootstrapping whole years of the panel puts a 90% interval of **8 to 566 days** on the headline signal, a ratio of 70.0×. Any formula that turns a half-life into a trading rate inherits that interval, which is a good reason not to tune one to three decimal places. |
| **Tradability** — does trading at the rate the decay implies beat trading at any other rate? | ![Useful](https://img.shields.io/badge/Useful-2ea44f?style=flat-square) | Trading at roughly the decay rate wins, and the two ways of getting there agree. Sweeping the rebalancing period on reversal_5d at 10bp, the best information ratio came at **42 days** against a fitted half-life of 57 — 0.73× the half-life. Rebalancing 1× faster gave -1.75 and 126× slower gave -0.37, against -0.18 at the optimum. The Gârleanu-Pedersen partial-trading rule reaches the same place from the other direction: its closed form recommends trading 42% of the way each period, and the brute-force sweep put the optimum at 20%. Grinold's law is the piece that needs handling with care. Counting bets naively — 50 names × 252 days — gives a breadth of 12,600 and predicts an IR of -0.27. Correcting for the decay rate *and* for the -0.02 residual correlation between names brings breadth to 221 and the prediction to -0.04, against a realised -0.34. The naive count overstates breadth by **57×**, and since IR scales with its square root, that is a 8× exaggeration of the achievable information ratio. |

> **In one sentence:** Signal half-lives here span 57 to 114 days, the best rebalancing period lands within a factor of 1.4 of the half-life, and counting bets naively overstates breadth by 57×.

## What we tested

Two things are always reported about a signal — how strong it is and what it
returned. A third decides whether either matters: **how fast it decays**. This study measures
decay directly and tests what the theory says to do about it.

**Decay, fitted to the right thing.** The rank IC is measured at horizons from one day to a
year, and the half-life is fitted to the **marginal** profile rather than the cumulative one — a
cumulative IC can keep climbing simply because the horizon is longer, which makes every signal
look more durable than it is. The estimator is calibrated first on synthetic signals with
*known* half-lives.

**Breadth, counted properly.** Grinold's IR ≈ IC × √BR is routinely applied by counting trades.
A signal with a 60-day half-life does not deliver 252 independent views a year, and names whose
*residual* returns are correlated do not deliver N independent bets either. Applying both
corrections shrinks breadth by a large factor, and since IR scales with its square root, the
naive count exaggerates achievable performance by roughly the gap between what factor backtests
promise and what factor funds deliver.

**Trading rate, checked two ways.** The Gârleanu-Pedersen closed form for partial trading is
compared against a brute-force sweep, and the rebalancing period against the fitted half-life.

**The study's own contribution is the uncertainty.** The trading rate is a smooth function of the
decay rate, which invites precision. Bootstrapping whole years of the panel puts a wide interval
on the half-life itself — so the formula cannot deserve more precision than its input, and
tuning it finely is fitting noise.
**Dedup:** distinct from **1001-purged-cv-embargo** (validation), **997-rebalance-timing-luck**
(implementation-date noise) and **860-backtest-overfitting** (parameter search); the subject here
is decay and the trading rate it implies.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a signal's expiry date matters more than its strength, and how to find it |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | IC decay profiles with a calibrated estimator, marginal versus cumulative fits, bootstrapped half-life intervals, breadth corrected for decay and residual correlation, and the trading rate checked against brute force |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`halflife/`](halflife/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
