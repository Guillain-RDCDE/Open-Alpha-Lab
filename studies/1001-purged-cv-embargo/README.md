# Study 1001 — The Leaky Fold 🧪

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — how much does ordinary k-fold cross-validation overstate a time-series model? | ![Confirmed](https://img.shields.io/badge/Confirmed-2ea44f?style=flat-square) | On SPY with a 20-day forward label, ordinary shuffled five-fold cross-validation reported an information coefficient of **+0.078** (R² +0.005). Walk-forward validation — the only scheme that resembles how a model is actually used — reported **-0.021**. The gap is **+0.099**, and it splits into two quite different causes. Letting the model see the future at all (shuffled minus sequential) is worth +0.052. **Label overlap** — neighbouring observations sharing 95% of their label days, with no date out of order whatsoever — is worth -0.002, which is **-2% of the total illusion**. That second channel is invisible to the usual advice about not shuffling time series, and at long label horizons it is the bigger one: the sweep shows it growing from +0.000 at a one-day label to +0.005 at 120 days. |
| **Tradability** — does purging and embargoing recover an honest estimate? | ![Partial](https://img.shields.io/badge/Partial-dab617?style=flat-square) | The fix works and it is cheap. Purging training observations whose label windows touch the test block brought the estimate to +0.028; adding a 1% embargo brought it to **+0.029** against walk-forward's -0.021 — a residual difference of 0.0501. And it does not achieve that by destroying everything: on synthetic data with a **planted** information coefficient of 1.00, purged-and-embargoed cross-validation recovered +0.855, so it still finds real signal. The cost is data: purging removed 0% of the training set at this horizon, which is the real reason people skip it. |

> **In one sentence:** Shuffled cross-validation flatters this model by +0.099 of information coefficient, -2% of it from overlapping labels rather than from shuffling — and purging with an embargo recovers the honest number to within 0.0501.

## What we tested

Cross-validation is the default way to estimate how a model will do on unseen
data. On time series it does not work, and this study measures the size of the illusion by
running an **identical** ridge model under five validation schemes where only the scheme differs.

Two distinct leaks are separated, which most treatments do not do. **Temporal leakage** — shuffled
folds putting the future in the training set — is the one everyone knows, and splitting by time
fixes it. **Label overlap** is the subtle one: predicting the next 20 days means observation *t*
and *t+1* share 19 of their 20 label days, so a training point beside the test block is nearly a
test point *with every date in the correct order*. The usual advice, "don't shuffle time
series", does nothing about it — and at long label horizons it is the larger of the two, which
the horizon sweep shows directly.

The fix is López de Prado's **purging** (drop training observations whose label window touches
the test block) and **embargo** (drop a gap after it, because autocorrelated features make
nearby points informative). Two checks keep it honest: on synthetic data with **zero**
predictability the honest schemes must report zero, and on data with a **planted** relationship
the purged scheme must still find it — a fix that destroyed real signal along with the leakage
would be no fix at all. The cost is reported too: purging removes a substantial share of the
training set, which is the real reason people skip it.
**Dedup:** distinct from **996-palindrome-dates** (multiple testing across hypotheses),
**860-backtest-overfitting** (parameter optimisation), **997-rebalance-timing-luck**
(implementation-date noise) and **554-walk-forward-optimisation** (a backtesting protocol rather
than a measurement of what CV overstates and why).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | how a model with no predictive power passes cross-validation, and the two separate reasons why |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | five validation schemes on identical data, a two-way leakage decomposition, the horizon sweep that shows overlap dominating, purging cost, and a planted-signal check that the fix does not overcorrect |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`leakyfold/`](leakyfold/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
