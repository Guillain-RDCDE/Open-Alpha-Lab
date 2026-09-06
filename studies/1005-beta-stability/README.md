# Study 1005 — Beta Has a Half-Life 📉

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — how much of a stock's beta persists from one period to the next? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | Partly, and much less than the two decimal places on a risk report imply. Across 40 names over 16 non-overlapping 252-day windows, regressing each period's cross-section of betas on the previous period's gives a slope of **0.802** (±0.027, R² 0.60) — Blume's finding, reproduced. A deviation from the average beta therefore decays by half in **3.1 years**. But the more important number is what that instability is made of. A beta measured over 252 days carries a mean standard error of 0.088, so two consecutive estimates would differ even if the true beta never moved. Decomposing the variance of the observed period-to-period change: **28% is estimation error** and only the remainder is genuine movement — a true standard deviation of 0.227 against an apparent 0.268. Betas move much less than they appear to; the estimates move a lot. The confirmation is the portfolio test: ten-stock portfolios cut the mean standard error from 0.0885 to 0.0317 and the noise share from 28% to 24% — measure beta better and less of its instability survives. Note the slope does **not** confirm this (0.773 against 0.802), and that is a finding in itself: regressing a noisy measure on a noisy measure attenuates the slope by var(true)/[var(true)+var(noise)], and diversification lowers both terms at once. The Blume slope confounds persistence with measurement quality and should not be read as a stability statistic at all — the synthetic control makes the point unanswerable, since a *drifting* beta there produces a **higher** slope than a perfectly constant one. |
| **Tradability** — does shrinking it toward one actually beat the raw estimate out of sample? | ![Useful](https://img.shields.io/badge/Useful-2ea44f?style=flat-square) | Shrinkage helps, the textbook constant is close to right, and the honest benchmark is harder than it looks. Predicting each period's beta from the previous one, root-mean-square error came to **0.260** using the raw estimate, 0.253 with Blume's 0.66 shrinkage, 0.255 with Vasicek's precision weighting, and **0.393 simply assuming 1.0 for every name**. That last figure is the one worth pausing on: the raw estimate holds it off, which puts a floor under how much any beta model is really contributing. Searching the shrinkage weight directly gives an optimum of **0.80** — against Blume's 0.66, and against the persistence slope of 0.802 that theory says it should equal, a coincidence close enough to be reassuring about the whole framework. Practically: re-estimate no more often than the half-life justifies, shrink toward one, and stop quoting the second decimal. |

> **In one sentence:** Beta's persistence slope is 0.80 — a 3.1-year half-life — but 28% of the apparent instability is estimation error, and shrinking toward 1.0 at a weight of 0.80 beats the raw estimate out of sample.

## What we tested

Betas are printed to two decimal places on risk reports, fed into
cost-of-capital calculations and used to size hedges — almost always without a standard error
and never with a statement of how long they remain valid.

**Persistence.** Regressing each year's cross-section of betas on the previous year's reproduces
Blume (1971): the slope is well below one, giving a measurable half-life for a beta's deviation
from the average.

**How much of the instability is real** — the question usually skipped, and the one that decides
what to do. A beta estimated from 252 daily observations has a standard error, so two
consecutive estimates differ even if the true beta never moves. Decomposing the variance of the
observed change into 2·mean(se²) plus genuine movement shows a large share of the apparent
instability is measurement. The confirmation is direct: ten-stock portfolios, whose betas are
measured far more precisely, persist substantially better. A synthetic world with a **perfectly
constant** true beta calibrates the artefact, and a drifting one shows the machinery detects real
movement when there is some.

**Does shrinkage help?** Blume's constant shrinkage, Vasicek's precision-weighted version and a
do-nothing baseline of assuming 1.0 for every name are scored on out-of-sample error. That last
baseline is the honest floor and is harder to beat than expected. The fitted shrinkage weight is
then compared with the persistence slope, which theory says it should equal — a consistency
check on the whole framework rather than on any single number.
**Dedup:** distinct from **1010-correlation-matrix-stability** (the covariance matrix),
**1012-benchmark-choice-and-alpha** (which benchmark) and **240-low-beta-anomaly** (whether beta
is priced); the subject here is the estimate's shelf life.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | how much of the beta on a risk report is real and how long it stays true |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | Blume persistence regressions, a variance decomposition of the instability into noise and movement, portfolio confirmation, four forecast methods scored out of sample, and a constant-beta control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`betahalflife/`](betahalflife/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
