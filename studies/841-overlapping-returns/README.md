# Study 841 — Overlapping-Returns Inflation 🔗

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is there a real edge to find? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | On a world we *built* with **zero predictability** (`beta = 0`), a monthly regression of the next-12-month return on a driftless predictor prints a naive **t = +4.84** and R² **3.9%** — a "5-sigma discovery" of an edge that does not exist. Synthetic-only method demo, so never `REAL`. |
| **Tradability** — could you trade it? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | An inflated t-stat and R² are statistical illusions — the apparent forecast has zero out-of-sample value and vanishes the moment you use a correct standard error. Nothing to harvest. |
| **Does overlapping returns inflate inference?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | The naive 5% test's rejection rate under the null soars **6% → 66%** as `h` goes 1 → 24 months, and mean R² climbs **0.2% → 3.1%**; the **Hodrick 1B** standard error stays well-sized (~6%) at every horizon, Newey-West only partly (→19%). The control shows the fix still detects a *real* edge. |

> **In one sentence:** a long-horizon predictive regression looks compelling because monthly-sampled `h`-month returns *overlap* — adjacent observations share `h−1` months — which induces an MA(h−1) autocorrelation that the naive OLS standard error ignores, manufacturing t-stats above 6 and a 3% R² out of a driftless predictor, exactly the trap the Hodrick (1992) standard error was invented to defeat.

## What we tested

Since the 1980s the case for return predictability leaned on **long-horizon predictive regressions**: forecast the cumulative return over the next 1–10 years from a valuation ratio, sampled monthly. Because the overlapping `h`-period returns share `h−1` months, the residuals are a moving average of order `h−1`, and OLS standard errors that assume no serial correlation are badly understated. We make the trap undeniable by running the regression on a synthetic world *built* to have **no predictability at all** (Stambaugh-form persistent predictor, `beta = 0`), then measure the rejection rate of the naive 5% test as the horizon grows: it explodes from an honest 6% at `h = 1` to **66% at `h = 24`**, with R² inflating alongside. We put two corrections against it — **Newey-West** HAC (`lags = h−1`) and the **Hodrick (1992) "1B"** estimator (which moves the summation onto the regressor for non-overlapping moments) — and a **positive control** with a genuinely planted edge proves the corrections still *detect real predictability*, they don't merely tame the null. **Dedup:** distinct from [838 HAC-Necessity](../838-hac-necessity/) (HAC on a *strategy's own daily P&L*, not a predictive regression), [835 Spurious-Regression](../835-spurious-regression/) (independent *unit-root/trending* series, a non-stationarity mechanism), and [346 Multiple-Testing](../346-multiple-testing/) (false significance from *many hypotheses*, a trial-count fix) — here a **single** stationary regression is inflated purely by the **overlap** of cumulative returns.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why adjacent 12-month returns "share" 11 months, how that fakes a stunning long-horizon forecaster from nothing, and the fix — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the MA(h−1) residual structure, naive vs Newey-West vs Hodrick 1B standard errors, the Monte-Carlo size (under the null) and R² inflation vs horizon, and the seed-robust positive control (power) |

The fingerprinted headline run (null-world fp `4111f0ae3f09`, as-of 2026-06-30) is in [docs/results.md](docs/results.md); the whole machinery runs offline and deterministic on the synthetic world in [`overlapping_returns/data.py`](overlapping_returns/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`overlapping_returns/`](overlapping_returns/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
