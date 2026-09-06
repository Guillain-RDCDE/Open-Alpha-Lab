# Study 967 — Window Shopping 🪟

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the window length actually change the estimate's accuracy? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | Yes, and by more than habit would suggest. Across estimation windows the MSE spread is **72%** for beta, **92%** for the mean return and **3%** for the volatility of the minimum-variance portfolio; the strongest pairwise Diebold-Mariano across the three experiments is **+3.45**. The three quantities do not agree with each other: beta wants **2**, the mean wants **10**, the covariance matrix wants **3**. |
| **Tradability** — is there one window a practitioner should default to? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | There is no single default. The mean is the extreme case: even the *best* window of a sector's own history is beaten by the **grand mean across all sectors** (MSE ratio 1.54), which is the empirical way of saying individual expected returns are not estimable from price history at all. Beta is estimable and mildly non-stationary — Blume shrinkage cuts its MSE by a further **-10%**. The covariance matrix is the one place where a short window is actively dangerous: at 11 assets, a one-year window has about 42.0 observations per estimated parameter. |

> **In one sentence:** How much history you should use depends entirely on what you are estimating: means want everything (and are hopeless anyway — the cross-sectional average beats a sector's own history), betas want a medium window plus shrinkage, and covariance matrices want enough rows per parameter to stay invertible — which a one-year window at 11 assets is not.

## What we tested

Every estimate a portfolio uses is computed from *some* window of history, and the
window is nearly always chosen by convention — five years because that is what the terminal
shows, or all of it because more data must be better. We test the choice out of sample on the
eleven Select Sector SPDRs, on three quantities that a portfolio actually needs: a **beta**, an
**expected return**, and a **covariance matrix**. At each year-end every parameter is estimated
from a rolling 1 / 2 / 3 / 5 / 10-year window and from an expanding window; each estimate is
then scored against what the following year delivered — the beta that materialised, the mean
that materialised, and for the covariance matrix the *realised volatility of the
minimum-variance portfolio it built*. Differences are tested with a HAC-corrected
Diebold-Mariano on paired squared errors, and two humbling benchmarks are added: **Blume
shrinkage** for beta, and the **cross-sectional grand mean** for expected returns.

The answer is not one number, and that is the finding: the three quantities want different
windows, for reasons that are bias-variance arithmetic rather than market lore.
**Dedup:** distinct from **966-har-vs-garch** (competing volatility *models*, not window
lengths), **975-covariance-shrinkage** (a better estimator for a fixed window, rather than the
window itself), **1008-beta-stability** (whether a stock's beta persists, measured
cross-sectionally on single names), **836-timing-luck** (*when* you rebalance, not how much
history you use) and **348-curve-fitting** / **968-bootstrap-choice** (fitting and inference
rather than estimation windows).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why 'use all the data' is right for one parameter and wrong for another, the estimate-then-wait experiment in pictures, and the humbling benchmark that beats every window |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the three out-of-sample experiments, paired-error Diebold-Mariano, Blume shrinkage, minimum-variance optimism versus observations-per-parameter, and the stationary-world synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`window_choice/`](window_choice/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
