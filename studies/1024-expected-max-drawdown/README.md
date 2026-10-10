# Study 1024 — The Drawdown You Were Promised 🤞

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the Gaussian drawdown promise mis-calibrated, ex ante? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | Written down from the preceding years only, the 1-year 95% drawdown band of a GBM was breached in **22%** of S&P 500 years (6/27, binomial p = 0.002), **15%** of Fama-French market years (12/81, p = 0.001) and 13% of Nasdaq years (2/15, p = 0.17) — **16%** pooled against a promised 5%. The machinery is calibrated on an i.i.d. Gaussian tape (5.8%), so the failure belongs to the tape — and it is a shape failure: the median year fell only **0.68×** the promised mean while the bad years overran the 95th percentile 3.3× too often. Twenty survivor stocks (a floor) breached in 12% of stock-years. |
| **Tradability** — is there a usable drawdown budget rule? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The pooled fix is **1.63×** the Gaussian 95th percentile, but the factor needed wandered from **1.03× to 1.92×** across tapes and eras (outside the pre-registered 1.5 ratio). Neither a stationary block bootstrap (15% breaches) nor a GARCH(1,1)-t from today's variance (14%) restored coverage on all three tapes. The best partial repair was a humbler mean — drift set to zero, 8% breaches — which then over-promises at 5–10 years. No return stream, so Investable is not on offer. |

> **In one sentence:** Fed the trailing mean and volatility, the Gaussian drawdown promise was broken in 16% of years instead of 5% — knowing mu and sigma is not knowing drawdown risk — and no fixed multiplier (it wandered from 1.0× to 1.9×) and no off-the-shelf model repairs it.

## What we tested

The claim, at full strength: a risk model that knows an asset's mean and volatility knows its
drawdown risk, because for a Brownian motion with drift the whole distribution of the maximum
drawdown over a horizon is solved ([Magdon-Ismail, Atiya, Pratap & Abu-Mostafa, 2004](docs/references.md)) —
and drawdown budgets and Calmar normalisations are built on *expected max DD ≈ f(μ, σ, T)*.
We simulate that promise **exactly** for a discretely-sampled GBM at each tape's own frequency
(sampling matters: a monthly mark hides about a fifth of the daily drawdown), write it down
before every complete calendar window from the preceding 5 years (10 on monthly data) only, and
count breaches of its 95% band with exact binomial tests on non-overlapping 1-, 3-, 5- and
10-year windows (moving-block bootstrap on overlapping ones) on the S&P 500 and Nasdaq **price**
indices, the Fama-French US market **total return** (1926–2018) and 20 **survivor** stocks — then
pit it against a stationary block bootstrap and a GARCH-t, and price the multiplier a risk
manager would need. A synthetic GARCH-t tape whose `signal_strength` dials clustering and fat
tails checks the machinery is calibrated at zero. **Dedup:** distinct from
**990-var-breach-count** (coverage of a one-day VaR, not of a multi-year, path-dependent
drawdown), **991-aggregational-gaussianity** (how fast the return distribution turns Gaussian,
not what that does to a risk budget), **813-maximum-drawdown-anomaly** and
**816-drawdown-duration** (drawdown depth and duration as cross-sectional return *signals*, not
as risk forecasts).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the promise drawn out, the S&P 500 year by year against it, why a safety factor does not travel |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | discretisation vs Magdon-Ismail, binomial and block-bootstrap coverage, PIT, the mean/shape/state decomposition, multiplier stability by era, survivors, synthetic calibration |

Reproducible headline run (as-of 2022-11-30, SHA-pinned bundled tapes): [docs/results.md](docs/results.md).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`quantlab/`](../../quantlab/) + [`promised/`](promised/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
