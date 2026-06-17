# Study 282 -- NYC-Temperature

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Cold-minus-warm spread **-2.3 bps/day**, Newey-West HAC t = **-1.18**; OLS slope HAC t = **+1.09** with R-squared ~ **0**; permutation p = **0.23**. Nothing clears \|t\| >= 2, and the spread even points the wrong way for the folklore. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | A cold-long/warm-short overlay loses money gross *and* net (net Sharpe **-0.19** at 1 bp one-way cost), churning **~55%** turnover -- you pay to harvest noise. |
| **Busted?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The trader-mood mechanism is real psychology but ~50x below the daily equity noise floor (~100 bps/day vol); it does not survive a HAC t-test. |

> **In one sentence:** the temperature on Wall Street does not move the tape -- the cold-vs-warm return spread is a couple of basis points pointing the wrong way, statistically indistinguishable from zero, and untradeable after costs.

## What we tested

The weather-mood / "Good Day Sunshine" folklore (Saunders 1993; Hirshleifer &
Shumway 2003): the local temperature at the exchange supposedly tilts the day's
equity returns. We operationalize "the temperature on Wall Street" as the NYC
daily temperature **anomaly** (today's temperature minus its day-of-year seasonal
normal -- so the calendar cannot masquerade as a weather signal), hardcoded from a
curated 1962-2025 Central Park monthly table in `data.py`, and pin it against
^GSPC daily *price* returns. We sort cold vs warm terciles, regress returns on the
standardized anomaly, run a permutation test, and -- because weather is heavily
autocorrelated -- carry a **Newey-West HAC** standard error throughout. A synthetic
positive control confirms the machinery detects a planted cold-day premium (HAC
t > 7); the real tape has nothing like it.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, the seasonal-cycle trap, the cold-vs-warm sort, the untradeable overlay in plain English |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | tercile sort with HAC t, OLS sensitivity slope, permutation distribution, net-of-cost overlay, the synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`nyc_temperature/`](nyc_temperature/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
