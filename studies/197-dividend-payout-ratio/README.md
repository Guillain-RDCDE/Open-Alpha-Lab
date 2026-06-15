# Study 197 -- Dividend-Payout-Ratio

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Payout -> 10y real earnings growth: slope +0.064, HAC t = +3.49 (120 lags), R^2 = 8.1%; survives IS/OOS split and non-overlapping annual test. Payout -> 10y real *returns*: slope -0.031, t = -1.15 (wrong sign). |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | 10-year horizon, no price-return signal (t = -1.15), 142 non-overlapping observations in 140 years -- not a timing strategy. |
| **Horizon** | ![10-year](https://img.shields.io/badge/Horizon-10--year-8b949e?style=flat-square) | Forward earnings growth materialises over a full decade; no sub-10-year signal found. |

> **In one sentence:** Arnott & Asness (2003) are right that a higher aggregate payout ratio predicts higher subsequent real earnings growth -- the signal is real, modest (R^2 = 8%), and survives out of sample -- but it does not predict higher price returns, so no timing strategy exists.

## What we tested

The Arnott & Asness (2003) counter-intuitive claim: in aggregate, a *higher* S&P 500 dividend
payout ratio (annual dividends / annual earnings) predicts *higher* subsequent 10-year real
earnings growth -- the opposite of the textbook Gordon-Growth intuition that reinvested earnings
fund growth. Low payout signals empire-building at sub-market rates; high payout signals
disciplined capital allocation. We use the Shiller monthly S&P 500 dataset (1882-2013, N = 1,699
monthly obs) with HAC OLS (Newey-West, 120 lags), an IS/OOS split at 1953, and a non-overlapping
annual robustness check, against an unconditional buy-and-hold baseline.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the counter-narrative in plain language, quintile bar charts, why the earnings finding doesn't help you trade |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC OLS with 120 lags, IS/OOS slopes, non-overlapping annual robustness, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`dividend_payout_ratio/`](dividend_payout_ratio/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
