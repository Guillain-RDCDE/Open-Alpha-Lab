# Study 118 -- Fed-Model

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | E/P alone: in-sample HAC *t* = **+4.98** at 10yr (120-lag NW, corrected for overlap), but OOS R² = **−0.003** -- essentially zero out of sample. Composite Fed-Model spread (E/P minus nominal yield): HAC *t* = **+1.78**, OOS R² = **−1.17** -- a catastrophic regime failure due to 1970s rate shock. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The 10-year horizon is untradable. The 1-year market-timing version returns +4.6%/yr vs +8.6%/yr buy-and-hold; active return **−4.0%/yr** (HAC *t* = −4.71). |
| **Beats buy-and-hold?** | ![No](https://img.shields.io/badge/No-8b949e?style=flat-square) | The spread is positive (stocks 'cheap') ~80% of the time; the strategy is almost always long, sits out of some bull markets when rates spike, and lags the index by 4%/yr. |

> **In one sentence:** the earnings yield carries a weak, regime-dependent long-run signal that dissolves out of sample -- adding the nominal bond yield (the classic Fed Model) makes it worse, not better, and no timing version beats passive buy-and-hold.

## What we tested

The "Fed Model" (Yardeni 1997): when the S&P 500 earnings yield (E/P = Earnings/SP500, from Shiller monthly data 1871-2023) exceeds the 10-year Treasury yield ("Long Interest Rate"), stocks are cheap relative to bonds and should outperform. We run predictive OLS regressions of 1-year and 10-year forward real total returns on the spread, with Newey-West HAC standard errors at 120 lags (the minimum needed for non-misleading inference with 10-year overlapping returns), compute out-of-sample R² (train 1871-1969, test 1970-2023), and evaluate a 1-year market-timing strategy vs buy-and-hold. We also test Asness's (2003) critique -- that comparing a real quantity (E/P) to a nominal yield is a unit mismatch -- by separately testing E/P vs the real rate (yield minus trailing CPI inflation).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the famous in-sample chart, the OOS reality check, the timing strategy vs buy-and-hold, why the model fails after 1970 |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t-stats with 120-lag NW correction, OOS R² (Campbell-Thompson 2008), the Asness real/nominal decomposition, the synthetic positive control |

Sources and literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`fed_model/`](fed_model/). **Not investment advice** -- research and education. See [LICENSE](../../LICENSE).*
