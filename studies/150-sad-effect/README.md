# Study 150 — SAD-Effect

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Onset mean +53.8 bps (t = -1.31 vs unconditional); recovery mean +94.1 bps (t = +0.68); daylight regression slope t = +0.16. No test clears \|t\| >= 2 over the full 155-year Shiller tape. In the KKL original sample (1950-2002) recovery reaches t = 1.83 — below the bar, and reversed post-publication (t = -0.85 in 2003-2026). |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The SAD calendar strategy (hold Dec-Mar, cash otherwise) achieves CAGR **3.6%** and Sharpe **0.55** vs buy-and-hold's **9.4%** CAGR and **0.71** Sharpe. Missing 8 months of equity risk premium per year is the price of a seasonal that isn't there. |
| **Data-mining critique confirmed?** | ![Yes](https://img.shields.io/badge/Data--mining%3A_Yes-8b949e?style=flat-square) | Absent pre-KKL (1871-1949), marginal in-sample (1950-2002), reversed post-publication (2003-2026) — the classic anomaly-decay signature documented by Kelly-Meschke (2010). |

> **In one sentence:** the SAD-Effect's seasonal daylight-driven return cycle is a data-mining artifact — statistically invisible over 155 years of S&P 500 history, barely present in KKL's original sample, and gone post-publication.

## What we tested

Kamstra, Kramer & Levi (AER 2003) proposed that Seasonal Affective Disorder — a clinical depressive
condition induced by shrinking daylight in autumn — raises investor risk aversion from September through
November, depressing equity returns, and that the reversal of daylight growth from December through March
creates a matching recovery. We test this against the Shiller S&P 500 monthly total-return series (1871-2026):
(1) the seasonal split of mean returns by SAD-onset (Sep-Nov) vs recovery (Dec-Mar) vs summer (Apr-Aug),
each vs the unconditional mean; (2) an OLS regression of monthly returns on the month-over-month change in
astronomical day-length (the KKL specification); (3) a SAD calendar strategy (hold equities only Dec-Mar,
cash otherwise) vs buy-and-hold. Following Kelly & Meschke (2010) we also break by sub-period to test
for data-mining. A deterministic synthetic tape with a tunable SAD premium serves as the positive control.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what SAD is, the seasonal pattern in plain English, why the strategy misses 8 months of the equity premium, the data-mining fingerprint |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | Welch t-stats by month group, Newey-West daylight regression, sub-period breakdown, Sharpe CI, vs buy-and-hold |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`sad_effect/`](sad_effect/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
