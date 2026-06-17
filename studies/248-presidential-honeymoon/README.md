# Study 248 — Presidential-Honeymoon

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Paired HAC t-stat = **−0.29** across 25 inaugurations (1929–2025); the honeymoon *underperforms* the control window by −1.34 pp on average. Post-2001 subsample: t = **−3.83** in the wrong direction. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Honeymoon buy-and-hold earns +3.29% raw vs +4.64% in the next window; negative before costs. No edge to extract even on paper. |
| **Does the market enjoy a honeymoon in a new president's first 100 days?** | ![BUSTED](https://img.shields.io/badge/BUSTED-8b949e?style=flat-square) | 44% win-rate (below a fair coin); the narrative has no empirical footing and reverses in modern data. |

> **In one sentence:** across 25 inaugurations since Hoover, the S&P 500's first-100-day 'honeymoon' return (+3.29%) is actually *lower* than the next 265 days (+4.64%), with a paired t-stat of −0.29 — the honeymoon is a myth.

## What we tested

The folk claim: markets enjoy a brief rally when a new president takes office, driven by
optimism, policy relief, or political goodwill.  We test it with a **within-term paired
design**: for each of the 25 inaugurations since Hoover (1929), we compare the S&P 500's
cumulative return over the first 100 calendar days (the honeymoon window) to the return
over the *next* 265 calendar days (the control).  Comparing honeymoon to a same-period
baseline removes trend and cycle effects; the paired design focuses the question sharply on
whether the inauguration timing itself generates excess returns.

This study is distinct from: [Study 81](../81-four-year-itch/) (year-3 cycle premium) and
[Study 159](../159-presidential-party/) (party effects).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim in plain language, the paired bar-chart, the modern-data reversal, why FDR 1933 (+78%) is an outlier not a signal |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-inauguration scatter, HAC t-stats, subsample breakdown, positive control with planted premium |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`presidential_honeymoon/`](presidential_honeymoon/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
