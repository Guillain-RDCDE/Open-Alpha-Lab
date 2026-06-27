# Study 523 — Investment-To-Assets

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## The claim

> *Do the heaviest capital-spenders underperform the disciplined ones?*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Long-low-IA / short-high-IA hedge **+0.04%/yr**, HAC *t* = **+0.01**; placebo p = **0.47** (buried in the shuffle null); low-IA beats only **55%** of random draws. The low and high IA terciles earn near-identical +19.4%/yr. Literature supports the effect on a broad universe, not on survivor large-caps. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No gross edge to begin with; after 10 bps × turnover and a 50 bps short borrow the hedge is **−0.66%/yr net**. Nothing to trade. |
| **Capex vs total-asset-growth channel?** | ![Busted](https://img.shields.io/badge/Channel-Busted-8b949e?style=flat-square) | The capex/PP&E investment channel is no more present than the total-asset-growth channel of [Study 244](../244-asset-growth/) — both vanish on the survivor large-cap panel. |

> **In one sentence:** Titman-Wei-Xie (2004) found that heavy capital-investors underperform on the broad US cross-section, but on a survivorship-biased large-cap basket the capex anomaly is a flat zero (hedge +0.04%/yr, *t* = 0.01) that turns negative after costs — the high-investment survivors are productive expanders, not over-investing failures.

## What we tested

Titman, Wei & Xie (2004) argue that firms ploughing a large fraction of their asset base
back into capital expenditure earn low future returns — managers over-invest, the market
over-extrapolates, and the stock disappoints. This is the **capex/PP&E channel** of the
investment anomaly, distinct from the total-asset-growth channel of
[Study 244](../244-asset-growth/) and packaged at the factor level as Fama-French **CMA**.

We compute **IA = CapEx_t / Total_Assets_{t-1}** from SEC EDGAR companyfacts (a ~13-year
deep fundamentals history, far longer than yfinance statements expose), sort a fixed
large-cap survivor basket (~27 names/year) into terciles each fiscal year, lag the signal
by a conservative report lag (12-month forward return beginning 4 months after fiscal
year-end, entered one trading day later — exactly one execution lag, no look-ahead), and
test whether the low-IA tercile beats the high-IA tercile. A label-shuffle placebo, a
random-portfolio null, explicit costs + short borrow, and a seed-robust synthetic positive
control round out the inference.

The basket is **survivorship-biased**: it covers only firms still large-cap in 2026.
Critically, the high-IA group on a survivor basket skews toward *successful* capex-heavy
expanders — semiconductor fabs, hyperscale datacentres, energy majors — the opposite of
the over-investing failures the anomaly was built around. (Note 2024: the high-IA tercile
returned +76% as the AI-datacentre capex names ripped.)

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the investment-to-assets recipe in plain English, why the survivor basket erases the sign, year-by-year results |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | tercile monotonicity (absent), HAC t-stats, label-shuffle placebo, random-portfolio null, costs, seed-robust synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`investment_to_assets/`](investment_to_assets/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
