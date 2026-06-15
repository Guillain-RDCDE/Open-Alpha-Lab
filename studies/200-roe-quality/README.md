# Study 200 — ROE-Quality

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Hedge (high-ROE minus low-ROE) **−5.0%/yr**, HAC *t* = **−1.48** — wrong sign of the theory; high-ROE Q5 underperforms the EW market by **−1.0%/yr** (*t* = −0.94); survivorship-biased panel (upper bound). |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Negative-sign point estimate with no statistical foundation; annual rebalancing costs across ~50–70 names per leg further erode the already-negative return. Not investable. |
| **ROE is "priced"?** | ![Yes--already_in](https://img.shields.io/badge/ROE_priced%3F-Yes--already_in-8b949e?style=flat-square) | High-ROE names in the S&P 500 survivor universe have already been bid up; the quality premium lives in the spread vs. the broader market, not within the index. |

> **In one sentence:** ROE-as-quality fails inside the S&P 500 survivor universe — high-ROE firms are already priced for perfection and subsequently underperform low-ROE firms; the theory's signal lives in broader markets, not in the concentrated survivors.

## What we tested

One of the oldest investment intuitions: companies that consistently earn high returns on equity
(ROE = net income / book equity) are "quality" businesses and should reward investors with higher
forward returns. We build ROE from the EDGAR fundamentals cache (NetIncomeLoss / lagged
StockholdersEquity, winsorised 1%/99%), sort ~300 S&P 500 names into quintiles, and test whether
the top quintile (high ROE) earns higher forward 1-year returns than the bottom quintile (low ROE)
and the equal-weight market. We also run gross profitability (GP/Assets, Novy-Marx 2013) as a
head-to-head comparison, and a random-portfolio control. Reporting lag: fiscal year y fundamentals
→ calendar year y+1 returns. Window: 2009–2025 (17 years). Survivorship-biased (named).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the quality story in plain language, why ROE fails here, survivorship bias explained |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | quintile return tables, HAC t-stats, random-portfolio null, GP/Assets comparison, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`roe_quality/`](roe_quality/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
