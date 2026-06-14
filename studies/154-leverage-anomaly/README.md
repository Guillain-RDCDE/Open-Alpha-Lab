# Study 154 — Leverage-Anomaly

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | LTD/Assets spread **−3.9%/yr** (*t* = −1.78, wrong direction); D/E spread **+3.1%/yr** (*t* = +1.41, right direction but below bar). The two measures disagree in sign; neither clears \|*t*\| ≥ 2. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | No confirmed edge to trade. Annual rebalance is cheap, but signal quality (hit rate ~29–53%, sub-2 t-stats) makes any implementation indistinguishable from a coin flip. |
| **Survivorship-biased** | ![upper_bound](https://img.shields.io/badge/Survivorship--biased-upper__bound-8b949e?style=flat-square) | S&P 500 current-membership panel projected back. Bankrupt and delisted high-leverage firms are excluded — the bias favours finding the anomaly, yet it is still absent. |

> **In one sentence:** the Penman-Richardson-Tuna (2007) leverage anomaly — low-leverage firms outperform high-leverage firms — does not replicate on this survivorship-biased S&P 500 EDGAR panel: the two simplest proxies (LTD/Assets and D/E) point in opposite directions and neither clears the inference bar.

## What we tested

The financial-leverage puzzle: standard Modigliani-Miller theory predicts more debt → more risk → higher expected equity returns. But Penman, Richardson & Tuna (2007) decompose the book-to-price ratio and find the *leverage* component is *negatively* priced — high-leverage firms underperform. We test the two simplest leverage proxies (LTD/Assets and Liabilities/StockholdersEquity) on the desk's shared EDGAR panel of S&P 500 companies (FY 2007–2024), using a conservative annual reporting lag, quintile sorts, and HAC t-stats. A random-portfolio control rules out concentrated-subset luck. The survivorship bias in the panel is named throughout and the results are treated as upper bounds.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the PRT claim in plain language, why the two measures disagree, what survivorship bias means for the conclusion |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-year HAC t-stats, random-portfolio null, measure-by-measure breakdown, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`leverage_anomaly/`](leverage_anomaly/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
