# Study 174 — Bitcoin-Rainbow

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Walk-forward HAC *t* = **+0.39**; in-sample *t* = +4.53 is +4.14 points of look-ahead bias. The bands are drawn after the prices they appear to predict. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Walk-forward **+2.2%/yr** vs buy-and-hold **+33.7%/yr**; strategy flat 80% of the time, missing Bitcoin's dominant trend. |
| **Overfitting?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The rainbow is a log-time OLS regression fitted on the whole history — a textbook in-sample curve-fit that always looks right in retrospect and always fails forward. |

> **In one sentence:** the Bitcoin Rainbow Chart is a log-time regression retrofitted through a decade of price history; strip out the look-ahead and the t-stat collapses from +4.53 to +0.39, while simply holding Bitcoin returns +34%/yr and demolishes the strategy.

## What we tested

The Bitcoin Rainbow Chart (Blockchaincenter.net, ~2014) fits `log(price) = alpha + beta * log(days since genesis)` on the **entire** available history and places nine coloured bands ("Fire sale" to "Maximum bubble territory") at fixed sigma offsets from the regression line. The recipe: buy in the cold bands, sell in the hot bands. We implement both the standard **in-sample** version (look-ahead-biased, the chart as commonly shown) and an **expanding-window walk-forward** version (honest: only uses data available at each date). We show that (1) the gorgeous in-sample t-stat is entirely an artefact of fitting the regression on the same data it evaluates, and (2) the walk-forward signal gives t = +0.39 — noise — while buy-and-hold on BTC earns +34%/yr over the same period. The strategy's fundamental flaw is that it keeps the investor flat 80% of the time, waiting for "extreme" bands that were defined using future prices.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the rainbow chart explained, why it *looks* prescient, the walk-forward reveal, cumulative return vs buy-and-hold |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t-stat dissection (IS vs WF), band sigma distribution, synthetic positive control (signal_strength sweep), cost sweep |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`bitcoin_rainbow/`](bitcoin_rainbow/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
