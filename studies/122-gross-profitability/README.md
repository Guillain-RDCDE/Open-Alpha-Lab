# Study 122 -- Gross-Profitability

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Hedge **+3.28%/yr**, HAC *t* = **+0.80** (|*t*| < 2). Literature prior (Novy-Marx 2013, Fama-French RMW) prevents a `NONE`, but real-tape numbers on 18 years of large-cap survivors do not independently clear the bar. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | Sharpe **+0.20**, max drawdown **-45.8%**, reversals of -25% in both 2024 and 2025. Annual rebalance is cheap but the signal is too noisy on S&P 500 alone. |
| **Survivorship-biased?** | ![Named](https://img.shields.io/badge/Named-8b949e?style=flat-square) | Universe = current S&P 500 projected backwards (213 tickers with both GrossProfit and Assets). Positive results are **upper-bound estimates**. |

> **In one sentence:** gross profitability (GP/A) shows a faint positive spread on the S&P 500 survivor panel -- positive mean, below-the-bar t-stat, and sharp reversals in the most recent two years -- consistent with the known quality premium being real in broad universes but attenuated to noise in large-cap survivors post-publication.

## What we tested

Novy-Marx (2013) "The Other Side of Value": GrossProfit / Assets (GP/A) predicts the cross-section of stock returns as reliably as book-to-market. We steelman the claim: annually sort the S&P 500 universe on GP/A (from EDGAR XBRL filings), go long the top quintile and short the bottom quintile with a one-year reporting lag, and measure the hedge against the equal-weight universe. The panel runs from 2008 to 2025 (18 years, 213 tickers with both fields present). The universe is survivorship-biased (current S&P 500 projected backwards), which we name explicitly and treat as giving upper-bound estimates. A deterministic synthetic panel with a tunable premium serves as the positive control.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the Novy-Marx claim in plain language, the synthetic positive control, the real EDGAR panel leg vs market chart, the honest verdict |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t-stat sweep, year-by-year hedge breakdown, equity curve and drawdown, turnover profile, survivorship discussion |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`gross_profitability/`](gross_profitability/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
