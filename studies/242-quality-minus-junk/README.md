# Study 242 -- Quality-Minus-Junk

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | Hedge **+2.71%/yr**, HAC *t* = **+0.91** (|*t*| < 2). Literature prior (Asness et al. 2019) prevents a `NONE`, but real-tape numbers on 16 years of large-cap survivors do not independently clear the bar. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | Sharpe **+0.18**, hit rate **50%**, max drawdown **−26%**. Recent reversals in 2024 (−9.5%) and 2025 (−17.1%) and a narrow spread on large-cap survivors make this structurally fragile. |
| **Does Asness's QMJ survive outside the backtest?** | ![Mixed](https://img.shields.io/badge/Mixed-8b949e?style=flat-square) | Real in broad universes (AQR evidence, FF RMW factor); attenuated to statistical noise in our 16-year large-cap survivor panel. The literature is stronger than the tape. |

> **In one sentence:** the QMJ composite (profitability + growth + safety) shows a faint positive spread on the S&P 500 survivor panel -- positive mean, below-the-bar t-stat, and two consecutive reversal years -- consistent with the known quality premium being real in broad universes but crowded and attenuated to noise in large-cap survivors post-publication.

## What we tested

Asness, Frazzini & Pedersen (2019) "Quality Minus Junk": a composite score of profitability
(GP/A, ROA, net-margin, CFO/Assets), growth (year-over-year change in profitability) and
safety (equity-to-assets) predicts the cross-section of stock returns with a reported
Sharpe ~1.3 on the US market (1957–2016). We steelman the claim: annually sort the S&P 500
universe on our three-pillar QMJ composite (Payout excluded — no share-count/dividend cache),
go long the top quintile (quality) and short the bottom quintile (junk) with a one-year
reporting lag, and measure the hedge against the equal-weight universe. The panel runs from
2010 to 2025 (16 years, 411 tickers). The universe is survivorship-biased (current S&P 500
projected backwards), which we name explicitly and treat as giving upper-bound estimates. A
deterministic synthetic panel with a tunable premium serves as the positive control.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the QMJ claim in plain language, the synthetic positive control, the real EDGAR panel quality vs junk chart, the honest verdict |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t-stat sweep, pillar decomposition, year-by-year hedge breakdown, equity curve and drawdown, turnover and cost discussion |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`quality_minus_junk/`](quality_minus_junk/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
