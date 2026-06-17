# Study 232 -- Mohanram G-score

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Hedge **-2.98%/yr**, HAC *t* = **-1.61** -- *inverted*. High-G stocks underperform low-G by 3%/yr on 19 years of S&P 500 survivors. No evidence of a positive predictive signal. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Sharpe **-0.35**, max drawdown **-47.9%**, hit rate 47%. Negative expected return; there is no alpha to trade. |
| **Can the G-score find growth stocks that deliver?** | ![BUSTED](https://img.shields.io/badge/BUSTED-8b949e?style=flat-square) | On large-cap survivors, the signal inverts: fundamentally stronger growth firms earned *less*, not more, than their weaker peers. |

> **In one sentence:** the Mohanram G-score, applied to 318 S&P 500 survivors over 2007--2025, produces a hedge of -3%/yr in the *wrong direction* -- high-fundamental-quality growth stocks underperform their low-quality peers -- consistent with large-cap growth stocks being systematically expensive regardless of their accounting quality.

## What we tested

Mohanram (2005) "Separating Winners from Losers among Low Book-to-Market Stocks":
the G-score packages eight accounting signals into a composite to identify growth/glamour
stocks with genuine fundamental support. We steelman the claim: annually sort the S&P 500
universe on the 8-signal G-score (built from EDGAR XBRL), go long the top quintile and
short the bottom quintile with a one-year reporting lag, and measure vs the equal-weight
universe. The panel runs 2007--2025 (19 years, 318 tickers). **G6/G7 substitution:**
revenue-growth and asset-turnover-growth replace R&D and advertising intensity (not in
desk cache); this is documented and may explain some divergence from the published paper.
The universe is survivorship-biased (current S&P 500 projected backwards), which we name.
A deterministic synthetic panel serves as the positive control.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the Mohanram claim in plain language, the synthetic positive control, the real EDGAR panel result, the honest verdict |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t-stat sweep, year-by-year hedge breakdown, equity curve and drawdown, component-level analysis, survivorship discussion |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`mohanram_g_score/`](mohanram_g_score/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
