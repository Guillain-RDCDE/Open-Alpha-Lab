# Study 141 -- Turnover-Anomaly

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![Mixed](https://img.shields.io/badge/Mixed-dab617?style=flat-square) | Q1-market hedge **-3.7%/yr**, HAC *t* = **-3.94**; Q1-Q5 spread **-12.9%/yr**, *t* = **-4.75** -- signal is real but **directionally reversed** vs the published claim (Datar-Naik-Radcliffe 1998). |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The Datar trade (long low, short high) loses **-12.9%/yr** before costs; the reversal is driven by survivorship bias, not a live edge. |
| **Survivorship?** | ![Not_Supported](https://img.shields.io/badge/Not_Supported-8b949e?style=flat-square) | S&P 500 survivor-only panel: Q5 is populated by today's winners (NVDA, AMZN, NFLX) known only in hindsight. Positive results for high-turnover are upper bounds, not live signals. |

> **In one sentence:** the Datar-Naik-Radcliffe (1998) high-turnover-underperforms signal inverts on the current S&P 500 survivor universe -- high-turnover stocks win, not lose -- but this tells us about survivorship bias, not a tradable alpha.

## What we tested

A 1998 paper by Datar, Naik, and Radcliffe documented that stocks with high **share turnover**
(annual trading volume / shares outstanding) subsequently *underperform*, explaining it through
a liquidity premium (Amihud-Mendelson) and divergence-of-opinion overpayment (Miller 1977).
We test it mechanically on the current S&P 500: EDGAR diluted-shares-outstanding (10-K annual
filings) plus yfinance annual trading volume, sorted into five equal-count quintiles each year,
with next-year returns from the shared EDGAR return panel. A shuffled-label null (500 draws)
confirms the signal is doing the ranking. Signal years 2008-2024 (17 annual observations), one-
year lag, equal-weight quintile portfolios. The universe is **survivorship-biased** -- only firms
that survived into the current S&P 500 are included, making high-turnover results upper bounds.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what turnover predicts, why the signal reverses, the survivorship story in plain English |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC *t*-stats, shuffled-label null, quintile profile, survivorship anatomy, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`turnover_anomaly/`](turnover_anomaly/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
