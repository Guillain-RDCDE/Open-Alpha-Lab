# Study 212 — Cannabis Stocks

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Beta=0.95× (t=7.57) is market-like, not disruptive; alpha=−40.9%/yr (t=−1.53, not significant) but economically devastating; R²=0.05 — 95% idiosyncratic noise. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Every cannabis ticker negative CAGR (−11% to −59%/yr); drawdowns −89% to −99.9%; vs SPY +15%/yr and Sharpe +0.83; timing rule Sharpe +0.063 (t=+0.18). |
| **Did the green-rush ever pay?** | ![Busted](https://img.shields.io/badge/Green--rush_paid%3F-Busted-8b949e?style=flat-square) | All five names (MSOS/MJ/TLRY/CGC/CRON) destroyed capital over every tested window; max drawdowns touch −99.9%. |

> **In one sentence:** Cannabis stocks (MSOS, MJ, TLRY, CGC, CRON) have been one-way wealth shredders since their US listings — catastrophic negative CAGRs (−11% to −59%/yr), drawdowns to near-zero, and a sector beta of ~1× to SPY with essentially no systematic component (R²=0.05); a 200-day momentum timing rule (Sharpe +0.063) fails to rescue the economics vs holding the index (Sharpe +0.83).

## What we tested

> *Did the green-rush ever pay, or is cannabis a one-way wealth shredder?*

The pitch: cannabis is a rapidly legalising, high-growth disruptive sector. Thematic ETFs (MSOS, MJ)
and leading names (TLRY, CGC, CRON) let investors capture regulatory tailwinds before Wall Street prices
them in. We test three precise claims against real data:

1. **Does cannabis carry alpha vs SPY?** OLS of MSOS daily log-returns on SPY (the beta/alpha test).
2. **Did any cannabis name or ETF beat SPY?** Buy-and-hold table across every tested ticker,
   over both the MSOS-era (2020-) and MJ-era (2017-) windows.
3. **Can a momentum overlay rescue it?** MSOS 200-day SMA signal vs SPY buy-and-hold.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the green-rush narrative, the real drawdown chart, why timing barely helps |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | OLS beta/alpha with HAC inference, multi-ticker performance table, timing HAC t-stat, positive synthetic controls |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`cannabis_stocks/`](cannabis_stocks/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
