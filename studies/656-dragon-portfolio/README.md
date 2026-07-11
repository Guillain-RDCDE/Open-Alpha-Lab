# Study 656 — Dragon-Portfolio 🐉

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the 5-sleeve mix genuinely diversify both regimes? | ![Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square) | Ex-vol, better Sharpe than 60/40 (**0.668 vs 0.601**, smaller drawdown) but **uncertified** (HAC *t* = −0.37, bootstrap CI crosses zero). Add the real long-vol sleeve and the aggregate Sharpe over the only testable window (2018-2026) turns **negative** (−0.148), even though that same sleeve paid off **+33 pp** in the 2020 crash and cushioned 2022. |
| **Tradability** — can you build it off the shelf? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | Cheap, liquid ETFs, trivial cost drag (≈3 bps/yr from 0→10 bps). But the *published* 5-sleeve weights **lost money outright** (CAGR **−1.28%/yr**) over 8+ years — the long-vol "insurance" hasn't repaid its premium yet on this tape. |
| **"A cheap ETF Dragon just works?"** | ![Busted](https://img.shields.io/badge/A_cheap_ETF_Dragon%3F-Busted-8b949e?style=flat-square) | The diversification kernel (equity+bonds+gold+trend) genuinely helped in both 2020 (deflationary) and 2022 (inflationary) — that part holds up. But VXX, the only cheaply-tradable long-vol proxy, bled **−98.8%** of its value over the sample, turning the full prescribed portfolio Sharpe-negative despite two crisis wins. |

> **In one sentence:** Chris Cole's Dragon Portfolio genuinely diversifies away from 60/40's failure modes — better risk-adjusted numbers ex-vol, real cushioning in both the 2020 crash and the 2022 stocks-and-bonds year — but none of it is statistically certified, and the only cheaply-tradable long-volatility proxy (VXX) has bled 99% of its value since 2018, turning the *actual* published 5-sleeve allocation Sharpe-negative on the only window we can test it on.

## What we tested

Cole's **100-year all-weather** thesis: equities + long bonds + gold + commodity trend +
LONG volatility (published weights **24/18/19/18/21%**), designed so one sleeve profits
in *every* macro regime rather than merely diversifying across them — the piece 60/40
structurally lacks is convexity to a deflationary liquidity crisis. We proxy the five
sleeves on liquid ETFs (SPY / TLT / GLD / a 12-month trend overlay on DBC / VXX — named
as a crude, decaying stand-in for real long-vol), monthly rebalance, and race a
**Dragon-lite** (ex-vol, 2007-2026, sees 2008 too) and the **full 5-sleeve Dragon**
(2018-2026 — yfinance's own VXX tape starts 2018-01-25, nearly a decade short of the
product's 2009 launch) against 60/40 and a static All-Weather-lite. We test both
recent regime shocks directly (2020 crash, 2022 both-down year) and are brutally
honest that the long-vol sleeve is not cheaply proxiable and the available history
can't see a full secular cycle. **Dedup:**
[68-all-weather](../68-all-weather/) (risk-parity, not this study's static weights),
[144-permanent-portfolio](../144-permanent-portfolio/) (no trend or vol sleeve),
[617-crash-insurance-cost](../617-crash-insurance-cost/) (standalone tail-hedge cost,
not an allocation blend) and [655-ivy-portfolio](../655-ivy-portfolio/) (trend-times
every sleeve, no long-vol) never test this exact 5-sleeve bundle. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why 60/40 has a blind spot, what each Dragon sleeve is *for*, and why the "insurance" sleeve is the hardest one to actually buy |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the HAC/bootstrap splits, the crisis-window anatomy, the VXX decay diagnostic, the DBMF proxy-quality check, and the synthetic crisis-alpha control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`dragon_portfolio/`](dragon_portfolio/). Every sleeve is a single, currently-listed
ETF — no cross-sectional basket, so no basket survivorship; VXX is itself a 2018
"Series B" relaunch, named on the Signal axis. **Not investment advice** — research &
education. See [LICENSE](../../LICENSE).*
