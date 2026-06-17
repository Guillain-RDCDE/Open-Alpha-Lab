# Study 213 — Meme Stocks

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | An investor who bought the equal-weight basket on **2021-01-28** (the day the WSB squeeze was front-page news) lost **−45%** total while SPY returned **+93%** over the same ~5 years. The "positive" strategy (buy on 2021-01-04) is carried by **one name out of six** (GME +366%); four of six names lost 42–92%. The momentum-timed backtest (n=6 trades) shows a mean net return of +513%, but those dominant triggers (GME, KOSS) fired in **Aug-Sep 2020 — months before the mania was identifiable** as a meme-stock trade. No signal survives honest timing. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | **100%+/yr annualised vol**, **−84% peak drawdown**, one delisting (BBBY), single-name concentration (GME carries the basket). The Sharpe ratio of the buy-and-hold basket (0.57) is far below SPY (0.89) despite a higher raw CAGR — and that CAGR itself is a one-name accident. No risk-adjusted edge; extreme left-tail skew. |
| **Myth check** — could you ride the mania or only be its exit liquidity? | ![BUSTED](https://img.shields.io/badge/BUSTED-8b949e?style=flat-square) | The meme-stock mania genuinely happened — but the money went to a small number of pre-mania holders, short-sellers who covered at the peak, and options market-makers. The typical retail participant who bought after the WSB coverage peaked was the exit liquidity, not the beneficiary. |

> **In one sentence:** the meme-stock mania was real, but riding it required buying weeks before it was identifiable as a meme trade — **the retail crowd that bought the news lost 45% while the S&P 500 doubled**.

## The claim

Could you have ridden the meme-stock mania, or only been its exit liquidity?

## What we tested

The **WSB meme basket** (GME, AMC, BB, BBBY, KOSS, NOK) equal-weight vs SPY,
from 2020 to 2025. Three scenarios:

- **Strategy A (early entry):** buy on 2021-01-04 (first trading day of the mania year),
  hold to 2025-12-31. Headline: +187% vs +98% for SPY. The catch: GME alone
  drove 366% while four names lost 40–92%; this is a **one-name bet** dressed
  as a diversified basket.
- **Strategy B (retail timing):** buy on 2021-01-28 (after the squeeze was
  everywhere in the media — the realistic retail entry). Result: **−45% vs +93%
  for SPY**. Exit-liquidity scenario confirmed.
- **Strategy C (momentum-timed):** enter each name when it closes +50% above
  its 60-day-ago close, hold 126 trading days. The dominant trades (GME, KOSS)
  triggered in **August–September 2020**, months before the WSB mania was
  identifiable. The apparent +513% mean net return is **look-ahead-contaminated**
  by the fact that we selected the basket in hindsight.

Survivorship bias is named and absorbed: BBBY is held through its full life
(bankruptcy April 2023, delisted); no silent removal. Costs: 20 bps one-way.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the Jan-2021 squeeze story, equity curves for all three scenarios, the exit-liquidity illustration |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | vol/Sharpe/drawdown tables, single-name concentration analysis, look-ahead contamination of the momentum signal, the survivorship handling |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`meme_stocks/`](meme_stocks/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
