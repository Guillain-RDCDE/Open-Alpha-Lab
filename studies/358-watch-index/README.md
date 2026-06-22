# Study 358 — Watches are an asset class? ⌚

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the watch market beat the S&P? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Over 2018–2025 the (cited, approximate) resale index returned **+5.8%/yr** vs **+15.0%/yr** for SPY — *more* vol, *deeper* drawdown. Mean annual excess **−9.5%/yr** (*t* = −1.05). The buyable proxies show **no** significant alpha (\|*t*\| < 1). |
| **Tradability** — is there an edge to harvest? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Even the gross index CAGR goes **negative (−2.8%/yr)** after a realistic ~20% dealer spread + 1%/yr carry. The only thing you can buy — `WOSG.L` (−77% drawdown, β≈2), `CFR.SW` — is higher-vol beta with no alpha. Illiquid, high-carry, round-tripped 2022–24. |
| **Watches beat the S&P?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | On return, Sharpe, drawdown **and** net-of-cost, the S&P wins every column. The winners bought before March 2022 and sold near the top — bubble survivors, not an asset class. |

> **In one sentence:** the "luxury watches are an asset class that beats stocks" claim fails on its own tape — the secondary-market index lagged the S&P by ~9%/yr with more risk, the bubble round-tripped −36% off its March-2022 top, the only listed proxies are alpha-free high-beta beta, and a realistic dealer-spread haircut turns the watches' gross return **negative**.

## What we tested

Real secondary-market watch indices (WatchCharts, Subdial, the Morgan Stanley×WatchCharts review) are **not freely API-available**, so we are transparent about it: we (a) hardcode a small, **clearly-cited, approximate** annual resale-index series — base 100 @ 2018, anchored on public reporting of the 2019–21 melt-up, the **March-2022** blow-off top, and the −10.7%/−6.1%/+4.9% (2023/24/25) round-trip — and (b) test the only **tradable** ways to own the trade: **Watches of Switzerland (`WOSG.L`)** and **Richemont (`CFR.SW`)** via yfinance, each benchmarked against **`SPY`** on CAGR, volatility, drawdown, alpha — and the part the pitch never charges, **dealer spread + illiquidity carry**. The resale index is a *labelled proxy*, never presented as the real index. (Same survivorship-narrated-as-system shape as [Study 301](../../301-triple-rsi/).)

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the "watches beat stocks" story feels true, the chart where the bubble round-trips, and where the spread eats the return — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | index vs SPY (CAGR/vol/MDD + annual-excess *t*), Newey-West proxy alpha, the carry haircut on NAV, and a synthetic bubble positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`watch_index/`](watch_index/). Resale index is a **hardcoded, cited, approximate proxy** — not a live feed; equity tickers are **labelled proxies** for the trade, not the resale price of a watch. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
