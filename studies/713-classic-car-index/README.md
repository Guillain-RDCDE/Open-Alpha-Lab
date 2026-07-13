# Study 713 — Classic cars are an asset class? 🏎️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the collector-car market beat the S&P? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Over 2005–2025 the (cited, approximate) car index compounded at **+8.1%/yr** vs **+10.9%/yr** for the S&P (total return) — even the *price-only* S&P won (+8.9%). Annual excess **−4.0%/yr** (*t* = −0.90). The famous "low risk" is an appraisal artifact: de-smoothed, the index Sharpe **falls 0.96 → 0.34**, *below* the S&P's. |
| **Tradability** — is there an edge to harvest? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Net of a ~22% auction round-trip + 2.5%/yr carry the gross **8.1% collapses to ~1.7%/yr** — cash-like. The only listed proxies are a barbell: Ferrari `RACE` (+23%/yr but *t*=+1.3, n.s.) and Aston Martin `AML.L` (−43%/yr, *t*=−2.78, a −98% drawdown). No clean, scalable way to own the trade. |
| **Cars beat equities?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | On total return, on *de-smoothed* Sharpe, on drawdown **and** net-of-cost, the S&P wins every column. The price-only near-tie dies on the dividends stocks pay and the carry cars charge. |

> **In one sentence:** the "collector cars are an asset class that beats stocks" claim fails on its own tape — the index lagged the S&P even before costs, its celebrated low volatility is an appraisal-smoothing illusion (de-smoothed, true vol *exceeds* the S&P's and the Sharpe halves), the only listed proxies are a juggernaut-and-a-wreck coin flip, and a realistic auction-spread-plus-carry haircut shrinks the return to a savings account.

## What we tested

Real collector-car indices (the HAGI Top Index, the Knight Frank Luxury Investment Index, Hagerty) are **not freely API-available**, so we are transparent about it: we (a) hardcode a small, **clearly-cited, approximate** annual price-index series — base 100 @ 2005, anchored on public reporting of the **2009–2015 melt-up**, the 2016–2020 **plateau**, the 2022 bump and the 2023–24 cooling — and (b) test the only **tradable** ways to own the trade: **Ferrari (`RACE`)** and **Aston Martin (`AML.L`)** via yfinance, each benchmarked against the S&P on **both** a total-return (`SPY`) and a price-only (`^GSPC`) clock — plus the two things the pitch never does: **de-smooth the appraisal index** and **charge the auction spread + storage/insurance/maintenance carry**. The car index is a *labelled proxy*, never presented as the real index. (Same "passion-asset beats stocks" shape as [Study 358 — Watches](../../358-watch-index/).)

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the "cars beat stocks" story feels true, the chart where the boom turns into a decade of plateau, why the "smooth ride" is a mirage, and where the auction fees eat the return — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | index vs S&P on both clocks (CAGR/vol/MDD + annual-excess *t*), the Geltner de-smoothing, Newey-West proxy alpha, the carry haircut on NAV, and a synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`classic_car_index/`](classic_car_index/). Car index is a **hardcoded, cited, approximate proxy** — not a live feed; equity tickers are **labelled proxies** for the trade, not the auction price of a car. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
