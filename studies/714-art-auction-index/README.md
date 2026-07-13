# Study 714 — Contemporary art is an asset class? 🖼️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the art market beat the S&P? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Over 2000–2025 the (cited, approximate) auction index returned **+5.7%/yr** vs **+8.8%/yr** for SPY — with a *deeper* drawdown (−44% vs −37%). Mean annual excess **−3.3%/yr** (*t* = −0.69, n=25). The buyable proxies show **no** significant alpha (\|*t*\| ≪ 2). |
| **Tradability** — is there an edge to harvest? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Even the gross index CAGR goes **negative (−0.15%/yr)** after a ~25% buyer's premium + 10% seller's commission + carry. The auction houses are all *private* (Sotheby's taken private 2019, Christie's/Phillips never listed); the one directly-listed proxy, `MCHN.SW` (Art Basel), **lost money** with a −95% drawdown. |
| **Art beats the S&P?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | On return, Sharpe, drawdown **and** net-of-cost, the S&P wins every column. The winners bought a specific name early and sold at a record — boom survivors, not an asset class. |

> **In one sentence:** the "contemporary art is an asset class that beats stocks" claim fails on its own tape — the auction index lagged the S&P by ~3%/yr with a deeper drawdown, both booms round-tripped (−44% in 2008–09, −20% in 2023–24), you literally can't buy the auction houses (all private), and a realistic 25% buyer's-premium haircut turns the art's gross return **negative**.

## What we tested

Real art indices (Artprice's *Contemporary Art Market* report, the **Sotheby's Mei Moses** repeat-sales index) are **not** freely API-available, so we are transparent about it: we (a) hardcode a small, **clearly-cited, approximate** annual auction-price index — base 100 @ 2000, anchored on public reporting of the 2000s melt-up, the 2008–09 crash, the 2014 peak, the **2021–22 records** boom (the Macklowe collection alone made **$922M**) and the **2023–24** correction — and (b) test the only **listed** ways to touch the trade: **MCH Group (`MCHN.SW`**, organiser of Art Basel) and **Kering (`KER.PA`**, whose owner Pinault controls Christie's) via yfinance, each benchmarked against **`SPY`** on CAGR, volatility, drawdown, alpha — and the part the pitch never charges, the **buyer's premium + seller's commission**. The auction index is a *labelled proxy*, never presented as the real index; there is **no listed auction house left to buy** (Sotheby's `BID` was taken private in 2019). (Same collectible-as-asset-class shape as [Study 358 — Watches](../../358-watch-index/).)

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the "art beats stocks" story feels true, the chart where both booms round-trip, and where the buyer's premium eats the return — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | index vs SPY (CAGR/vol/MDD + annual-excess *t*), Newey-West proxy alpha, the premium haircut on NAV, and a synthetic bubble positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`art_auction_index/`](art_auction_index/). Art index is a **hardcoded, cited, approximate proxy** — not a live feed; equity tickers are **labelled proxies** for the trade, not the hammer price of a painting. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
