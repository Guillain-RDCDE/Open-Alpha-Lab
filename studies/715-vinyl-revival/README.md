# Study 715 — Vinyl is back — a trend to trade? 🎵

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is there a tradable vinyl edge over the S&P? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The *trend* is real (RIAA vinyl revenue **+21.8%/yr**, 18 straight years) — but revenue growth isn't an investable return, and the annual-excess win isn't even clean (*t* = **+0.46**). What you can *buy* shows **no** significant alpha: WMG *t* = **−1.23**, SPOT *t* = **+0.12**, UMG.AS *t* = **−0.66**. |
| **Tradability** — can you harvest it? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | You can't custody the RIAA series. WMG (**+2.8%/yr**) and UMG (**+1.3%**) *lagged* SPY (**+13.6%**) with deeper drawdowns; SPOT beat it on **streaming** (β≈1.65, **−75%** drawdown, zero alpha). Physically flipping records nets **negative** once a supply-expanding market meets marketplace spread + storage. |
| **Already priced in?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | Vinyl is ~**7%** of a business ~**84%** streaming; the majors are priced on streaming and no proxy carries a vinyl premium. A famous, public, 18-year trend is not a secret you can trade. |

> **In one sentence:** the vinyl revival is genuinely real — U.S. vinyl revenue out-grew the S&P for a decade — but it's a *revenue* trend you can't buy: the only listed proxies show no significant alpha (two badly lagged SPY, the one that beat it did so on streaming), and physically flipping records loses to the spread, so "vinyl is back, trade it" is a category error the market has already priced.

## What we tested

A recurring lifestyle-and-markets claim: the **vinyl revival** is a durable, tradable trend
— records are back, sales grow every year (per the [RIAA year-end statistics](https://www.riaa.com/u-s-sales-database/)),
so put money on it and ride the boom. We take the strongest version: (a) a small,
**clearly-cited, approximate** hardcoded RIAA vinyl-revenue series (base 100 @ 2010) — a
*labelled proxy for the trend*, never a live tape or a price — and (b) the only **tradable**
ways to own the trade — **Warner Music (`WMG`)**, **Spotify (`SPOT`)** and **Universal Music
(`UMG.AS`)** via yfinance, each benchmarked against **`SPY`** on CAGR, volatility, drawdown
and Newey-West alpha — plus the part the pitch never charges, the **collector round-trip
spread + storage carry**. (Same "passion asset beats stocks" shape as [Study 358](../358-watch-index/).)

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why "vinyl is back, trade it" feels true, the chart where a real trend meets an untradable series, and where the spread eats the collector — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | revenue index vs SPY (CAGR/vol/MDD + annual-excess *t*), Newey-West proxy alpha, the collector carry haircut on NAV, and a synthetic revival positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`vinyl_revival/`](vinyl_revival/). The vinyl series is a **hardcoded, cited, approximate proxy** for industry revenue — not a live feed and not a price; equity tickers are **labelled proxies** for the trade, dominated by streaming. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
