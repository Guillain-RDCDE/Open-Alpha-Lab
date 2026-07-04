# Study 636 — Exchange-Listing-Pop 🚀

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

**The "Coinbase effect": a coin pops double digits on its listing — then gives it all back. True?**

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — pop *and* give-back on the tape? | ![Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square) | On **129 Coinbase listings 2019–2026** priced on Binance: BTC-adjusted pop **+11.00%** in [−5..0] (cluster *t* = **+5.74**, month-clustered *t* = +5.30, placebo *p* < 0.0005 across 20 seeds) and fade **−17.45%** in [+1..+30] (*t* = **−5.73**) — the give-back **exceeds** the pop. Median ≈ mean, 76% of pop clusters positive. **Survivorship** named: only Coinbase-served candle histories enter; mis-dated events would dilute, not create. |
| **Tradability** — can you deploy it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | Buying at the first tradable close (day-0 close, one lag) loses **−12.6% net** in 30 days (*t* = −4.1) — the folklore trade is a money shredder. The *mirror* trade (short the fresh listing / long BTC) nets **+10.1%/event** (*t* = +3.3) at 25 bps + 10%/yr borrow, +6.9% (*t* = +2.25) at 50%/yr — and **dies at 100%/yr** (+2.8%, *t* = 0.9), exactly where borrow on a hot new listing actually trades. Short-only, borrow-bound: not INVESTABLE. |
| **"Gone before you can act?"** | ![Confirmed](https://img.shields.io/badge/Gone_before_you_can_act%3F-Confirmed-8b949e?style=flat-square) | **+10.23 pp of the +11% pop accrues in [−5..−1]** — inside the announcement-to-listing gap (2 days in the Pro era, weeks since the 2022 roadmap) — the listing day itself is **+0.77% (*t* = 0.82)**. The 2022-04-28 transparency roadmap **halved** the pop (+17.2% → +7.1%, Welch *t* = +2.61). By your first fill, it's over. |

> **In one sentence:** the Coinbase effect is gloriously real — +11% into the listing, −17% over the month after (both ~6 sigma on 129 events) — but the pop is finished before the first close you can trade, buying it there loses 12% net, and the only trade the tape certifies is *shorting* the freshly listed coin, an edge that lives or dies on whether you can actually borrow a hot new alt.

## What we tested

We rebuild the "Coinbase effect" end-to-end from the venues' own public APIs: day 0 per coin = the **first daily candle of its USD product on Coinbase Exchange** (129 events 2019–2026, hardcoded with a per-row venue price-agreement check that kills cross-venue ticker collisions), priced on **Binance daily klines** — the coins trade there long before Coinbase lists them. Event study on BTC-adjusted CARs with **same-day clustering** (Coinbase lists several coins per blog post), a random-date placebo (2,000 draws, 20-seed robust), month-level cluster robustness, and the **2022-04-28 "listing roadmap" natural experiment** (post-DOJ-case transparency). Tradability charges the only executable version — enter at the **day-0 close** (one lag), 10/25/50 bps one-way per leg, shorts pay borrow with a 10→100%/yr sensitivity. A deterministic synthetic world with a plantable pop + fade proves the machinery stays quiet on the null. *(Dedup: [249-index-inclusion](../249-index-inclusion/) is the equity cousin — the S&P 500 membership pop, dead ex-TSLA; [635-coinbase-premium](../635-coinbase-premium/) is the BTC price gap between venues; [294-coinbase-rank](../294-coinbase-rank/) is app-store attention. This is the **listing-event** study.)*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a Coinbase listing does to a coin's price, the mountain-shaped chart (up +11%, down −17%), why buying the news loses money, and who actually gets the pop — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | clustered CARs + placebo Welch *t*, month-cluster robustness, the roadmap-era natural experiment, the cost × borrow sweep on the short, and the synthetic faithful-engine control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`exchange_listing_pop/`](exchange_listing_pop/). Day 0 = first Coinbase Exchange USD candle (the venue's own record); prices = Binance USDT klines; abnormal = coin minus BTC. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
