# Study 711 — A Birkin beats the S&P (and even gold)? 👜

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the bag beat stocks & gold? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Over 2015–2025 the (cited, approximate) resale index returned **+5.0%/yr** vs **+14.7%/yr** for SPY and **+14.6%/yr** for gold — it *lost* both races. Annual excess **−10.7%/yr** vs SPY (*t* = −1.95), **−10.9%/yr** vs gold (*t* = −1.52). The closest tradable leg, Hermès's own stock, is NW *t* = **1.93 < 2** — and it isn't the bag. |
| **Tradability** — is there an edge to harvest? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The gross index CAGR goes **negative (−7.2%/yr)** after a realistic ~30% consignment spread + carry. The only thing you can buy — `RMS.PA`, `MC.PA`, `KER.PA` (−74% drawdown) — is single-stock luxury beta. Illiquid, wide-spread, no scalable book. |
| **Beats stocks & gold?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | On return **and** net-of-cost the bag loses to both assets the headline names. The famous **14.2%/yr** is a cherry-picked, survivorship-laden 1980–2015 marketing number. |

> **In one sentence:** the "an Hermès Birkin out-returns the S&P and even gold" claim fails on the modern tape — the resale index trailed *both* by ~10%/yr, it's a lovely low-vol store of value but no return machine, a ~30% consignment spread turns even its gross return **negative**, and the only leg that came close (Hermès the *stock*, not the bag) still can't clear *t* ≥ 2.

## What we tested

The viral **Baghunter (2016)** claim — recycled by luxury media and Credit-Suisse-style collectibles notes — that a Birkin returned **~14.2%/yr over 1980–2015**, "never a down year," beating the S&P (quoted at ~8.7%) *and* gold. Real handbag-resale indices (Knight Frank Luxury Investment Index, Art Market Research) aren't freely API-available, so we (a) hardcode a small, **clearly-cited, approximate** annual resale index — base 100 @ 2015, anchored on Hermès's routine price hikes, the 2020–22 melt-up and the 2023–24 luxury-handbag cooling — and (b) test the only **tradable** ways to own the trade: **Hermès (`RMS.PA`)**, **LVMH (`MC.PA`)** and **Kering (`KER.PA`)** via yfinance, each benchmarked against **`SPY`** *and* **`GLD`** on CAGR, volatility, drawdown, alpha — and the part the pitch never charges, a **~30% consignment/dealer spread + illiquidity carry**. The resale index is a *labelled proxy*, never presented as the real index. (Same "luxury object as an asset class" shape as [Study 358 — Watch-Index](../358-watch-index/).)

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the "Birkin beats stocks and gold" story feels true, the chart where the bag comes last, and where the consignment spread eats the return — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | index vs SPY & GLD (CAGR/vol/MDD + annual-excess *t*), Newey-West maison alpha, the consignment haircut on NAV, and a synthetic compounder positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`birkin_index/`](birkin_index/). Resale index is a **hardcoded, cited, approximate proxy** — not a live feed; equity tickers are **labelled proxies** for the trade, not the resale price of a bag. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
