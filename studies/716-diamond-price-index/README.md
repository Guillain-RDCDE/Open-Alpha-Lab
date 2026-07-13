# Study 716 — Short the diamond? 💎

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is there a harvestable *diamond* edge? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | No listed leg clears \|*t*\| ≥ 2 in the trade's favour: Signet alpha *t* = **+0.32**, Lucara alpha *t* = **−1.73**. The index's **−19%/yr** shortfall (*t* = −2.30) is on a hardcoded proxy and measures the *diagnosis*, not a tradable signal. |
| **Tradability** — can you get paid for the collapse? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Short the miner net of a 30%/yr borrow: **−22.6%/yr** — the miner fell **−86%** and the short *still* lost (volatility drag + borrow). The physical stone nets **−16.5%/yr** after the retail→resale haircut. No diamond future; nothing scales. |
| **Diamonds collapsing?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | The price index round-tripped **−31.8%** off its early-2022 peak as lab-grown wholesale fell ~80–90%. The *diagnosis* is real — it just isn't a trade. |

> **In one sentence:** the "lab-grown is killing natural diamonds, short it" thesis is **correct on the diagnosis and untradable in the P&L** — the price index fell ~1.5%/yr while the S&P compounded ~17%/yr, yet the only listed legs are alpha-free beta (Signet) or single-mine idiosyncrasy (Lucara), and even a *perfectly-timed* short of the −86% miner loses **−22.6%/yr** once you pay borrow and eat its +40% up-months.

## What we tested

A recurring luxury/finance-media pitch: lab-grown diamonds are chemically identical and ~90% cheaper, natural prices are collapsing (Rapaport RAPI / IDEX / Zimnisky, De Beers cutting rough), so **short the diamond** — short the miners, or buy a beaten-down miner for the rebound. Real polished-diamond indices are **not** freely API-available, so we (a) hardcode a small, **clearly-cited, approximate** annual price series — base 100 @ 2018, anchored on the early-2022 peak and the ~−18%/−11% (2023/24) RAPI declines — and (b) test the only **tradable** expressions: **Signet (`SIG`)** the jeweler and **Lucara (`LUC.TO`)** a pure-play miner, each benchmarked against **`SPY`** on CAGR, vol, drawdown and alpha — plus the frictions the pitch never charges: **short borrow** on an illiquid penny-stock and the **retail→resale spread** on a physical stone. The price index is a *labelled proxy*, never the real index. (Same shape as [Study 358 — Watches](../../358-watch-index/).)

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the diamond-collapse story is *true* and still couldn't be traded, and where borrow + the resale spread eat the return — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | index vs SPY (CAGR/vol/MDD + annual-excess *t*), Newey-West proxy alpha, the borrow-charged short book, the resale haircut, and a synthetic collapse positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`diamond_price_index/`](diamond_price_index/). Price index is a **hardcoded, cited, approximate proxy** — not a live feed; equity tickers are **labelled proxies** for the trade, not the price of a polished stone. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
