# Study 622 — Thematic-ETF-Curse 🪤

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do thematic ETFs bleed risk-adjusted returns after launch? | ![Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square) | The calendar-time portfolio of thematics in their first 36 months loses **−16.30%/yr** of CAPM alpha (HAC *t* = **−3.27**; first 12 months: −15.00%/yr, *t* = −2.04), robust to lags, membership floor and dropping all of ARK; the broad-index-launch placebo is **clean** (+0.31%/yr, *t* = +1.17). **Survivorship named** — dead thematics are missing from yfinance, which biases *against* this finding. Nuance: seasoned thematics bleed too (young-vs-seasoned spread ≈ 0) — the launch starts the bleed, it isn't a uniquely poisoned window. |
| **Tradability** — can you monetize the curse? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | The beta-hedged short nets **+13.2%/yr (t = 2.90)** at 5 bps + 300 bps borrow and still clears at 600 bps — but the alpha sits in exactly the names with double-digit real-world borrow (cannabis, crypto, sentiment funds), dies by 1000 bps (*t* = 1.37), and the accessible version (don't buy launches) is an avoidance rule, not a return stream. |
| **"Buy the −50% dip"?** | ![Busted](https://img.shields.io/badge/Buy_the_--50%25_dip%3F-Busted-8b949e?style=flat-square) | Down 50% is mid-bleed, not cheap: the dip-bought book loses **−19.39%/yr** of alpha (*t* = −2.80), the average dip lost **36.6 pp** to SPY over the next 3 years, and **75%** of the 32 events lost. The hype premium keeps bleeding below the halfway mark. |

> **In one sentence:** across 48 thematic ETF launches since 2005, the fund's birthday really was the sell signal — the young-thematics book bled **−16%/yr of CAPM alpha (HAC *t* = −3.27)** while identical-construction broad-index launches bled nothing, the dead funds Yahoo forgot would only make it worse, and buying the −50% dip just caught the same knife lower — yet the clean short is throttled by borrow and capacity, so the tradable lesson is the free one: **don't buy the launch**.

## What we tested

Ben-David, Franzoni, Kim & Moussawi (RFS 2023) claim specialized ETFs are launched at the peak of a theme's hype and lose ~5%/yr risk-adjusted afterwards. We rebuild it on **48 thematic ETF launches (2005–2021)** — solar, cyber, cannabis, cloud, genomics, metaverse, blockchain, the full ARK complex — with **launch = the first candle** (each ticker validated against the issuer's inception month to catch Yahoo ticker reuse). The Signal axis is a **calendar-time portfolio** of funds in their first 12/36 complete months, CAPM-regressed on SPY excess returns with a **Newey-West *t***; contrasts: the same book on *seasoned* thematics (months 37+), a beta-controlled young-minus-seasoned spread, and a **placebo of 13 broad plain-vanilla index-ETF launches**. An event-time abnormal-return curve cumulates the bleed (−22.5% by month 36). Tradability shorts the young book (beta × SPY hedge, 5 bps one-way, borrow swept 300/600/1000 bps). The third axis buys each fund's first **−50% drawdown** with a one-month lag and holds 36 months. A deterministic synthetic panel with a planted post-launch drag proves the machinery (null does not fire; −8%/yr planted is recovered). **Survivorship is named on the Signal axis and runs against the claim** — the delisted worst bleeders are absent, flattering the panel. One execution lag everywhere; prices total-return; alphas excess-vs-excess. As-of **2026-06-30**.

Siblings, for the dedup-minded: [334-ark-innovation](../334-ark-innovation/) is one fund's boom-bust; [393](../393-ai-datacenter-basket/)–[396](../396-reshoring-basket/) test today's thematic baskets. This study is the **launch-timing claim across the category**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why fund companies launch a Solar/Metaverse/Cannabis ETF exactly when it's on magazine covers, what a dollar at every launch actually earned, and why "it's down 50%, it's cheap now" kept not working — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the calendar-time CAPM alphas with NW *t* (lags 6/12), the broad-launch placebo, the young-vs-seasoned spread, the event-time curve, per-fund vs calendar-time reconciliation, the borrow sweep on the short, the dip test, and the planted-drag synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`thematic_etf_curse/`](thematic_etf_curse/). The signal is fund age (public information); the placebo is broad-index launches; the myth-check is the −50% dip-buy. Panel is thematic **survivors** — named on the Signal axis, and the bias runs against the finding. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
