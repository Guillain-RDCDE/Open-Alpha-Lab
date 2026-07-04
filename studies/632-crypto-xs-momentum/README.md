# Study 632 — Crypto Cross-Sectional Momentum 🪙

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do last week's winning coins keep winning? | ![Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square) | Weekly winners-minus-losers quintiles on a 44-coin panel pay **+164 bps/week** (ann. ≈ +134 %, Sharpe 1.12) at **HAC *t* = 3.56**, robust at 3–4-week formations (*t* = 2.95 / 2.61) and against a 20-seed random-rank placebo (~0). Clears the *t* ≥ 2 bar on the real tape. **Survivorship named**: 11 delisted/dead pairs (LUNA's −100 % week, FTT's −93 % week) soften but don't remove the live-name tilt. |
| **Tradability** — can you deploy it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | The sort churns **~1.5× NAV per leg per week**: net of 10 bps one-way the spread still clears the bar (*t* = 2.47), at 25 bps it's *t* = 1.46, at 50 bps it's dead. Bull-loaded (bull *t* = 3.51 vs bear 1.13) and decayed (2023–2026 *t* = 1.30 alone). The long-only winners tilt survives 25 bps (*t* = 2.21) — real but conditional, not INVESTABLE. |
| **"Does it survive the 2022 bear?"** | ![Busted](https://img.shields.io/badge/Survives_the_2022_bear%3F-Busted-8b949e?style=flat-square) | Calendar 2022: **+10.5 bps/wk, HAC *t* = 0.13, hit 48.1 %** — nothing, straight through LUNA and FTX. Lagged bear-regime weeks overall: *t* = 1.13; bull−bear Welch *t* = 2.09. Crypto momentum is a **bull-market factor**. |

> **In one sentence:** Liu-Tsyvinski-Wu holds up — sorting ~35 top coins on last week's return and holding the winner quintile one week pays +164 bps/week at HAC *t* = 3.56 on 2017–2026 Binance data (with LUNA's death week in the panel) — but the weekly churn eats it beyond major-pair fees, the *t* collapses in bear regimes, and in 2022 it paid exactly nothing — **Real, but Fragile**, and the "crypto's one true factor" story fails its bear-market test.

## What we tested

We rebuild the Liu-Tsyvinski-Wu cross-sectional momentum factor as a weekly quintile panel: **44 top-cap coins** (Binance spot 1w klines spliced with yfinance `-USD` backfills, 2017 → 2026), including **11 delisted-from-Binance / dead pairs** — LUNA truncated at its halt so its −100 % crash week stays in and the unrelated Terra 2.0 relisting stays out — to soften survivorship, which we name on the Signal axis regardless. Each week we rank eligible coins on their past *k*-week return (headline *k* = 1, the claim as stated), go long the top quintile and short the bottom, equal-weight, with **exactly one week of execution lag**. Inference is Newey-West (4 lags) on the weekly spread; costs are one-way bps × measured traded NAV with the short leg paying 10 %/yr borrow; a 20-seed random-rank placebo and a planted-AR(1) synthetic control prove the machinery. Sub-periods and a lagged BTC 30-week-SMA regime split answer the third axis. Distinct from the desk's [251-crypto-reversal](../251-crypto-reversal/) (opposite-sign reversal) and [222-altseason-rotation](../222-altseason-rotation/) (BTC-dominance rotation): this is the coin-level **continuation** panel. As-of **2026-06-28** (last complete UTC week).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "buy last week's winners" means, why it worked, what it did through the LUNA/FTX year, and why fees and bear markets are the catch — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC *t*'s across formations, sub-period and regime splits, turnover × cost grid with short borrow, the 20-seed placebo, and the planted-momentum synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`crypto_xs_momentum/`](crypto_xs_momentum/). Weekly winners-minus-losers quintiles, one-week lag, delisted pairs included. Panel is still survivor-tilted — named on the Signal axis. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
