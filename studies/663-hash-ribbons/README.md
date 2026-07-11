# Study 663 — Hash-Ribbons ⛏️🎗️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a hashrate-capitulation buy signal actually precede outsized BTC rallies? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | Only **4** genuine signals in twelve years of tape (SMA(30) crossing back above SMA(60) after a ≥21-day, ≥8% capitulation). Every horizon (30/90/180/365 days) beats the unconditional average, but no Welch *t* clears **2** (best +1.27 at 90 days) and no random-date placebo beats *p* = 0.10 (best 0.128 at 180 days). Directionally right, statistically unproven. |
| **Tradability** — can you build a strategy on it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | A 180-day timer built on the 4 signals shows a real Sharpe edge (**+1.10 vs +0.92**) but leaves **more than half the wealth on the table** (+770% vs +1,503%) by sitting in cash 78% of the time — and the whole construction rests on **3** distinct holding episodes. |
| **Beats random buy dates?** | ![Busted](https://img.shields.io/badge/Beats_random_buy_dates%3F-Busted-8b949e?style=flat-square) | Directionally ahead of a random-date placebo at every horizon, but never clears the conventional *p* < 0.05 bar (best *p* = 0.128, 180d). Four historical events can't separate "the ribbon knows something" from "BTC mostly went up." |

> **In one sentence:** Capriole's Hash Ribbons has fired only **four** times in Bitcoin's
> history — every one of them was followed by an above-average rally, but with n = 4 (two of
> them overlapping) neither a Welch *t* (best +1.27) nor a random-date placebo (best *p* =
> 0.128) can certify it, and a timer built on the signal beats buy-and-hold's Sharpe while
> giving up half its total return by sitting mostly in cash — a real but tiny, uncertifiable
> pattern, not a market-timing crystal ball.

## What we tested

Capriole Investments' **Hash Ribbons** (Charles Edwards, 2019): watch Bitcoin's 30-day hashrate
SMA against its 60-day SMA. A "capitulation" is the stretch where the 30-day sits below the
60-day (unprofitable miners shutting off rigs, often forced to sell BTC first); the buy signal
fires the day the 30-day claws back above the 60-day. We hardcode the same curated month-end
hashrate table as sibling study [292-bitcoin-hashrate](../292-bitcoin-hashrate/) (Blockchain.com
digitised), linearly interpolate it to a daily path (a named limitation — smooths day-to-day
noise, keeps the multi-month capitulation cycles), and flag genuine crossovers with a magnitude
filter (≥21 days below, ≥8% peak-to-trough hashrate decline). That yields **4** signals — 2019,
two in 2020, and 2021. We measure forward BTC returns after each at 30/90/180/365 days against
the unconditional distribution (Welch *t*, honestly noted as barely meaningful at n=4) and a
20,000-draw random-date placebo; backtest a 180-day timer against continuous buy-and-hold; and
run a deterministic synthetic control to prove the detector is unbiased. **Dedup:**
[292-bitcoin-hashrate](../292-bitcoin-hashrate/) (continuous hashrate-growth regression and a
3/6-month-MA exposure rule — a different question), [221-mayer-multiple](../221-mayer-multiple/)
(price-based valuation, no hashrate), [323-btc-halving](../323-btc-halving/) (the halving
calendar, not an observed signal), [210-crypto-trend](../210-crypto-trend/) (200-day price SMA
trend-following) and [633-btc-vol-targeting](../633-btc-vol-targeting/) (a vol-sizing overlay) —
none of them test the literal 30d/60d hashrate crossover as a rare discrete buy event. As-of
**2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what miner capitulation means, why the four signals happened when they did, and why "it worked every time" isn't the same as "it's proven" |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the crossover-detection filter, the Welch/placebo forward-return tests at four horizons, the timer-vs-buy-and-hold backtest, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`hash_ribbons/`](hash_ribbons/). Hashrate is a hardcoded, interpolated curated series
(no survivorship concept — a single network's own metric); BTC-USD is a single-survivor asset,
named. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
