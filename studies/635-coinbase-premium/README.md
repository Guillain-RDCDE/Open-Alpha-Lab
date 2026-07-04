# Study 635 — Coinbase-Premium 🪙

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

**When Coinbase trades rich to Binance, do US institutions tip the next move?**

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the premium lead the market? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The lean is positive at **every** horizon and era, but the claim as stated (premium level → next-day return) is **HAC *t* = +1.42** on 8.9 years / 3,240 days. The legs that clear 2 — winsorized h = 1-3 (*t* = 2.10-2.30), raw h = 3 (+2.09), 2023+ (+2.87) — each need a spec or sample choice. And the celebrated 2020-21 "institutional bid" era itself predicts **nothing** (*t* = +0.56). A whisper, not a certified signal. |
| **Tradability** — can you time BTC with it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Long-only z-score overlay: timing alpha over exposure-matched buy & hold = **+0.80 HAC *t*** at 10 bps one-way, **negative** at 25 bps, and window-fragile (z20/z63/z126 alpha *t* = 2.15/0.80/1.48 — spec search, not edge). 20-seed random twin ≈ 0 confirms the machinery pays nothing for luck. |
| **"Dead since the ETFs?"** | ![Busted](https://img.shields.io/badge/Dead_since_the_ETFs%3F-Busted-8b949e?style=flat-square) | The premium **compressed ~10-20×** (2026 H1: negative on 145/181 days) — but per bp it now says **22× more** (slope difference *t* = +2.72) and 2023+ is the only slice clearing the bar alone (*t* = +2.87). The signal shrank and **sharpened**; it did not die. |

> **In one sentence:** the Coinbase premium is a real *coincident* footprint of the US bid — 2020-21 sat persistently rich, 2026 sits persistently cheap — but as a *leading* signal it is a sub-2-sigma whisper (raw next-day HAC *t* = +1.42) that no cost-charged timing rule certifies, and the ETF era compressed the gap while, per bp, making it *sharper*, not dead.

## What we tested

We rebuild the industry's "Coinbase Premium Gap" from the venues' own public APIs — Coinbase Exchange BTC-USD daily candles vs Binance BTCUSDT daily klines, same 00:00-UTC close, 2017-08-17 → 2026-06-30 — and test the folklore end-to-end: the regime narrative (the 2020-21 institutional bid is on the tape: positive on 72-88% of days, then a 10-20× compression), predictive HAC-*t* regressions of forward 1-7d returns on the premium's level and change (with a 1%/99% winsorized leg for the named **USDT-peg caveat**), a long-only trailing-z timing overlay with exactly one execution lag and one-way costs × traded NAV, an exposure-matched **20-seed** random-timing twin, and a pre-registered post-2022 death check with an HAC-SE slope-difference test. A deterministic synthetic two-venue world with a plantable "premium leads returns" knob proves the machinery stays quiet on noise and lights up on a planted lead. *(Dedup: [294-coinbase-rank](../294-coinbase-rank/) is the Coinbase **App-Store rank** — retail attention; this study is the **exchange price premium** — the institutional-flow footprint.)*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the premium is, why 2020-21 made it famous, what a "leading indicator" has to prove, and why a fat-looking gap can still be statistical noise — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC slope *t*'s across horizons, winsorized robustness, the cost-charged overlay + 20-seed random twin, era splits with a slope-difference test, and the planted-lead synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`coinbase_premium/`](coinbase_premium/). The signal is the **cross-venue price premium** `cb/bn − 1` at the shared UTC close; the named caveat is the **USDT leg** (peg stress moves the series with no Coinbase flow). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
