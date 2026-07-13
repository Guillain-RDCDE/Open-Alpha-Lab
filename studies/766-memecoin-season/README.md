# Study 766 — Memecoin-Season 🐕

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a momentum rotation harvest the memecoin blow-off past BTC? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Rotation-minus-BTC weekly excess is **t = +0.59** (n = 267) — nowhere near \|t\| ≥ 2. A **random** weekly coin-pick matches or beats it **51.7%** of the time (p = 0.52). No sub-period is certified. And the raw "memecoins won" claim is one survivor: over the window **SHIB +300%** but **DOGE −80%**, **BTC −5%**. |
| **Tradability** — can you build a strategy on it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Net of a mild 30 bps/leg the rotation returns **−73.6%** vs BTC **−2.5%** and an equal-weight basket **+217.2%**, with an **−89%** drawdown at **162%/yr** vol. It loses **60% even at zero cost** — the failure is structural (buy-high-sell-low in ±162% assets), not fees. Its one flattering leg (owning SHIB) is survivorship. |
| **Beats a coin flip?** | ![Busted](https://img.shields.io/badge/Beats_a_coin_flip%3F-Busted-8b949e?style=flat-square) | A dartboard weekly allocation does as well or better about half the time (p = 0.52 on return, 0.41 on Sharpe). The momentum ranking extracts no exploitable structure. |

> **In one sentence:** The memecoin-season *phenomenon* is half-real and half-survivorship (the
> one dog-coin that survived, SHIB, did outrun Bitcoin), but the memecoin-season *strategy* — a
> weekly momentum rotation across BTC/DOGE/SHIB — is a mirage that finishes dead last, loses money
> even for free, and can't beat a coin flip, because a positive weekly edge is devoured whole by a
> −½σ² volatility tax at 162%/yr vol.

## What we tested

Crypto folklore's "memecoin season": in euphoric windows the dog-coins blow past Bitcoin, and you
just [rotate into whatever's pumping](https://www.blockchaincenter.net/altcoin-season-index/) to
harvest it. We build the literal mechanical version — every Friday, hold whichever of **BTC, DOGE,
SHIB** had the best trailing 4-week return, over the next week (one-week execution lag, 30 bps per
leg) — and race it against holding Bitcoin and against an equal-weight basket, over the common
window all three trade (2021-04 → 2026-06, yfinance). Survivorship is named on the Signal axis:
DOGE and SHIB are the two memecoins that *survived* out of thousands, so every number is an ex-post
upper bound. A random-rotation placebo, a 2021-mania/after split, and a persistence-planting
synthetic control complete the teardown.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the memecoin-season story, why only one coin actually won, why the rotation finishes last, and why it loses even for free |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the excess-over-BTC *t*, the volatility tax (positive mean, negative compounding), the 4,000-seed random-rotation placebo, the mania/after split, and the momentum-persistence synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`memecoin_season/`](memecoin_season/). BTC/DOGE/SHIB are single-survivor assets and the
memecoin sleeve is survivorship-selected — both named on the Signal axis. Price-only == total-return
for crypto (no dividends). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
