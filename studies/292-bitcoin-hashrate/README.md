# Study 292 -- Bitcoin-Hashrate

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Next-month BTC return on this-month hash-rate growth: HAC *t* = **-0.05** (1mo), no horizon clears *t* >= 2; in a horse race against BTC's own price momentum the hashrate slope collapses to *t* = -0.07. The "price follows hashrate" co-movement is a shared up-trend (spurious in levels); causality runs price -> hashrate. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The folk "long when hashrate rises" rule prints +83%/yr only by being long a 1000x survivor 86% of months -- and it **still loses to buy-and-hold** (Sharpe 0.94 vs 1.04) net of 30 bps. Hash-Ribbons "wins" only by being long even more. No incremental edge survives the one honest benchmark. |
| **Survivorship bias** | ![Named](https://img.shields.io/badge/Survivorship--biased-8b949e?style=flat-square) | BTC is the single surviving crypto that ~1000x'd; the hash-rate / price co-trend is conditioned on that survival. Every long-biased result is an ex-post upper bound. |

> **In one sentence:** Bitcoin's hash rate and its price both trend up, so they *look* linked -- but hash-rate growth has zero predictive content for next-month returns (HAC *t* = -0.05, and it dies completely in a horse race against price momentum), and the famous "long when hashrate rises" rule only appears to work because it is long a moonshot asset most of the time while still underperforming buy-and-hold.

## The claim

> *Does Bitcoin's hash rate predict its price?*

## What we tested

We join a curated month-end **hash-rate series** (EH/s, digitised from the public
Blockchain.com chart, 2014-2026) to the **BTC-USD monthly close**. Forget levels
(two trending series are a spurious-regression trap): we test *growth rates*. We
(a) regress next-month BTC return on this month's hash-rate growth at 1/3/6-month
horizons with HAC t-stats; (b) run a **horse race** adding BTC's own price
momentum, to see whether hashrate adds anything; (c) backtest the folk **"long
when hashrate rises"** timing rule against **buy-and-hold** net of 30 bps one-way
costs (one-month execution lag, long-only, price-only returns); and (d) check the
**Hash-Ribbons** 3/6-MA crossover. A deterministic synthetic positive control
(planted `beta`) confirms the regression recovers a real hashrate -> price
lead-lag when one exists (*t* = 5.13) and reads zero on the null.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the two trending lines, why co-movement isn't prediction, the honest growth-rate test, and why the timing rule "wins" only by being long a moonshot |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC predictive regressions at 1/3/6 months, the price-momentum horse race, timing-vs-buy-and-hold net of costs, the Hash-Ribbons crossover, cost/lag honesty, and the synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`bitcoin_hashrate/`](bitcoin_hashrate/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
