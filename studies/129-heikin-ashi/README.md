# Study 129 — Heikin-Ashi

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Pooled gross **+1.37 bps/trade**, HAC *t* = **+0.56**; Δ vs random-direction control = +0.35 bps (noise); bootstrap Sharpe 95% CI [−0.15, +0.26]. Apparent long-side *t* = +3.67 is **replicated by a coin** (*t* = +3.35) — it is bull-market drift, not HA skill. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Gross is not significant; at 1 bp round-trip cost *t* = +0.15 — below noise floor. No positive break-even cost exists. |
| **Smoothing adds lag, not signal?** | ![Confirmed](https://img.shields.io/badge/Smoothing_adds_lag%2C_not_signal%3F-Confirmed-8b949e?style=flat-square) | HA flips at ~half the rate of raw candles (984 vs 1,910/ticker), yet the directional *t*-stat vs a coin is **the same** — the delay that comes with smoothing provides no forecasting benefit. |

> **In one sentence:** the Heikin-Ashi colour-flip trades a smoothed trend signal that beats a random coin by +0.35 bps/trade — statistical noise — while the appealing long-only results are explained entirely by 15 years of equity bull-market drift that an actual coin captures just as well.

## What we tested

A mainstay of retail charting and YouTube tutorials: replace standard candles with Heikin-Ashi smoothed candles (HA_close = OHLC/4; HA_open = recursive average of prior HA open and close), then trade the **colour flip** — go long on a red-to-green flip, go short on green-to-red. The smoothing is pitched as filtering "whipsaws" and keeping you "on the right side of the trend." We take that literally: build the causal HA series (no look-ahead — the recursion is evaluated bar-by-bar using only prior data), detect every colour flip on the daily chart, enter the next bar's open, and measure the direction against a **symmetric ±1 ATR** barrier exit. The only fair direction test: if HA knows something, it beats a random coin on the same entries. We also run the **raw-candle colour-flip baseline** (close >= open) to check whether the smoothing itself adds value over the unsmoothed version, across four liquid daily tapes (SPY, QQQ, IWM, AAPL) over 15 years.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the HA recipe, the "smoothing filters noise" claim, the honest coin test, why the long-side *t* is fool's gold |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-instrument HAC *t*, long/short decomposition, bootstrap Sharpe CI, HA vs raw candle comparison, cost sweep, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`heikin_ashi/`](heikin_ashi/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
