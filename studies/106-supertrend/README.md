# Study 106 — Supertrend

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | Pooled gross **+32.32 bps/trade**, HAC *t* = **+3.27**; beats a random-direction control by **+28.6 bps**, bootstrap 95% CI on Sharpe fully positive ([+0.18, +0.81]). |
| **Tradability** — does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | Costs trivial at ~8 flips/yr; the binding constraint is **~2.5%/yr gross** — low absolute return, multiplier-specific (only mult=3 works), long-side driven, bull-market-era sample. |
| **Beats a coin?** | ![Confirmed](https://img.shields.io/badge/Beats_a_coin%3F-Confirmed-8b949e?style=flat-square) | The flip clearly outperforms random-direction entries at the same dates on a fair symmetric bet — the ATR band filter adds real information at the daily horizon. |

> **In one sentence:** unlike the 5-minute SMA crossover (Study 72, signal None), the Supertrend(10, 3) daily flip carries a statistically real directional signal (t=+3.27) — but with only ~8 flips per year per stock, the annualised gross return is a modest ~2.5%, and the signal is multiplier-specific and long-side driven, making it fragile rather than investable.

## What we tested

The Supertrend indicator — ATR(10, multiplier 3), the TradingView default — is among the world's most-viewed technical indicators. When price crosses above the ATR band, the band locks as trailing support and the signal flips bullish; when price falls below the band, it flips bearish. We took that literally: run the canonical Supertrend(10, 3) flip signal on **four liquid daily tapes** (SPY, QQQ, IWM, AAPL, 10 years), enter the next bar's open **in the direction of the flip** with a **symmetric ±1 ATR** exit (the only direction-fair payoff), and pin it against a **random-direction control** on identical flip dates. We also sweep costs, split long vs short flips, check two fixed-day forward horizons (5/20 days), and test alternative ATR parameters to probe robustness. A deterministic synthetic tape with tunable momentum serves as the positive control, confirming the engine recovers an edge only when an edge exists.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what Supertrend is, why it is different from a coin, the fair bet result in plain language, why the signal is real but the return is modest |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-instrument HAC *t*, bootstrap Sharpe CI, long vs short flip decomposition, cost sweep, parameter robustness, sub-period stability, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`supertrend/`](supertrend/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
