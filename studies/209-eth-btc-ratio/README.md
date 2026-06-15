# Study 209 — ETH-BTC-Ratio

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | 20-day ratio momentum HAC *t* = **+3.69** net of 5 bps cost; alpha vs 50/50 *t* = **+9.44**; bootstrap 95% CI **[+0.58, +2.00]** (0% of resamples negative); signal persists OOS (*t* = +2.12) and in 6 of 8 calendar years. |
| **Tradability** — does it survive costs, capacity, scale? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | Feasible execution (~29 trades/year, low cost); but **−68% max drawdown**, extreme vol, no diversification, concentrated crypto-only risk. Suitable only for crypto-dedicated capital. |
| **One-cycle risk?** | ![Warning](https://img.shields.io/badge/One--cycle_risk%3F-Caveat-8b949e?style=flat-square) | Entire 7.5-year history is a single crypto bull cycle (ETH 2017-2026). Signal survives OOS but regime shift risk is real. |

> **In one sentence:** the 20-day ETH/BTC ratio momentum rotation beats a 50/50 ETH+BTC baseline with a robust *t* = +3.7 and zero-negative bootstrap CI on the 2017-2026 tape — a real signal, but one strapped to a single asset class, a single cycle, and a −68% drawdown that will end most mandates before the edge has time to prove itself.

## What we tested

Crypto folklore holds that the ETH/BTC price ratio is a "risk-on / risk-off" dial within crypto:
when the ratio rises, capital rotates into Ethereum and broader alts; when it falls, it retreats
to Bitcoin. The trade: on the 20-day sign of the ETH/BTC log-ratio, hold **100% ETH** (ratio up)
or **100% BTC** (ratio down), rebalancing daily with a 5 bps round-trip cost. Pinned against a
**50/50 daily-rebalanced baseline** — the honest comparison, since both assets had extraordinary
secular returns. Lookback sweep from 5 to 120 days with Bonferroni correction. OLS alpha vs
baseline. Train/test time split. ETH-USD and BTC-USD (yfinance), 2017-11-09 to 2026-06-15,
n = 3,140 daily observations.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the ratio as a sentiment dial, the rotation recipe, fair comparison vs 50/50, year-by-year stability, the one-cycle caveat |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t-stats, bootstrap Sharpe CI, OLS alpha vs baseline, lookback sweep with Bonferroni correction, cost sweep, train/test split, positive synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`eth_btc_ratio/`](eth_btc_ratio/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
