# Study 439 — Linear Regression Channel 📉

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the slope rule beat the market? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The 60-day regression-slope long/flat rule's daily out-performance over buy-and-hold has **HAC *t* = −1.51** (block-permutation **p = 0.68**) — *negative* and nowhere near the **t ≥ 2** bar. No window from 20–200 days clears it, and it loses to buy-and-hold on **5 of 6** panel names. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The wafer-thin SPY net Sharpe edge (**0.71 vs 0.65**) isn't significant, **inverts cross-section**, and the long/short variant collapses to Sharpe **0.21**. There is no edge for costs to kill — even at **zero** cost it doesn't beat the index. |
| **"Beats a moving average"?** | ![Not Supported](https://img.shields.io/badge/Beats_a_moving_average%3F-Not_Supported-8b949e?style=flat-square) | Slope active *t* (**−1.51**) ≈ SMA active *t* (**−2.03**) — **both fail to beat the market**. The slope's only genuine difference is **lower turnover** (3.3 vs 15.9 round-trips/yr): a smoothness artefact, not alpha. |

> **In one sentence:** the "Linear Regression Channel" sells the least-squares slope of recent prices as a *leading, smoother* trend signal than a moving average — but on 21 years of SPY the slope timing rule's edge over buy-and-hold is **negative and insignificant** (HAC *t* = −1.51), it loses to the index on 5 of 6 other names, and it is **no better than a plain moving average** (whose only real difference is that the slope trades less). A rolling OLS slope is a lagging linear filter of past prices dressed up as a leading indicator.

## What we tested

We compute the **60-day rolling OLS slope of log-price** on **SPY total-return** daily closes (2005–2026), turn it into a long/flat timing rule (long when the slope is positive, flat otherwise), and race it **net of costs and excess-vs-excess** against (a) buy-and-hold and (b) the obvious simpler benchmark it claims to beat — the price-vs-60-day-SMA rule — so the *"it's better than a moving average"* claim is actually tested. Each rule earns the next bar's return (one documented execution lag), pays one-way costs × NAV turnover (shorts pay borrow), and is judged on the **HAC *t* of its daily out-performance over buy-and-hold**, plus a 21-day-block permutation placebo. We sweep costs and windows, test a long/short variant, and check 6 instruments. A deterministic synthetic control with a *planted* trend confirms the harness lights up when a slope edge genuinely exists — so the real-tape null is a true negative.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the regression channel is, why a "fitted line" is just a lagging average, the equity-curve race, and why staying invested beats the rule — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | rolling-OLS slope as a long/flat rule, NET excess-vs-excess vs buy-and-hold, HAC *t* on the active return, a block-permutation placebo, the head-to-head vs an SMA filter, cost/window sweeps, the 6-name panel, and a planted-trend power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`linear_regression_channel/`](linear_regression_channel/). Tape is SPY **total-return** (auto-adjusted) daily closes via yfinance, cache-first. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
