# Study 294 — Coinbase-Rank

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

**Is the Coinbase App-Store rank a top signal?**

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | BTC's forward abnormal return (net of a market model on ETH) over [+1, +5] after a Coinbase rank spike = **−5.2%** — the omen's direction — and permutation p ≈ **0.001**. **But** the robust cross-sectional t is only **−1.69** (< \|2\|), only **8/15** events are negative, the *median* event is **−1.0%**, and dropping one crash collapses it (t → **−1.27**). A whisper on a hindsight-curated list, not a real signal. *(Selection bias named here.)* |
| **Tradability** — does it survive costs, capacity, lag? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | SHORT BTC at the spike-day close (one-day lag), hold 5d, pay 20 bps one-way × NAV **plus** 5 bps/day borrow: net **+2.4%/trade**, **t = 0.80**, hit-rate **53%** — a coin flip. Shorting the strongest-trending major asset on this loses in expectation. |
| **Busted?** | ![Mostly](https://img.shields.io/badge/Mostly-dab617?style=flat-square) | Direction is right but the effect is fragile (a couple of real blow-offs do the work) and un-harvestable at daily resolution after fees and borrow. |

> **In one sentence:** Crypto really *is* a touch weaker after the Coinbase app tops the chart — but it's a noisy, sub-2-sigma whisper carried by a few famous crashes, and you can't short it for a profit once you pay the lag, fees, and borrow.

## What we tested

We hardcode 15 of the most widely reported "Coinbase app spiked toward #1 in the
App Store" dates (2017–2025) in `data.py` and pull BTC-USD / ETH-USD daily closes
from Yahoo (cache-only). We fit a market model `BTC = α + β·ETH` on non-event days
(so ETH absorbs the broad-crypto move), measure BTC's **forward** abnormal return
over [+1, +5] after each spike (the omen is a *contrarian top*, so we look at the
days AFTER the spike), and test it with a **10,000-draw permutation test** plus an
outlier-robust cross-sectional t. Then the only question that pays: can you *trade*
it? We SHORT BTC at the **close of the spike day** (a one-day execution lag), hold
3/5/10 days, charge 20 bps one-way × NAV **and** 5 bps/day borrow (shorts pay
borrow), and benchmark against BTC long. A synthetic positive control (planted
post-spike drift) confirms the engine detects a real effect; the null tape confirms
it cries no wolf.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the omen, the post-spike fade, why ETH is the fair benchmark, why you can't short it |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | market-model abnormal returns, forward CAR window, permutation test, outlier robustness, the lagged short backtest, the positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`coinbase_rank/`](coinbase_rank/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
