# Study 595 — Managed-Futures Sleeve 🧯

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a 15% trend sleeve improve a 60/40? | ![Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square) | **Real on the correlation · Weak on the certified improvement.** The "near-zero correlation" half is decisive: **−0.054** [−0.167, +0.060] over 297 months, live DBMF **−0.21** / KMLM **−0.43**. The improvement half points the right way everywhere — Sharpe **0.643 → 0.738**, maxDD **−32.4% → −23.4%**, sleeve alpha on the 60/40 **+8.1%/yr at NW t = 2.10** — but the block-bootstrap **ΔSharpe CI spans zero** at every sleeve size ([−0.06, +0.24] at 15%) and the alpha t collapses to **1.35-1.92** across neighbouring lookbacks/costs. Mild futures-panel survivorship named. |
| **Tradability** — can you deploy it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | Genuinely accessible — DBMF/KMLM are liquid one-ticket wrappers and the **0.85%/yr fee is charged in every headline**; the live window shows the same story net of fees (ΔSharpe **+0.119**, DD −20.1% → −13.5%). But the benefit is **regime-dependent**: the sleeve *dragged* on the blend for the whole 2009-2019 decade (ΔSharpe **−0.107**), and the certified edge is absent. Cheap to buy; not promised to pay. |
| **"Just bonds in disguise?"** | ![Busted](https://img.shields.io/badge/Bonds_in_disguise%3F-Busted-8b949e?style=flat-square) | Zero bond loading (corr **−0.021**, R² **0.0%**); in 2022 bonds lost **−13.2%** while the sleeve made **+33.1%** (DBMF **+21.6%** live); moving the same 15 points into bonds gives a *worse* Sharpe (0.664 vs 0.738) and no 2022 protection (−15.15% vs −9.02%). |

> **In one sentence:** the managed-futures sleeve's diversification pitch is half-true on 25
> years of tape — the near-zero (even negative) correlation is real and it is *not* bonds in
> disguise (2022: bonds −13%, sleeve +33%), but the Sharpe improvement itself (+0.10, drawdown
> cut by 9 points) never clears the desk's inference bar once you bootstrap it or nudge the
> lookback, and it spent 2009-2019 quietly dragging — insurance that provably lowers
> correlation, not provably raises Sharpe.

## What we tested

The allocator's claim, not the trend-follower's: **does carving 15% of a 60/40 into a
trend-following sleeve improve the portfolio?** Siblings [31-trade-winds](../31-trade-winds/)
and [518-time-series-momentum](../518-time-series-momentum/) already graded the *standalone*
premium **Weak** — we cite them and do not re-litigate it. We build a simple 12-month TSMOM
book (sign of trailing 12-month return, inverse-vol 40%/σ per contract, monthly rebalance,
one-month lag, 5 bps one-way, shorts post margin with no borrow) on the desk's shared
18-futures panel (2001-09 → 2026-05, 297 months), cash-collateralise it, charge the live ETF's
0.85%/yr fee, and blend it 15/85 with a monthly-rebalanced SPY/VBMFX 60/40. The decisive
statistics are the sleeve's **Newey-West alpha on the 60/40** (the mean-variance criterion), a
**paired moving-block bootstrap of ΔSharpe**, the **correlation with Fisher-z CI**, and the
**2022 attribution** — cross-checked on the live wrappers (DBMF/KMLM, 2019+) and against a
24-seed random-sign placebo (a zero-edge sleeve *hurts* the blend, −0.077 ΔSharpe). A seeded
synthetic world with a planted MF Sharpe proves the machinery fires on real edge and stays
quiet on none. Excess-vs-excess Sharpe races throughout; as-of 2026-07-03.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a managed-futures sleeve is, why "uncorrelated" is the honest half of the ad, what actually happened in 2022, and why the improvement is real-looking but unprovable — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the NW alpha criterion, the paired block-bootstrap ΔSharpe CI, sleeve-size/cost/lookback robustness, sub-period regime splits, the bonds-in-disguise regression, the live-ETF cross-check, the random-sign placebo and the synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`managed_futures_allocation/`](managed_futures_allocation/). Futures panel = study 31's shared cache (yfinance front-month splices; mild survivorship named on the Signal axis). Replication is gross of real-world slippage beyond the modeled bps — the live ETFs are the honest net tape. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
