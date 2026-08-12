# Study 894 — Trend Overlay on 60/40 📉

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a 200-day trend filter on 60/40 cut drawdown *and* keep the return? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The overlay delivers a **real, robust drawdown & vol cut** — max DD **−12.5% vs −30.8%** (−18 pp), vol 6.9% vs 11.4%, positive in *both* eras. But the *Sharpe advantage* is **not robust**: +0.13 with a bootstrap CI that **straddles zero** ([−0.24, +0.52], P>0 = 0.74), the daily return difference is *negative* and insignificant (−0.87 bps/day, NW *t* = −1.18 — it gives up ~1.9 pp/yr of CAGR), and the edge is **front-loaded on 2008**, fading to +0.07 post-2017. Real risk reduction, no robust risk-adjusted *outperformance*. *Short history (BIL from 2007) flatters a one-crash trend rule — named on Signal.* |
| **Tradability** — is the edge bankable? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | Switching costs are trivial (5 bps one-way trims the Sharpe advantage only 0.13→0.08), so a **tax-deferred** account really does buy a calmer ride. But there is no net Sharpe *outperformance* to bank: a modest **15% short-term tax flips the advantage negative** (−0.09), 25% takes it to −0.29, and the edge decays across eras. Real-but-thin & tax-eaten → **Fragile**, not Investable (the drawdown protection survives costs, so not a Mirage). |

> **In one sentence:** a 200-day trend filter on the 60/40 book **roughly halves the
> drawdown and volatility** and lifts the Sharpe on paper — but it does so by *giving up
> return*, the Sharpe edge is statistically indistinguishable from zero and rests on 2008,
> and a normal short-term tax bill erases it: **real risk cut, no bankable free lunch.**

## What we tested

Lay Faber's 200-day moving-average trend filter over the **static 60/40** (SPY/IEF): hold
each leg while it is above its own 200-day MA, step it to **BIL** cash otherwise — timing
the equity *and* the bond sleeve independently, at fixed 60/40 target weights. Real tape:
**yfinance daily total-return SPY/IEF/AGG/BIL, 2007-05-30 → 2026-06-30** (BIL's 2007 launch
sets the window — a short, one-crash-each sample, named on the **Signal** axis). We grade it
**excess-of-cash vs excess-of-cash** against the static book: a HAC *t* on the return
difference, a **paired block-bootstrap** CI for the Sharpe advantage, a two-era cut, a
calendar table, a switching-cost grid, a short-term-gains **tax drag**, and a 12-seed
synthetic positive control (a planted deep-bear world the filter can duck; a flat null it
cannot). **Dedup:** [110-faber-timing](../110-faber-timing/) is the **single-asset** equity
rule (in/out of cash), not a two-leg book; [97-balancing-act](../97-balancing-act/) is the
**static** 60/40 itself, not an overlay on it; [592-dual-momentum-gem](../592-dual-momentum-gem/)
**rotates between** assets on relative momentum (this filters each leg's *own* price with no
rotation); [626-unemployment-trend-timing](../626-unemployment-trend-timing/) times on a
**macro** trend, not price. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a trend filter *should* de-risk a balanced book — and why "cuts drawdown" is true but "keeps the Sharpe" is not robust |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the excess-vs-excess race, the paired Sharpe-advantage bootstrap, the two-era cut, the calendar table, the cost grid, the tax drag, and the synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`trend6040/`](trend6040/). Real tape: yfinance daily total-return closes
(`auto_adjust=True`), cached under `_cache/` (BIL from 2007 → short one-crash history,
named on Signal). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
