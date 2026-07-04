# Study 591 — Vol-Managed Portfolio 🌡️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does 1/RV scaling earn alpha? | ![Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square) | Real on the US mega-cap legs, unsupported elsewhere: SPY alpha **+2.87%/yr** at NW ***t* = 2.00** (at the wire) and QQQ **+5.49%/yr** at ***t* = 2.61**, shuffled-signal placebo **p = 0.030** (200 seeds) — but **EFA *t* = 0.84 and IWM *t* = 0.72**, and the long-only 1.0× variant sits at *t* = 1.44. Sharpe 0.717 vs 0.606, excess-vs-excess, gross. No survivorship (single index ETFs). |
| **Tradability** — can you deploy it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | Turnover is tiny (0.21 \|Δw\|/mo) and capacity unlimited — but the alpha *needs the 1.5× levered half* of the calendar, and a realistic retail margin spread decertifies it (**+2.46%/yr, *t* = 1.72** at 5 bps + 1% borrow; *t* = 1.44 at 10 bps + 2%). The Sharpe gain survives; the certified alpha doesn't. |
| **Does it dodge crashes?** | ![Mixed](https://img.shields.io/badge/Dodges_crashes%3F-Mixed-8b949e?style=flat-square) | Only the ones that announce themselves: GFC **−31% vs −51%** and 2022 **−16% vs −20%** dodged (vol was already high going in) — COVID 2020 **−18.2% vs −19.4%**, no dodge (entered ≈1.5× levered off a record-calm January; a monthly signal can't turn that fast). |

> **In one sentence:** scaling SPY by last month's inverse realized variance really does raise the Sharpe (0.72 vs 0.61) and its Moreira-Muir alpha clears the bar at the wire on US mega-caps (*t* = 2.00 SPY, 2.61 QQQ; placebo p = 0.03) while failing on EFA/IWM — and since the edge lives in the levered half and dies under a retail borrow spread, it grades a real-but-Mixed signal in a Fragile vehicle that dodges slow-burn crashes but not sudden ones.

## What we tested

Moreira & Muir (2017, JF) claim volatility-managed portfolios — weight = c/RV(previous month), monthly rebalance — beat their unmanaged selves because **variance is forecastable while the equity premium is not**. We rebuild it honestly on SPY 1993→2026 (QQQ/EFA/IWM robustness): a **past-only expanding normaliser** (no ex-post variance matching), a **1.5× cap**, **exactly one execution lag**, **excess-vs-excess** races off ^IRX, and a **Newey-West alpha regression** of managed-on-unmanaged with appraisal ratio, a **200-seed shuffled-RV placebo**, crash-window drawdowns (2008 / 2020 / 2022), and a cost sweep charging turnover *and* a retail borrow spread on the levered fraction. A three-world synthetic control (risk-priced null must earn nothing; planted leverage-effect world must light up — it does, mean *t* = 2.09 over 20 seeds) proves the machinery. Distinct from [06-clockwork-vol](../06-clockwork-vol/) (fixed-period vol *cycles*) and [130-vol-risk-premium](../130-vol-risk-premium/) (the *IV−RV options* spread): this is the Moreira-Muir **scaling** strategy — last month's realized variance deciding how much of the same asset you hold. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why holding *less* in scary months can beat holding everything always, the thermostat chart, which crashes it dodged (and the one it walked straight into) — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the HAC alpha regression + appraisal ratio, the shuffled-signal placebo, asset & cap robustness, the borrow-spread cost sweep, and the three-world synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`vol_managed_portfolio/`](vol_managed_portfolio/). The signal is last month's realized variance; the myth-check is the crash-dodging claim. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
