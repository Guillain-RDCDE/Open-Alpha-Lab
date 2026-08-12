# Study 896 — Risk-Parity + Trend 🔀

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a 200d trend gate improve risk-parity's Sharpe *and* drawdown? | ![Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square) | *Real on the drawdown, weak on the Sharpe.* Full-sample max DD **−9.68% vs −19.95%** (roughly halved), robust across **both** sub-eras and every crisis (2008 −19→−8%, 2020 −17→−7%, 2022 bond bear −20→−9%), and certified as genuine **timing** by a 200-seed shuffled-gate placebo (**p = 0.000** — same-frequency random gates average −17% DD). But the return leg never clears the bar: Sharpe advantage **+0.077** with a paired-bootstrap 95% CI **[−0.25, +0.39]** straddling zero, excess-return diff **−0.93%/yr** (HAC *t* = −0.65), and a sign that **inverts** across eras (+0.181 → −0.027). Four young, hand-picked sleeves on one 18-year survivor tape — named. |
| **Tradability** — can you deploy it as an edge? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | Mechanically easy: turnover **2.40× NAV/yr**, penny-spread ETFs (~0.1%/yr at 5 bps), long-or-cash (no shorts/borrow), and every number survives **20 bps** (Sharpe adv +0.031, DD shield intact). But what survives certification is **risk control, not excess return**: you give up a little growth (terminal wealth **×2.97 vs ×3.39**) for a much smoother path, and the uncertified Sharpe edge rests on one crisis-front-loaded tape. A real shield, not a bankable edge. |

> **In one sentence:** bolting a per-sleeve 200-day trend gate onto an inverse-vol
> risk-parity book (SPY/TLT/GLD/DBC, de-risked sleeves parked in T-bills) **genuinely
> halves the drawdown** — robust across eras and crises, and a 200-seed placebo proves
> it's timing, not just holding less — but the Sharpe-improvement half of the pitch is
> **not certified** (advantage +0.077, bootstrap CI through zero, sign flips era-to-era),
> so it grades a **Mixed** signal in a cheap-to-run **Fragile** vehicle.

## What we tested

The multi-strategy staple of stacking **risk parity** (the diversified core) with
**trend following** (the crisis de-risker). We build study 68's inverse-vol risk-parity
book across **SPY / TLT / GLD / DBC** (weights ∝ 1/trailing-60d-vol, monthly rebalance),
then gate each sleeve with its **200-day moving average**: hold the sleeve only while it
is above its 200d line, otherwise that sleeve's risk budget sits in **BIL** T-bills for
the month — **exactly one execution lag** (yesterday's prices set today's gate). Tape:
yfinance daily **total-return** ETFs, 2008-04 → 2026-06 (4,591 race days after burn-in,
ann = 252). We race RP+trend vs plain RP **excess-of-cash on both legs** (minus BIL):
Sharpe, max DD, a **HAC *t* on the daily excess-return difference**, and a **paired
block-bootstrap of the Sharpe difference**; a **two-era cut** and a **per-crisis
drawdown ledger** check robustness; a **200-seed shuffled-gate placebo** splits *timing*
from *mere de-risking*; a cost sweep charges one-way bps × turnover × NAV (no borrow — the
book is long-or-cash). A seeded synthetic control (a no-downtrend null must earn nothing;
a planted bear-regime world must light up — mean Sharpe adv +0.254, 90% of 20 seeds)
proves the machinery. **Dedup:** distinct from [68-all-weather](../68-all-weather/) (the
**plain** risk-parity book this study gates), [110-faber-timing](../110-faber-timing/)
(the same 200d rule on a **single asset**), [595-managed-futures-allocation](../595-managed-futures-allocation/)
(a trend sleeve **added** to a portfolio, not gating the sleeves in place),
[894-trend-6040](../894-trend-6040/) (trend on a **60/40** budget, not a risk-parity one),
and [656-dragon-portfolio](../656-dragon-portfolio/) (a fixed offense/defense *cocktail*
including a trend sleeve, not a trend gate on a risk-parity book). As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what risk-parity + a trend switch means, the growth curve, the crisis-by-crisis drawdown ledger (each ~halved), the year-by-year "wins in crises, drags in bull markets" bars, and the honest price tag |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the excess-vs-excess Sharpe race, the paired Sharpe-difference bootstrap (CI [−0.25, +0.39]), the two-era cut (sign flip), the 200-seed shuffled-gate placebo (Sharpe p = 0.135 · DD p = 0.000), the cost sweep, and the two-world synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`rp_trend/`](rp_trend/). The signal is a per-sleeve 200-day SMA gate on top of
an inverse-vol risk budget (one month of lag); the myth-check is whether trend improves
risk-parity's Sharpe as well as its drawdown. **Not investment advice** — research &
education. See [LICENSE](../../LICENSE).*
