# Study 625 — Starting-Yield-Bond-Decade 📜

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the yield you buy at pin your bond decade? | ![Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square) | On **15 non-overlapping decades** (1871–2021, Shiller GS10 + ^TNX splice): **R² = 0.923**, slope t = **+12.50**; slope indistinguishable from **1**, intercept from 0 (the duration-arithmetic identity). Worst of all 120 anchor phases: R² 0.865, t 8.75; overlapping windows HAC(120) t = **+32.71**; both century-halves hold. Naive rule "decade return = starting yield" misses by only **0.64 pp/yr**. Caveat: one country's never-defaulted yield path. |
| **Tradability** — can you deploy it? | ![Investable](https://img.shields.io/badge/Tradability-Investable-2ea44f?style=flat-square) | One purchase per decade: TreasuryDirect commission **$0**, ~2 bp one-way spread → **0.20 bp/yr** drag (~319× below the forecast error), >$25T capacity, long-only. But be clear what you buy: a **known *nominal* decade** — beta, not alpha — and nominal only (vs *real* returns the R² collapses to **0.24**). |
| **Works for stocks too?** | ![Busted](https://img.shields.io/badge/Works_for_stocks%3F-Busted-8b949e?style=flat-square) | Same design on 1/CAPE vs forward 10y nominal S&P total return: **R² = 0.205**, decade t = **1.76** (< 2), MAE **4.57 pp/yr** (7× the bond error). The arithmetic is bond-specific; the stock-side repair is [study 120's ECY](../120-excess-cape-yield/) (R² 0.70 — a statistical regularity, not an identity). |

> **In one sentence:** the 10-year yield on the day you buy really is your next bond decade —
> R² **0.92** with slope ≈ 1 on 15 non-overlapping decades since 1871, tradable for 0.2 bp/yr
> in the deepest market on earth — but it locks the **nominal** decade only (the 2020 cohort's
> 0.93% start is playing out as promised, via a −23% drawdown), and the trick does **not**
> transfer to stocks.

## What we tested

Bogle's "Occam's razor" claim (1991; updated 2015) and Leibowitz's duration-targeting
convergence: the entry yield explains ~90% of a constant-maturity Treasury portfolio's next
decade. We build a constant-maturity 10-year **nominal** total-return roll from Shiller's
monthly GS10 (1871→2023) spliced with ^TNX (→2026-06), Swinkels-2019-style closed-form pricing,
one documented lag (month-*t* average yield → enter end of month *t*). **Primary test:**
OLS on non-overlapping decades (unit = decade, as in [study 120](../120-excess-cape-yield/)) —
R², slope t, and the identity test slope = 1 / intercept = 0 — plus a full 120-phase anchor
sweep. **Secondary:** all 1,746 overlapping windows with a Newey-West (120-lag) t. We autopsy
the **2020 cohort** (0.93% start: −2.21%/yr realised so far through the 2022 crash — the
identity converging, not failing), show the lock is **nominal only** (real-return R² 0.24),
and run the same design on stocks (fails — that repair is study 120's ECY). A deterministic
synthetic control (same pricing engine, tunable mechanics/noise blend, nulls averaged over
20 seeds) proves the machinery. As-of **2026-07-03**, last complete month **2026-06**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a bond decade is arithmetic and not forecasting, the see-saw that makes the 2022 crash *part of the promise*, what the 2020 buyer was told and got, and why stocks don't work this way — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | non-overlapping decade OLS + identity test, 120-phase anchor sweep, HAC(120) overlapping regression, sub-period splits, the nominal-vs-real collapse, the stock comparison, and the seed-averaged synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`starting_yield_bond_decade/`](starting_yield_bond_decade/). Sibling of
[study 120 — Excess-CAPE-Yield](../120-excess-cape-yield/) — same decade-unit design, bond side.
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
