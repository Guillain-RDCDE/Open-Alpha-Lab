# Study 380 — Curve-Roll-Down 🛝

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does riding the curve beat cash? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | A 5y duration sleeve out-earns cash by **+1.65%/yr** and the full-sample HAC *t* (**2.46**) just clears the bar — but it is **entirely the 1990–2009 secular bond bull** (*t* = **3.46**). Post-2010, with rates rising, the excess collapses to **+0.16%/yr** at *t* = **0.17**. The "edge" is the duration term premium harvested in a non-repeatable one-way rate decline — regime-survivorship, not a law. |
| **Tradability** — can you deploy it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | Costs are negligible (2 bps), but the sleeve is **pure duration risk**: **+5.1%/yr** when rates fall, **−2.7%/yr** when they rise, at a gross Sharpe of only **0.34**. The slope-timing overlay never separates from simply being long duration. A carry trade a single hiking cycle wipes out is not a NAV-scale free lunch. |
| **Free lunch?** | ![Busted](https://img.shields.io/badge/Free_lunch%3F-Busted-8b949e?style=flat-square) | Roll-down is sold as a **static-curve** certainty; the curve takes it back the moment rates move. Promised **+4.76%** roll+carry, realized total **+4.33%**, and one rising-rate year flips the excess negative. It's the **term premium in a costume**. |

> **In one sentence:** "riding the yield curve" earns the duration term premium — real on average and statistically borderline (HAC *t* = 2.46) — but that whole edge is the 1990–2009 secular bond bull (*t* = 3.46 then, *t* = 0.17 since 2010), it is pure duration risk that loses 2.7%/yr when rates rise, and its promised "roll-down" is a static-curve fiction a single hiking cycle erases, so it is weak-as-signal, fragile-as-trade, and a busted free lunch.

## What we tested

True constant-maturity Treasury yields **are** free on yfinance (`^IRX`/`^FVX`/`^TNX`/`^TYX`), so we read the real curve over **36.5 years** and build a **5-year duration sleeve**. We compute the textbook *promised* roll + carry from each day's curve, then the **realized** total return once the curve actually moves a year later (`y0 − D·Δy`), and the **excess over cash**. Inference uses a **Newey-West HAC** *t* on the overlapping annual windows (a naive *t* over-counts by ~√overlap), a steep-curve placebo, regime splits, and a deterministic synthetic control with a planted slope→edge knob that confirms a bare carry baseline is *not* a predictive edge. (Same carry-as-premium / free-lunch-busted pattern as [Study 364 — FX-Carry-Trade](../364-fx-carry-trade/), and the slope-timing sibling of [Study 132 — Yield-Curve-Steepener](../132-yield-curve-steepener/).)

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "rolling down the curve" actually means, why it looks like free money, and why one year of rising rates takes it all back — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | promised-vs-realized roll+carry, realized excess with a Newey-West HAC *t*, the secular-bull sub-period cut, a steep-curve placebo, costs, and a synthetic carry-baseline / power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`curve_roll_down/`](curve_roll_down/). The 5y sleeve and 4-pillar curve are explicit approximations of a par bond and the full CMT curve. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
