# Study 681 — Relative-Rotation-Graph 🔄📊

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the quadrant tell you which sector to buy? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Selection is indistinguishable from a matched-random pick of the same monthly count (*t* = **−0.35**), doesn't beat the plain 6-1 momentum sort it claims to improve on (*t* = **−0.93**, wrong sign), and underperforms the naive equal-weight basket by a **certified** −2.96%/yr (*t* = **−2.07**) — a real drag, not a real edge. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | **77% one-way monthly turnover**, the worst Sharpe (0.404) of four books even before the trailing costs get worse, and a certified 3.7–4.8 pt/yr shortfall to equal-weight in the two most recent decades (2010s, 2020s). Nothing to trade. |
| **Does the chart actually rotate clockwise?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | Pooled across 2,195 quadrant changes, sectors move in the claimed **Leading → Weakening → Lagging → Improving** direction **61.7%** of the time vs a 33.3% random baseline, and immediate reversals are rare in 3 of 4 states. The picture is descriptively honest — it just doesn't predict *which* sector to hold. |

> **In one sentence:** the Relative Rotation Graph's own quadrant story — sectors really do rotate clockwise through Leading/Weakening/Lagging/Improving more often than chance (62% vs 33%) — turns out to be true, but buying "Leading" earns no edge over picking the same number of sectors at random (*t* = −0.35), trails a plain one-dimensional 6-1 momentum sort, and certifiably underperforms just holding all 11 sectors equally (*t* = −2.07) at 77% monthly turnover: a real chart, a Mirage strategy.

## What we tested

Julius de Kempenaer's **Relative Rotation Graph** (RRG Research / StockCharts / Bloomberg
`RRG<GO>`): plot each sector's **RS-Ratio** (relative-strength level vs SPY) against its
**RS-Momentum** (the rate of change of that level) — four quadrants, a claimed clockwise cycle,
and a trading rule: buy sectors in **Leading**, avoid **Lagging**. We build both axes ourselves as
rolling z-scores (63-day level, 21-day rate-of-change — RRG vendors don't publish their exact
constants), go long the equal-weighted Leading quadrant each month on the 11 SPDR sector ETFs vs
SPY, and race it against SPY, an equal-weight sector basket, a plain 6-1 top-3 momentum sort (the
1-D signal RRG claims to beat), and — the decisive test — a Monte-Carlo matched-random control
that picks the *same number* of random sectors RRG held each month, isolating quadrant-selection
skill from the mechanical cash-timing drag of "go to cash when nothing is Leading." A 20-seed
synthetic positive control with a tunable planted relative-drift proves the harness is powered.
**Dedup:** [225-sector-rotation](../225-sector-rotation/) is the same universe with a plain 1-D
momentum sort (used here as the direct control); [506-industry-momentum](../506-industry-momentum/)
races the same sectors long-short against single names; [246-defensive-sectors](../246-defensive-sectors/)
is a two-sector risk-off timing canary, not an 11-sector rotation rule. None of them build the
RRG's own two-axis quadrant construction — this study is the first to test it directly.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the quadrant chart shows, why the rotation story is visually true, and why "Leading" still can't tell you which sector wins |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the RS-Ratio/RS-Momentum construction, the matched-random control, the cost sweep, the sub-period breakdown, the transition-matrix myth-check, and the synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`relative_rotation_graph/`](relative_rotation_graph/). No survivorship — all 11 sector
ETFs are used throughout their own live history (XLRE from 2015, XLC from 2018, named). **Not
investment advice** — research & education. See [LICENSE](../../LICENSE).*
