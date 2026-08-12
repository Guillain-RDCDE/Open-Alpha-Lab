# Study 898 — Managed-Vol Equity 🎚️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a constant-vol thermostat on SPY raise the Sharpe (Moreira-Muir)? | ![Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square) | *Real on the tail control, weak on the Sharpe.* Max DD **−31.1% vs −56.5%** (COVID −13.7% vs −33.9%), robust across the whole 3×3 grid and both eras, and certified as genuine **timing** by a 200-seed shuffled-vol placebo (**p = 0.000** — same-distribution random weights average −57.4% DD); with average weight **0.96** it de-risks in storms, not just holds less. But the return leg never clears the bar: HAC alpha **+2.78%/yr at *t* = 1.67** (grid *t* 1.44–1.74, era *t* 0.96/1.08), and the +0.110 Sharpe advantage has a bootstrap CI **[−0.128, +0.355]** straddling zero. Single ~19-year SPY tape — short-history, named. |
| **Tradability** — can you deploy it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | Mechanically ideal: turnover **9.15× NAV/yr** (~1 bp ≈ 9 bps/yr), SPY capacity unlimited, and every number survives **5 bp + 1% borrow** (Sharpe 0.613, DD −31.4%). But what survives certification is **risk control, not excess return**: β = 0.56 < 1 means you *give up* ~1.1 pp/yr of excess CAGR for the smoother path, and the Sharpe uplift that would make it an edge never reaches *t* = 2. A real shield, not a certified paycheck. |

> **In one sentence:** scaling SPY to a constant ~12% vol target really does halve the
> heart attacks — the −56% drawdown becomes −31%, the thermostat holds vol near 12%
> (median 12.5%), and a 200-seed placebo proves it is genuine timing — but the "higher
> Sharpe" half of the Moreira-Muir pitch never certifies on this single 19-year tape
> (alpha *t* = 1.67, bootstrap CI straddles zero), so it grades a **Mixed** signal in a
> cheap-to-run but **Fragile** vehicle: a risk shield, not a bankable edge.

## What we tested

The equity-desk staple that ports **Moreira & Muir (2017), "Volatility-Managed Portfolios"**
to a single asset: hold `w = min(2.0, 12% / RV_21d)` of **SPY** and the rest in **bills (BIL,
a real total-return T-bill ETF)**, rebalanced daily with **exactly one execution lag** (the
weight for day *t* uses returns through close *t−1*). Tape: yfinance SPY + BIL total-return,
2007-05-30 → 2026-06-30 (4,802 daily closes, ann = 252). We race the managed book against
buy-and-hold SPY **excess-of-cash on both legs** (SPY − BIL): Sharpe, max DD, a **HAC alpha
regression** (the Moreira-Muir "did the Sharpe rise?" test), a HAC *t* on the return
difference, and a **leverage-timing decomposition** (`mean = β·exposure + α·timing`). A 3×3
**grid** (targets 10/12/15% × windows 21/42/63d) checks robustness; a **paired block-bootstrap**
puts a CI on the Sharpe advantage; a **200-seed shuffled-vol placebo** splits *timing* from
*mere exposure reduction*; a cost sweep charges one-way bps × |Δw| + borrow on the levered
fraction. A seeded synthetic control (risk-priced null must earn nothing — mean *t* = −0.01;
planted leverage-effect world must light up — mean *t* = +4.98, 97% of 30 seeds) proves the
machinery. **Dedup:** [16-storm-shy](../16-storm-shy/) is a **dip-buyer** that *adds* exposure
after down-days (opposite posture); [591-vol-managed-portfolio](../591-vol-managed-portfolio/)
is the **broad multi-factor** Moreira-Muir overlay (this is the single-asset SPY case with a
**real bill leg** + timing decomposition); [590-sharpe-hacking](../590-sharpe-hacking/) is the
vol-target Sharpe caveat we answer directly; [12-paper-prophet](../12-paper-prophet/) the
paper-vs-live reminder. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "targeting 12% vol" means, the thermostat, the heart-attack ledger (−31% vs −56%), and the honest catch — a smoother ride whose *higher-Sharpe* promise can't be proven — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the HAC alpha regression, the leverage-timing decomposition, the 3×3 grid, the paired Sharpe-gap bootstrap, the 200-seed shuffled-signal placebo (alpha p = 0.070 · DD p = 0.000), the cost sweep, and the two-world synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`managed_vol/`](managed_vol/). The signal is trailing 21d realized vol (one-day lag);
the myth-check is the Moreira-Muir "higher Sharpe" claim vs a real bill leg. **Not investment
advice** — research & education. See [LICENSE](../../LICENSE).*
