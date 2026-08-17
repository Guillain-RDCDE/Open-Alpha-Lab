# Study 949 — Riding the TIPS Curve 🪜

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the roll-down carry real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Extending along the real curve paid **nothing**: excess-of-cash Sharpe falls monotonically **+0.256 → −0.001** from VTIP to LTPZ, every long-minus-short difference is negative and insignificant, every Sharpe CI straddles zero. The duration-hedged residual is positive in all four buckets and significant in **none** — best **full-sample** HAC *t* **+1.55** gross, **+0.94** net, bootstrap CI [−0.66%, +1.73%] — it *shrinks* as you extend, and across the **7.2 pre-shock years** it is **+0.18%/yr gross (*t* = +0.19)** against **+2.36%/yr net in 2021-2023**. **1 of the 56 *t*s here reaches 2** (**+2.04**, gross, inside the hand-picked shock window; Bonferroni bar ≈3.2) — the rival hypothesis on cue, not the claim surviving. **Survivorship:** nil in the usual sense (four live ETFs, none delisted); the real selection is the **inception gate** — VTIP's 2012 launch confines the study to the post-GFC real-rate regime. |
| **Tradability** — is it bankable? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The best net figure anywhere is **+0.84%/yr** (504-day beta, *t* = +1.24); the base case pays **+0.60%/yr at 2.16% vol**. Both come from a spread that is short a Treasury fund every day: it turns negative at **~100 bps/yr borrow** and collapses to **+0.17%/yr (*t* = +0.26)** if you remove 2021 alone. The long-only version is worse — LTPZ compounded at **+0.49%/yr** against cash's **+1.59%**, for a **−41%** drawdown. |

> **In one sentence:** on 13.5 years of listed linkers there is **no roll-down carry in real yields** — real duration went entirely uncompensated (long TIPS under-earned T-bills while drawing down 41%), the part of a linker's return a duration-matched nominal hedge cannot explain never clears |*t*| ≥ 2 on the full sample and *falls* as you extend, and what little residual exists is one inflation shock: a **long-breakeven leg meeting a CPI surprise**, not a slope you were paid to ride.

## What we tested

Four inflation-linked maturity buckets — **VTIP** (~2.5y real duration), **SCHP** and
**TIP** (~7y), **LTPZ** (~20y) — raced **excess-of-cash** against **BIL**'s actual total
return, then each one hedged against a **duration-matched nominal Treasury fund** (SHY /
IEF / TLT) with the beta fitted on the trailing 252 days **through day *t*−1 and applied
at *t*** (the study's one and only execution lag). Daily **total-return** closes
(`auto_adjust=True`) over VTIP∩BIL 2012-10-16 → 2026-06-30. Newey-West *t*s, 21-day block
bootstraps, an era cut around the 2021-2023 inflation shock, a drop-2021 jackknife, cost
and **borrow** sweeps, alternative pairings and beta windows, a **multiplicity audit that
ranks all 56 *t*s** the study computes, and a synthetic control that recovers a planted
2%/yr carry at *t* = +3.65 while firing on 0/8 nulls. **Proxies:** the 30 bps/yr borrow,
the 2 bps one-way cost and the linker↔nominal pairing are assumptions, all swept.
**Dedup:** distinct from **380-curve-roll-down** (the same claim on the
*nominal* curve, built from constant-maturity yields, not a hedged fund residual),
**381-tips-breakeven** (breakeven as a *predictive signal*, where our residual is its
regressor), **868-global-curve-slope-carry** (cross-sectional, across countries), and
**886-agency-mbs-carry** / **906-em-local-hedged** (same hedge-out-the-factor method,
different factor).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "riding the curve" promises, why the long bucket lost to T-bills, the single year that is the whole result, and what borrow does to a permanent short |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the excess-of-cash ladder race, lagged rolling-beta hedge, HAC *t*s and bootstrap CIs, era cut (gross **and** net), jackknife, cost/borrow/pairing/window sweeps, the 56-*t* multiplicity audit, the live synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`tips_roll/`](tips_roll/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
