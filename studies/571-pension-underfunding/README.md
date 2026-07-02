# Study 571 — Pension-Underfunding 🕳️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do underfunded-pension firms earn lower returns? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | **No real tape.** The point-in-time pension-footnote data the anomaly needs (PBO, plan assets, funded status / market cap) is not reachable from a no-key retail stack, so the desk's bar — *a robust t ≥ 2 on a **real** tape* — cannot be met. On the synthetic world the engine cleanly recovers the planted puzzle (slope-*t* **−4.64**, IC **−0.26**, quintile spread **+8.79%/yr** at *t* **2.45**, placebo *p* **0.006**) and stays flat at the null (slope-*t* **+0.01**). Machinery proof, not a claim on the tape. |
| **Tradability** — could you harvest it? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The short leg is deeply-underfunded, stressed old-economy names — exactly the hard-to-borrow tail (150 bps borrow eats ~1.7pp: gross **+8.79%** → net **+7.09%** even in the synthetic world). And there is **no free way to source the signal**: you cannot build the funding measure without Compustat pension items. |

> **In one sentence:** the Franzoni-Marin (2006) pension-underfunding anomaly — firms carrying the
> biggest off-balance-sheet pension holes earn *anomalously low* returns because the market
> under-reacts to the hidden leverage — is a real, published effect, and this study proves the
> engine would catch it (a clean **−4.64** slope-*t* and **+8.79%/yr** spread on a planted world),
> but the point-in-time pension-footnote data is **not free**, so there is no real tape to certify
> it on: `NONE` (not evidenced on a real tape) × `MIRAGE`.

## What we tested

The **pension-underfunding anomaly** (Franzoni & Marin 2006): a firm's defined-benefit pension plan
can be badly underfunded — the projected benefit obligation (PBO) exceeds plan assets — and that
shortfall is a senior, off-balance-sheet liability. Franzoni-Marin find the market *under-reacts* to
it, so the most-underfunded firms earn the *lowest* subsequent returns (≈−7%/yr). We build the
funding measure `funding_gap = (assets − PBO)/mktcap`, sort firms by the depth of the hole into
quintiles, and test whether well-funded names beat underfunded ones: a two-sample *t* on the
well-minus-underfunded forward-return spread, a **label-shuffle placebo** null, a firm-level slope /
IC (whose *sign* is the puzzle), costs + a punitive short borrow, a four-design robustness sweep,
and a deterministic, seed-robust synthetic positive control that plants the effect and proves the
engine catches it (and stays flat at the null). **This study is synthetic-only** — the real
point-in-time pension-footnote panel is not reachable for free, so the SIGNAL axis is capped at
`NONE` and the data-availability limitation is named openly (like the desk's
[273 Lego-Returns](../273-lego-returns/), [275 Whisky-Cask](../275-whisky-cask/) and
[276 Sneaker-Resale](../276-sneaker-resale/)). *Distinct from the on-balance-sheet
[154 Leverage-Anomaly](../154-leverage-anomaly/) and the failure-probability
[540 Distress-Risk-Anomaly](../540-distress-risk-anomaly/): this is the **pension hole** as hidden
leverage.*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a pension hole is, why "big hole → lower returns" is a puzzle, and why we can only test it on a synthetic world |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the quintile sort with a two-sample *t*, the placebo null, the firm-level slope/IC, the robustness sweep, costs + borrow, and the seed-robust synthetic positive control |

The fingerprinted reproducible run (synthetic panel, `underfunding_alpha = -0.05`, seed 571, 300
firms, panel fp `b43234ffbd01`, as-of 2026-06-30) is in [docs/results.md](docs/results.md); the
offline machinery runs entirely on the deterministic synthetic world in
[`pension_underfunding/data.py`](pension_underfunding/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`pension_underfunding/`](pension_underfunding/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
