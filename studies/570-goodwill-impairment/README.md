# Study 570 — Goodwill-Impairment 🧾

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

**Can a bloated goodwill balance predict the write-down — and the stock drop — before it happens?**

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does high goodwill/assets predict impairments and negative returns? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | **No real tape exists** for a no-key retail stack (no point-in-time goodwill panel, no tagged impairment events), so this is **synthetic-only** and capped at `WEAK` — `REAL` needs *t* ≥ 2 on a *real* tape. The literature leans the right way (Li et al. 2011; Hayn–Hughes 2006) and the engine is faithful: on the planted world it recovers a low-minus-high spread **+5.78%** (Welch *t* **+3.86**, placebo *p* **0.0005**), firm slope-*t* **−4.22**, and stays flat at the null (mean slope-*t* **−0.05** / 25 seeds). |
| **Tradability** — could you trade it? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Nothing to trade: there is no real result, only a validated detector. The high-goodwill short leg (recent over-payers, smaller, corporate-action-prone) would carry a hard borrow and thin capacity on top of a signal that has never touched a real tape. Synthetic net **+4.08%** after 5 bps/leg + 150 bps borrow — but net-of-nothing-real is a mirage. |

> **In one sentence:** the overpaid-acquisition idea — a bloated goodwill/assets ratio predicting the write-down *and* the drop — is coherent, literature-backed, and cleanly caught by the engine on a planted world (spread *t* +3.9, firm slope-*t* −4.2, flat at the null), but there is **no point-in-time goodwill / impairment tape** a no-key retail stack can reach, so it lands `WEAK` × `MIRAGE`: a real idea and a working detector, with an honest data wall between them and a tradable claim.

## What we tested

The **overpaid-acquisition** claim (accounting): a high **goodwill / total-assets** ratio flags a
firm that paid up for acquisitions; the excess purchase price sits on the balance sheet until an
impairment test forces a **write-down**, and the market's slowness to price it leaves high-goodwill
firms with a higher **impairment rate** and **lower** forward returns. Because a point-in-time
goodwill panel with tagged impairment events is **out of reach** for a no-key retail stack, this is
a **synthetic-only** study: a deterministic, seeded panel (`goodwill_impairment/data.py`) with one
knob (`ret_alpha`) that plants the return drag while a second (`imp_beta`) plants the impairment
link. We build the signal (goodwill/assets), a quintile sort with a two-sample *t*, a
**label-shuffle placebo** null, a two-proportion *z* on the impairment-rate gap, a firm-level slope
(whose *sign* is the puzzle), costs + a punitive high-goodwill borrow, a bucket-fraction robustness
sweep, and a seed-robust synthetic positive control (25 seeds) that recovers the planted effect and
stays flat at the null. *The data-availability limitation is named on the Signal axis; a
synthetic-only study can never be `REAL`. Distinct from the desk's accruals anomalies
([231](../231-sloan-accruals/), [522](../522-percent-operating-accruals/)) — this is the goodwill /
overpaid-M&A channel, not accrual reversal.*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what goodwill is, why a bloated balance means "we overpaid", how the write-down and the drop follow, and why we can only test it on a synthetic world |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the quintile sort with a two-sample *t*, the placebo null, the impairment-rate gap (two-prop *z*), the firm-level slope, the bucket-fraction robustness sweep, costs + borrow, and the seed-robust synthetic positive control |

The fingerprinted reproducible run (synthetic panel, `ret_alpha=-0.14`, seed 570, panel fp
`6bd491ebb228`, as-of 2026-06-30) is in [docs/results.md](docs/results.md); the offline machinery
runs entirely on the deterministic synthetic world in
[`goodwill_impairment/data.py`](goodwill_impairment/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`goodwill_impairment/`](goodwill_impairment/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
