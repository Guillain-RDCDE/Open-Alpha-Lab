# Study 934 — Lump Sum vs DCA 💸

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the lump sum's advantage real? | ![Real](https://img.shields.io/badge/Real-2ea44f?style=flat-square) | Over **217** start months the lump sum finishes richer **76.0%** of the time (Wilson CI [69.9%, 81.2%]) and by **+5.05 cents per invested dollar** — HAC *t* **+3.19**, non-overlapping *t* **+2.18**, bootstrap CI **[+1.57, +7.90]**. Same sign in both eras, in all five conditional cuts, and on the 2000-2026 history (*t* = **+3.25**). |
| **Tradability** — is it an edge you can bank? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | It is **beta, not timing**. DCA's average exposure is only **13/24** of a lump sum's, so race it against a *static* 54.2% stock / 45.8% bill portfolio and the whole gap disappears: **−0.04c**, *t* = **−0.08**, CI **[−1.18, +0.98]** (reward per unit dispersion 0.651 vs 0.608, both excess of the same cash). It also dies where the premium dies — **+0.67c, *t* = +0.26** across the 2000s, and **+1.02c** with a CI through zero on bonds (IEF). |
| **Does DCA lower risk?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | Genuinely: dispersion ratio **0.587**, worst window **−36.1%** vs the lump sum's **−45.9%**. But the same 54.2% static portfolio lands with near-identical dispersion (**0.0926** vs DCA's **0.0999**) *and* identical terminal wealth — the calm comes from the weight, not from the twelve tranches. |

> **In one sentence:** drip-feeding a windfall over a year costs about **five cents on every dollar** and loses three times in four — but that bill is not the price of bad timing, it is the price of **owning half as much stock for a year**, and once you match the exposure the twelve tranches are worth exactly nothing, in either direction.

## What we tested

$1, twelve months, two ways in, both valued on the **same** terminal date: all-in at the
start, versus twelve equal month-end tranches with the waiting balance parked in **BIL** at
its *actual* total return. Every start month of SPY∩BIL 2007-05-30 → 2026-06-30, one
execution lag (month-end close, filled next day), 1 bp one-way, no shorting. Wilson
interval, HAC *t*, block bootstrap — then the control that sets the second stamp: an
**exposure-matched** race against a static portfolio holding DCA's own analytic average
weight, both arms read excess of the same cash leg. Plus hindsight cuts on valuation (a
**price PROXY**, not CAPE) and drawdown, era and decade cuts, a long-history extension
under an explicit **0% cash assumption** (pinned start — the shared cache's SPY depth
moves), cost / ticket / tranche sweeps, and an **IEF** variant.
**Dedup:** **101-slow-and-steady** asked the DCA-side question on daily windows with idle
cash pinned at **0%**; here the cash leg is real T-bills, starts are the month-ends a drip
follows, and the exposure control is new. **241-buy-the-dip** is a *conditional* entry
rule, not a schedule; **97-balancing-act** / **102-free-rebalance** manage a portfolio you
already hold rather than the one-off decision of how to get in.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | who finishes richer and by how much, why it is not a market view, what you are really buying when you average in, the "but surely when it's expensive?" test |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | overlap-corrected inference, the exposure-matched control, dispersion decomposition, conditional / era / decade cuts, the long-history extension, cost-ticket-tranche sweeps, the two-sided synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`lump_vs_dca/`](lump_vs_dca/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
