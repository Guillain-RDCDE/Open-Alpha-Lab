# Study 906 — EM Local Bonds FX-Hedged 🌏

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does stripping the FX leave a real local-rate carry? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The FX-strip is **mechanically real**: EMLC is ~50 % dollar-basket FX (β = −1.12, HAC *t* = −11.9), and the UUP overlay lifts the excess-of-cash Sharpe from **+0.03 → +0.20** and halves the drawdown (**−32 % → −17 %**), the same on LEMB/EBND. But the residual local carry is **+1.62 %/yr at HAC *t* = +0.94** — a bootstrap Sharpe CI of **[−0.21, +0.69]** straddling zero — and it **loses to just owning USD-EM debt** (hedged − EMB = **−1.48 %/yr**). Real mechanism, no robust premium. |
| **Tradability** — is it bankable? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Even *gross* the hedged carry (Sharpe +0.20) is dominated by the plain USD-EM ETF **EMB** (Sharpe +0.39); after the UUP-overlay re-strike cost (0.46 %/yr) net is **+1.15 %/yr** (*t* +0.67) and the net premium vs EMB is **−1.95 %/yr**. A simpler, cheaper ETF wins outright. |

> **In one sentence:** EM-local bonds really are ~half FX, and a dollar-index overlay genuinely
> strips that drag and halves the drawdown — but what's left is a **+1.6 %/yr local carry at
> *t* < 1** that a bootstrap can't separate from zero and that **loses to simply holding
> USD-EM debt (EMB)**, so the "hidden local carry" is a real *mechanism* with no bankable edge.

## What we tested

EM local-currency bond ETFs (**EMLC**, LEMB, EBND) earn `local_bond_return + EM_FX_return`;
the FX leg is so volatile that the fat local rate vanishes. No clean FX-hedged EM-local ETF
exists on US tape, so we build a **proxy hedge** — a long **UUP** (dollar-index / DXY-basket)
overlay, sized by the variance-min ratio `b = cov(EMLC−BIL, UUP−BIL)/var(UUP−BIL)` — and race
the **hedged** leg **excess-vs-excess** (minus BIL) against USD-EM debt (**EMB**) and cash.
Inference is **Newey-West HAC**, a circular-block bootstrap Sharpe CI, a two-era cut
(2021-01), a 36-month walk-forward hedge (no look-ahead), a costed overlay, and a
planted-carry synthetic control. **Proxy caveat, named on the Signal axis:** UUP tracks the
*developed-market* DXY basket, not the EMLC currency basket (BRL/MXN/IDR/ZAR/THB), so it
strips only the systematic dollar move and leaves a ~+0.10 residual EM-FX beta. **Dedup:**
distinct from [612-em-debt-carry](../612-em-debt-carry/) (the **USD**-EM sibling, no FX
question), [662-em-local-bonds](../662-em-local-bonds/) (the **unhedged** local premium),
[364-fx-carry-trade](../364-fx-carry-trade/) (currency carry *as* the trade), and
[889-dollar-hedge-overlay](../889-dollar-hedge-overlay/) (a dollar overlay on a different
underlying) — here the overlay is specifically an **EM-FX proxy hedge** on local-EM bonds.
As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why EM-local yields look fat, how the currency eats them, what a dollar-index overlay does, and why the leftover carry still loses to the boring USD-EM ETF — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the EMLC~UUP hedge regression, the excess-vs-excess race, HAC *t*'s, the bootstrap Sharpe CIs, the walk-forward hedge + residual EM-FX beta, era splits, drawdowns, the costed overlay, and the planted-carry synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`em_hedged/`](em_hedged/). The hedge is a long-UUP (DXY-basket) **proxy** overlay
on EM-local bonds — a deliberate developed-market stand-in for an EM-FX hedge, labelled
throughout. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
