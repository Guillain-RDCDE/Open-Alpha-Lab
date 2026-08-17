# Study 951 — The Crossover Rung 🪜

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the BBB/BB boundary really the best-paid rung? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | The excess-Sharpe ladder does hump on the boundary (**+0.602** crossover vs +0.437 broad HY, +0.241 IG) and the sign is positive in every cut — but the robustness fails. Adjusted for duration and equity beta the boundary beats broad HY by **+2.51%/yr at HAC *t* = +2.01**, *on* the bar not past it: it survives only **5 of 15** one-year-out deletions and collapses to **+1.50%/yr (*t* = +0.92)** after 2019. Swapping both legs for their sibling funds gives **+0.63%/yr (*t* = +0.48)** — but that is mostly the same failure, not a new one: on that shorter window the headline pair itself only reads +0.96%/yr. The other half of the claim — beating *investment grade* — is absent: **+1.51%/yr, *t* = +1.15**, CI [−0.61%, +3.90%]. *Survivorship: six living ETFs; failed crossover funds are not on this tape. One US credit history, two downgrade waves.* |
| **Tradability** — can you bank the rung? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | Not a cost mirage: owning ANGL instead of HYG inside a sleeve you already hold is one switch, charged on both legs (sell HYG *and* buy ANGL, 2 × 5 bps one-way ≈ 0.007%/yr amortised), fees already inside the tape, and it still bought **+0.165** of excess Sharpe. Fragile because the price of admission is a **7.3 pp deeper drawdown** (−29.3% vs −22.0%) for a premium two years carry (2016, 2020 → drop both and *t* = +1.61), and the pure long/short spread dies on borrow: *t* = **+1.69** at 50 bps. |

> **In one sentence:** the credit ladder really does hump at the crossover rung — reward-per-risk peaks exactly on the BBB/BB boundary — but the premium is a **downgrade-wave payoff, not a rung**: it beats broad high yield only at the significance bar and only before 2019, it never beats investment grade once duration and equity beta are stripped out, and what you actually buy with the switch is a deeper drawdown wearing a premium's clothes.

## What we tested

Race the whole ladder — **AGG → LQD → ANGL → HYG** — on daily total-return closes, every rung
**excess-of-cash** (minus BIL) and then regressed jointly on **IEF** (duration) and **SPY**
(equity beta), so the winner is not simply whoever carries the most risk. The headline is the
two-factor alpha on the *return difference* between rungs, with HAC *t*, a joint block-bootstrap
CI, an era cut, a one-year-out jackknife, an HAC-bandwidth sweep, and the both-legs-swapped
**FALN − USHY** cross-check, over ANGL∩BIL 2012-04-11 → 2026-06-30 — all three head-to-heads on
that one common window. **Proxies, labelled:** the crossover rung is proxied by fallen-angel ETFs
(no free daily BBB−/BB+ tape exists); IEF/SPY are a two-factor adjustment, not a credit-factor
model, and their betas are **fitted in sample** (the fitted hedge is never traded — both
expressions hold raw funds); the short-leg borrow rate is an assumption, swept 0→100 bps.
**Dedup:** distinct from **[610-fallen-angels-premium](../610-fallen-angels-premium/)**
(monthly ANGL-vs-HYG *selection* claim, controls added one at a time, stamped Real) — 951 asks the
*ladder* question instead, adds the **ANGL − LQD** leg 610 never ran, and puts the shared leg
through a jackknife and a sibling-fund swap that downgrade it; from **115-credit-spreads** (spreads
timing *equities*), **832-high-yield-credit-momentum** (a credit *trend* timer),
**885-ultra-short-credit-pickup** (the cash-to-IG step) and **892-corporate-bond-ladder** (a
*maturity* ladder, not a quality one).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | who is forced to sell a downgraded bond, why the ladder humps on the boundary, the two years that carry the whole premium, and the drawdown you pay for it |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the two-factor alpha race, HAC *t* on the return difference, joint block-bootstrap CI, era cut, year jackknife, bandwidth sweep, borrow sweep, the live synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`crossover_credit/`](crossover_credit/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
