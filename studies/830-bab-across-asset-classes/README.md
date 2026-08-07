# Study 830 — BAB Across Asset Classes ⚖️🌐

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does betting-against-beta earn an alpha *across asset classes* (Frazzini-Pedersen "everywhere")? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | On nine liquid asset-class ETFs (2007–2026) the beta-neutral BAB factor earns **+0.54 bps/day** (Newey-West *t* = **+0.31**); even as a CAPM alpha only **+2.86 bps/day** (HAC *t* = **+1.61**) — neither clears \|t\| ≥ 2. The book's *realized* market beta is **−0.83**: the multi-asset "low-beta" long leg is dominated by Treasuries and gold, so this is a disguised **long-duration / short-equity** tilt, not a clean SML arbitrage. It worked only weakly pre-2016 (alpha *t* = +1.95) and **reversed** after (−0.25), and a 1,000-permutation placebo shows the beta sort adds nothing beyond the mechanical net-long leverage tilt (observed +0.54 vs cloud mean +3.52, two-sided p = 0.98). A 20-seed synthetic control recovers a *planted* flat-SML premium cleanly (fires **1/20** nulls ≈ the nominal 5%). *Survivorship: current-membership ETFs — magnitudes are an upper bound.* |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Flat gross, flat net: **+0.37 bps/day** at 1 bp one-way (*t* = +0.17), **+0.09** at 5 bps — indistinguishable from zero at any cost, on a **2.6×-gross** levered book turning over 0.07 of NAV/day. |

> **In one sentence:** Frazzini-Pedersen's "betting against beta everywhere" **does not survive**
> at the multi-asset level — a beta-neutral long-low / short-high book across nine asset classes
> earns no significant alpha (NW *t* = +0.31), is really a disguised long-duration bet (realized
> beta −0.83), and pays nothing net, so the honest read is **claimed signal absent, paycheck a
> mirage**.

## What we tested

Frazzini & Pedersen (2014), **"Betting Against Beta"**, lifted to the **multi-asset** level:
the flat security-market line is claimed to hold *everywhere*, so ranking whole asset classes
by their beta to a common market and going **long low-beta (levered to unit beta) / short
high-beta (de-levered to unit beta)** should earn a positive alpha. We build it on **nine
liquid asset-class ETFs (yfinance daily total-return, 2007-04-11 → 2026-06-30)** — SPY, EFA,
EEM, TLT, LQD, HYG, GLD, DBC, VNQ — with each asset's **Frazzini-Pedersen rolling beta** to
their equal-weight market (ρ over 252d × σ ratio over 63d, shrunk toward 1), formed
point-in-time (beta known at the close of `t−1`, one shift, zero look-ahead), a Newey-West *t*
on the daily factor, a HAC CAPM alpha, a 1,000-permutation placebo, a two-era cut, a costed
levered timer, and a 20-seed synthetic positive control. The nine ETFs are a **current-membership**
survivor set — named on the **Signal** axis. **Dedup:** [238-betting-against-beta](../238-betting-against-beta/)
is BAB in the **single-stock** cross-section (the original); [660-carry-everywhere](../660-carry-everywhere/)
sorts asset classes on **carry**, not beta; [68-all-weather-risk-parity](../68-all-weather-risk-parity/)
is a long-only inverse-vol **allocation**, not a long-short beta-neutral factor. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a flat SML *should* pay a low-beta premium — and why across asset classes it collapses into a long-duration bet |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the FP rolling betas, the factor Newey-West *t* and HAC CAPM alpha, the 1,000-permutation placebo, the two-era cut, the levered cost math, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`bab_multiasset/`](bab_multiasset/). Nine asset-class ETFs pulled via yfinance
(current-membership → magnitudes are an upper bound). **Not investment advice** — research &
education. See [LICENSE](../../LICENSE).*
