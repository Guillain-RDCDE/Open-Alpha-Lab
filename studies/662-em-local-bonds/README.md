# Study 662 — EM-Local-Bonds 🌍💱

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does local debt pay a better compensated carry than USD EM debt? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Not just unproven — **backward**. A Local basket (avg EBND/LEMB) trailed EMB (USD EM) by **−2.96%/yr** excess-of-cash, clearing **\|t\| ≥ 2** at every Newey-West lag (3/6/12mo: −2.02/−2.16/−2.17), block-bootstrap 95% CI **[−5.62%, −0.20%]** — wholly negative — and a 41.5% hit rate (Wilson excludes 50%). |
| **Tradability** — worth deploying on its own? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Local's own 14.5-year excess-of-cash Sharpe is **−0.023** — worse than EMB (+0.296) *and* AGG (+0.138) — for a **−27.5%** drawdown and a highly significant, uncompensated dollar beta (**−1.05**, *t* = −13.3). No edge here for costs to erode; the point estimate is already negative gross. |
| **"Extra yield eaten by currency depreciation?"** | ![Confirmed](https://img.shields.io/badge/Extra_yield_eaten%3F-Confirmed-8b949e?style=flat-square) | The isolated Local-minus-EMB spread vs the dollar clears decisively (β = **−0.358**, Newey-West *t* = **−3.93**): every point of dollar strength costs the spread ~36 bps beyond EMB's own loss. Named exception: 2022's duration shock hit EMB *harder* than Local despite the dollar's biggest rally of the sample — the full-sample regression is the reliable read, not any one crisis window. |

> **In one sentence:** local-currency EM bonds (EBND/LEMB) did not just fail to pay a better
> compensated carry than USD EM debt (EMB) over 2011–2026 — they trailed it by a robustly
> negative **−2.96%/yr** (Newey-West *t* ≈ −2.0 to −2.2, bootstrap CI entirely below zero), with
> a highly significant, uncompensated **−1.05 beta to the dollar** (*t* = −13.3) explaining why:
> the currency leg the "fat yield" is supposed to compensate you for instead quietly drained it.

## What we tested

The claim, steelmanned: *"local-currency EM sovereign bonds carry a materially higher yield than
hard-currency EMBI debt — a real, compensated carry, not a free lunch."* We build a Local basket
(the simple average of **EBND** and **LEMB**, two ETFs on different index families — a
cross-provider check, not one benchmark counted twice) and compare its total-return,
excess-of-cash spread against **EMB** (USD EM) and **AGG** (US aggregate), yfinance,
2011-11 → 2026-06 (the sample where all six instruments — Local, EMB, AGG, the dollar proxy UUP
and the cash proxy BIL — co-exist). The headline is a paired one-sample and Newey-West *t* on the
monthly Local-minus-EMB gap, cross-checked with a circular block-bootstrap CI. A separate
regression isolates the FX channel: the Local-minus-EMB *difference itself* regressed on UUP
(dollar strength), netting out the credit-cycle component both legs share. Three named
dollar-strength episodes (2013 taper tantrum, 2015 EM-FX selloff, 2022 strong-dollar Fed-hiking
cycle) show the drawdown anatomy. A deterministic synthetic control with an "extra yield" knob
and an independent dollar-tied "drag" knob proves the machinery faithful. **Dedup:**
[612-em-debt-carry](../612-em-debt-carry/) is the USD side of this exact asset class (no
currency leg); [364-fx-carry-trade](../364-fx-carry-trade/) and
[660-carry-everywhere](../660-carry-everywhere/) test spot-FX and multi-asset carry, never
EM local-currency *bonds* specifically; [339-convertible-bonds](../339-convertible-bonds/) sets
the precedent this study's `NONE` stamp follows (a claimed payoff found statistically real but
backward). No survivorship — every instrument is a single open fund, not a cross-sectional
panel. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why local EM bonds quote a fatter yield in the first place, what a "currency mismatch" is and why local debt shifts it onto *you*, and what actually happened to the money |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the paired/Newey-West/bootstrap splits, the isolated FX-beta regression, the crisis-window anatomy, the Sharpe race and the synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`em_local_bonds/`](em_local_bonds/). EBND, LEMB, EMB, AGG, UUP and BIL are single open
funds (no survivorship panel). **Not investment advice** — research & education.
See [LICENSE](../../LICENSE).*
