# Study 557 — Borrow-Fee-Signal

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the borrow fee predict returns? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | On the synthetic tape the effect is **real and the engine catches it**: cheap-to-borrow beat expensive-to-borrow by **+12.8%** (two-sample *t* **+2.53**, placebo *p* 0.009; firm slope-*t* **−2.58**), and it **survives controlling for short interest** (joint fee-*t* **−2.15** vs SI-*t* **+0.15**). But it can only be `WEAK`: **no free real borrow-fee tape exists** (private data — Markit/S3), so REAL is unreachable, and even here the seed-robust quintile *t* averages only ~1.5 — a modest effect. |
| **Tradability** — does the spread pay? | ![Fragile](https://img.shields.io/badge/Fragile-dab617?style=flat-square) | The alpha lives on the **expensive-to-borrow short leg** — the fee is both the signal **and** the bill. At the modest quintile fees it survives (gross **+12.8%** → net **+11.8%**, 5 bps/leg + real 3.1% borrow, 3-month hold), but the real hard-to-borrow tail routinely carries **20–100%+** fees that would swamp a 12% spread. Signal and cost share a variable. |
| **Distinct from short interest?** | ![Yes](https://img.shields.io/badge/Yes-2ea44f?style=flat-square) | The fee (price of shorting) carries signal **beyond** short-% of float (Study 262's variable): in a joint regression the fee's slope survives while short interest's collapses to noise (*t* +0.15). Plant the effect in SI instead and the fee-sort's incremental *t* dies to −0.42 — it reads the fee, not SI. |

> **In one sentence:** how expensive a stock is to borrow — the securities-lending fee, the *price* of shorting demand against lendable supply — is a real, literature-backed cross-sectional predictor of low future returns that carries information **beyond** raw short interest, but on a synthetic-only tape (no free real fee data exists) the effect is modest (`WEAK`) and structurally hard to harvest because the alpha sits on exactly the leg that is expensive to borrow (`FRAGILE`).

## What we tested

The **borrow-fee premium** (Cohen, Diether & Malloy 2007; D'Avolio 2002; Drechsler & Drechsler
2016): the securities-lending fee is the market-clearing price of shorting demand against lendable
supply, and *special* / hard-to-borrow names (high fee) go on to earn **negative** returns — with
information the raw short-interest quantity can't see. Because **no clean free historical borrow-fee
tape exists** (the data is private: Markit / IHS DataExplorers, S3 Partners), this study is
**synthetic-only**: a deterministic 120-name fee cross-section (seed 557) with a single knob that
plants the effect, a general-collateral floor plus a *special* right tail, and a **correlated but
distinct** short-interest column. We sort into quintiles by fee, run a two-sample *t* on the
cheap-minus-expensive spread, a **label-shuffle placebo**, a **firm-level slope**, an
**incremental regression** on both fee and short interest (the dedup from short interest), a
bucket-width robustness sweep, honest costs where **the short leg pays its own observed borrow**,
and a seed-robust synthetic positive control (25 seeds) that proves the engine catches a planted
effect and stays flat at the null — plus a short-interest-only null showing the fee-sort doesn't
launder an SI effect. *Distinct from [Study 262 — Short-Interest](../262-short-interest/), which
sorts on the short **quantity**; this study sorts on the **price** of that demand.*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a borrow fee is, why "expensive to borrow" might mean "about to fall", the cheap-vs-expensive chart, and why the trade is a trap |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the quintile sort with a two-sample *t*, the placebo null, the firm-level slope, the incremental-over-short-interest regression, the bucket-width sweep, costs where the short leg pays its own borrow, and the seed-robust synthetic control + SI-only null |

The fingerprinted synthetic headline run (120 names, planted `fee_alpha = −0.035`, panel fp
`f48c7a1477b7`, as-of 2026-06-30) is in [docs/results.md](docs/results.md); the offline machinery
runs on the deterministic world in [`borrow_fee_signal/data.py`](borrow_fee_signal/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`borrow_fee_signal/`](borrow_fee_signal/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
