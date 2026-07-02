# Study 559 — Dark-Pool-Ratio 🌑

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a stock's dark-pool ratio predict its return? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | **No free real tape exists** — FINRA ATS volume is weekly & lagged, off-ATS internalised flow is excluded, clean daily feeds are paywalled — so the one number that could earn `REAL` (a robust IC-*t* ≥ 2 on a *real* DPR panel) can't be computed. On the synthetic null the IC is ≈ 0 (IC-*t* **+0.44**, placebo *p* **0.65**); the *only* apparent lift (raw null IC-*t* **+0.77** over 25 seeds) is the **size/liquidity confound** — control for size and it collapses to **−0.19**. Synthetic-only ⇒ capped below `REAL`. |
| **Tradability** — does the sort pay? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | A long-dark/short-lit book whose only apparent edge is a size confound, on a panel that can't be built from free data. Null gross spread **+1.28%** → net **+0.58%** after 5 bps/leg + 50 bps borrow — noise, halved by costs. |
| **"Informed accumulation in the dark?"** | ![Unproven](https://img.shields.io/badge/Unproven-8b949e?style=flat-square) | The engine banks a *planted* DPR effect at IC-*t* **+2.32** (25 seeds, still +1.39 after netting size) — the detector is faithful. But microstructure theory (Zhu 2014; Comerton-Forde & Putniņš 2015) says dark share's effect on information is *non-monotone and possibly bearish*, and no free tape lets us test it. Untestable, not confirmed. |

> **In one sentence:** the "rising dark-pool ratio = smart money loading up" story can't be tested on a free retail stack — the daily off-exchange panel doesn't exist (FINRA ATS is weekly, lagged, and misses wholesaler internalisation) — and in the synthetic world the naive version is a **size/liquidity confound** (raw null IC-*t* +0.77 → −0.19 once size is netted out), while the engine, proven on a planted effect at IC-*t* +2.32, would catch a real signal if a tape existed.

## What we tested

The microstructure folklore: when a bigger slice of a stock's volume prints **off-exchange** (dark
pools, ATSs, internalisers) rather than on the lit tape, that is supposedly *informed accumulation*
and the stock drifts up. Because **no free per-name daily dark-pool-ratio tape is reachable** (FINRA
ATS volume is weekly and lagged, the large off-ATS internalised flow is excluded, clean feeds are
paywalled), this is a **synthetic-only** study — it proves the machinery and names the tape gap on
the Signal axis. We build a deterministic DPR panel (one knob `dpr_alpha` plants the effect; `0` is
the null), measure the **Spearman information coefficient** with a Fisher-z *t*, sort a **quintile
long-short** with a two-sample *t*, run a **label-shuffle placebo**, add costs + a short borrow, and
close with a **seed-robust (25-seed) positive control**. The twist: the DPR is correlated with
size/liquidity, so a naive sort shows a *false* positive at the null (IC-*t* +0.77) that **vanishes
once size is controlled** (−0.19) — the study's central honesty point. A synthetic-only study can
never be `REAL` (that needs a robust *t* ≥ 2 on a real tape). *Distinct from
[376 MOC-Imbalance](../376-moc-imbalance/) (close-auction imbalance) and
[418](../418-money-flow-index/)/[419](../419-chaikin-money-flow/) (lit price×volume oscillators):
this is the **venue-of-execution** signal.*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a dark pool is, why "more dark = smart money" is folklore, why the free data doesn't exist, and why the apparent signal is really about size |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the Spearman IC with a Fisher-z *t*, the quintile long-short, the label-shuffle placebo, the size-confound control, costs + borrow, the monotone effect sweep, and the seed-robust positive control |

The fingerprinted synthetic headline run (null panel fp `11f28bf40b8c`, planted fp `fbdeeb6093c5`,
as-of 2026-06-30) is in [docs/results.md](docs/results.md); the whole study runs offline on the
deterministic world in [`dark_pool_ratio/data.py`](dark_pool_ratio/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`dark_pool_ratio/`](dark_pool_ratio/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
