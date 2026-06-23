# Study 371 — SKEW-Index 🪝

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does SKEW predict beyond VIX? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | In a forward-return regression that **already controls for VIX**, SKEW's own coefficient is **t = +0.35 (1m) / +0.49 (3m)** with HAC errors — far from t ≥ 2, and *positive*-signed (the opposite of a warning). The one "significant" raw bucket (3m Welch *t* = **−2.33**) is a pseudo-replication artifact: ~13 distinct high-SKEW episodes, block-bootstrap *t* = **−0.50**. No information beyond VIX. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The believers' trade — de-risk to cash while SKEW is in its top decile — **underperforms** buy-and-hold SPY (**9.04%** vs **10.94%** CAGR; Sharpe **0.57** vs **0.65**). Sitting out high-SKEW days is a strict drag because they are not followed by losses. |
| **Warns before crashes?** | ![Busted](https://img.shields.io/badge/Warns_before_crashes%3F-Busted-8b949e?style=flat-square) | After a high SKEW the forward tail-event rate is **lower** than usual (**1.7%** vs a **4.4%** base rate for −10% drawdowns; placebo *p* ≈ 1.0). The "black-swan gauge" fires before *calm*, not before crashes — it prices the tails the market *fears*, which mostly don't arrive. |

> **In one sentence:** the CBOE SKEW index is marketed as the crash warning the VIX can't give, but across 33 years its forward-return coefficient *controlling for VIX* is statistically zero (HAC t ≈ 0.4), high-SKEW days are followed by **fewer** tail events than average (1.7% vs 4.4%), and fading them only **lowers** return and Sharpe — so SKEW is a vivid black-swan story carrying no information beyond VIX and no edge to trade.

## What we tested

The CBOE **SKEW** index reads the price of out-of-the-money S&P 500 puts relative to at-the-money options, so a high SKEW is supposed to mean the market is *paying up for crash insurance* — a tail warning the symmetric, at-the-money **VIX** can't give. We take daily `^SKEW`, `^VIX` and `SPY` closes (1993–2026, **8,330** joint days) and ask the only question that matters: does SKEW carry forward-return or tail-event information **beyond** what VIX already prices? The decisive test is a forward-return regression on *both* SKEW and VIX with HAC (Newey-West) standard errors; we also bucket forward returns by SKEW level (with a block bootstrap that respects the heavy overlap), check whether high SKEW raises the forward tail-event probability, and cost a "fade high SKEW" sleeve. A deterministic synthetic control with a *planted-edge knob* confirms the engine lights up when there **is** a VIX-orthogonal signal and stays quiet when there isn't.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what SKEW actually measures, why "warns before crashes" is a story the data doesn't tell, and why you can't trade a gauge that lights up before calm — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the VIX-controlled HAC regression, the bucket *t* vs its block-bootstrap correction, the tail-warning test vs base rate, the costed fade sleeve, and a synthetic planted-edge control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`skew_index/`](skew_index/). SPY here is a **price-only** proxy (no dividends), labelled on the Tradability axis. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
