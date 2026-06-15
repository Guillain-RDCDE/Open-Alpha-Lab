# Study 201 — Dividend-Growth

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![WEAK](https://img.shields.io/badge/WEAK-dab617?style=flat-square) | Grower spread vs EW basket: **−0.73%/yr**, HAC *t* = **−1.51** (26 years). Direction wrong, below the |*t*| ≥ 2 bar. With 78% of the basket qualifying as growers, the filter barely differentiates. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Survivorship-biased universe + near-zero negative spread + publicly known information = no investable edge. |
| **vs EW (same universe)?** | ![No Edge](https://img.shields.io/badge/No_Edge-8b949e?style=flat-square) | Growers lag the equal-weight basket by 0.73%/yr. High-yield non-growers show +2.40%/yr (HAC *t* = 0.92) — not significant. |

> **In one sentence:** in a 40-name survivorship-biased large-cap basket, the dividend-growth screen (≥ 3 consecutive raises) does not produce a statistically significant forward-return premium over the equal-weight baseline — the filter picks 78% of the basket, so it barely differentiates, and the point estimate goes the wrong way.

## What we tested

A cornerstone of retail investing: companies that consistently raise their dividends signal quality, capital discipline, and durable earnings — so "dividend growers" should deliver superior long-run total returns vs high-yield payers who are not growing. We build annual dividend-growth streaks from yfinance `.dividends` data for a **40-name large-cap basket (survivorship-biased — named, not hidden)**, classify "growers" as names with ≥ 3 consecutive raises, and measure their forward calendar-year total return vs (a) the equal-weight full basket and (b) high-yield non-growers. A synthetic panel with a tunable planted premium confirms the engine works — it needs a real premium of ~6–8%/yr to clear the inference bar on 20-year data, and the real tape returns −0.73%/yr.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, the positive control, the real-tape result in plain language, the structural barriers |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-year spread chart, grower count vs basket, Sharpe comparison, HAC t-stats, streak-threshold sensitivity, synthetic power analysis |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`dividend_growth/`](dividend_growth/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
