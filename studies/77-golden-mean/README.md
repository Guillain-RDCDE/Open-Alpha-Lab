# Study 77 — Golden-Mean

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Fibonacci arm +24.80 bps/5-day, HAC *t* = +2.96 — but so does the placebo (+28.78 bps, *t* = +3.94); Fibonacci − placebo = **−3.98 bps**. The positive return is the equity drift, not Fibonacci magic. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The control arm outperforms the signal; no positive break-even over a random entry exists. Fibonacci levels add zero incremental value over randomly-placed control levels. |
| **Beats random levels?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | 53.3% Fibonacci bounce rate vs 53.5% placebo; Fibonacci underperforms randomly-placed levels on 4 of 6 instruments. Round numbers beat midpoints by +1.16 bps — statistical noise. |

> **In one sentence:** Fibonacci retracements and round numbers do not produce more or better bounces than randomly-placed control levels — the apparent 53% "success rate" is the equity risk premium that any level-based entry inherits, and the +24.80 bps forward return is actually beaten by placebo levels (+28.78 bps).

## What we tested

A widespread belief in retail trading: when price pulls back from a recent swing high or low, it will bounce — specifically at the 38.2%, 50%, and 61.8% Fibonacci retracement of that swing, and at round numbers ($5/$10/$50/$100 steps). We take this literally: we identify swings on six liquid daily tapes (SPY, QQQ, AAPL, MSFT, TSLA, NVDA) over ~25 years, locate level touches within a 40-day forward window, and measure the 5-day bounce return. The critical discipline is a **placebo control arm** — the identical pipeline with randomly-placed levels within the same swing range. Only a Fibonacci-vs-placebo advantage constitutes evidence; the equity drift alone is not.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, what the real data shows, why the "53% success rate" is an equity drift illusion, why it cannot be traded |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-instrument HAC *t*, placebo comparison, per-ratio breakdown, the synthetic mean-reversion positive control, round-number analysis |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`golden_mean/`](golden_mean/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
