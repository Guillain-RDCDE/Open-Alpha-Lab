# Study 216 -- Hemline-Index

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Fisher's exact p = **0.444** at n = 10 decades. A 70% hit rate sounds compelling until you realise the naive "always predict bull" rule scores 80% with no fashion data at all. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Hemline timer returns +5.28%/yr vs buy-and-hold +6.51%/yr: **−1.23%/yr** underperformance. The signal is not just absent -- it actively destroys value. Plus: hemline direction is only observable retrospectively. |
| **Myth-check: do rising skirts really ring the market bell?** | ![BUSTED](https://img.shields.io/badge/Myth--check-BUSTED-8b949e?style=flat-square) | The hit rate (70%) is arithmetically below the naive baseline (80%). Fisher p = 0.444. The 1940s, 1970s, and 2010s bull markets happened despite falling hemlines. Busted on every dimension. |

> **In one sentence:** the Hemline Index achieves a 70% decade hit rate that is outperformed by simply predicting "bull market" every decade, with a Fisher p of 0.44 and a timer that loses −1.23%/yr to buy-and-hold -- a near-perfect specimen of spurious-correlation folklore.

## What we tested

The claim, attributed to Wharton economist George Taylor (c. 1926) and popularised in 1970s-80s financial media: *rising hemlines accompany bull markets; falling hemlines accompany bear markets.* We encode decade-level hemline direction (rising/falling) using fashion-history consensus for the 1920s through 2010s -- ten decades, the entire observable universe of the claim. We test the decade-by-decade agreement rate against a Fisher's exact test (the appropriate tool for n = 10), compute the phi coefficient, simulate a hemline-based timing strategy vs buy-and-hold, and compare against the naive "always predict bull" baseline. The look-ahead problem (hemline direction only knowable retrospectively) is documented as a structural flaw that makes the claim untradeable regardless of its statistical properties.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the decade-by-decade panel, the 70%-vs-80% baseline comparison, the look-ahead impossibility, and why correlation in fashion cycles reflects shared macro drivers not forecasting |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | phi coefficient, Fisher's exact test, chi-square comparison, timing backtest anatomy, positive control confirming the engine works, proxy-subjectivity sensitivity |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`hemline_index/`](hemline_index/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
