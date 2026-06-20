# Study 346 — Multiple-Testing 🎯

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is any battery effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | A methodology demo, not an anomaly hunt. On a synthetic null of 100 effects with **provably nothing to find**, the naive |*t*|>2 screen still reports ~3 "winners" (all false); every correction reports 0. No new real signal is claimed. |
| **Tradability** — is there an edge the search created? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The naive count manufactures discoveries that fail every correction; a battery of marginal calendar tilts at turnover is nothing to harvest, and the few survivors (turn-of-month) are tiny and crowded. |
| **Can you trust a naive *t*-stat from a family of tests?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | Under the global null the naive screen fires a false positive **99%** of the time vs ~5% for the corrections; on the real SPY battery naive (13) overstates the honestly-corrected count (Bonferroni/Holm 7, BH 10) by nearly 2×. |

> **In one sentence:** test 100 calendar effects and a few will clear *t* > 2 by luck alone — so a backtest *t*-stat means nothing until you say how many tests it was the best of and which correction (Bonferroni, Holm, or Benjamini-Hochberg) you applied, with the FWER-vs-FDR choice being a measurable power-vs-false-lead trade-off, not a matter of taste.

## What we tested

The promise behind every "*t*-stat over 2, so it's real" claim holds for **one** pre-registered test — but nobody tests one calendar effect; they quietly try every day-of-week, every month, turn-of-month, the Santa rally, sell-in-May… and report the winners. Harvey, Liu & Zhu (2016) argue this is why the factor-zoo hurdle should be nearer *t* > 3. We make the reckoning operational: screen a **battery of 38 named calendar effects** on SPY (1995–2026) with autocorrelation-robust *t*-stats, then turn the *p*-values into discoveries four ways — the naive |*t*|>2 count, **Bonferroni** and **Holm** (control the family-wise error rate), and **Benjamini-Hochberg** (control the false discovery rate) — and measure each procedure's power and error against a deterministic synthetic battery with a *known* mix of true positives and nulls. (Distinct from [Study 343](../../343-data-mining-roulette/), which runs White's Reality Check on the single best of N random *rules*, and [Study 344](../../344-backtest-overfitting/), which uses the Deflated Sharpe Ratio + PBO on one over-tuned strategy; here it is the FWER-vs-FDR correction bake-off across a *family of named effects*.)

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a few of 100 fake effects always "win", which fix to use, and the real-tape collapse from 13 to 7–10 in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC *t* per effect, FWER (Bonferroni/Holm) vs FDR (Benjamini-Hochberg), the power-vs-error trade-off, the global-null Monte-Carlo calibration, and the real SPY battery |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`multiple_testing/`](multiple_testing/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
