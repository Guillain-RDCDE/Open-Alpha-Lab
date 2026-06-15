# Study 188 — Head-Shoulders

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | H&S top: n = 3 (inference impossible). Inverse H&S: n = 20, max \|t_excess\| = 1.97 (Bonferroni threshold 2.58). No horizon passes. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Strict detection fires ~0.15 signals per ticker per year — too rare to be a strategy. Loosening the filter re-introduces selection bias. |
| **Pattern frequency** | ![Too--rare](https://img.shields.io/badge/Pattern_frequency-Too--rare-8b949e?style=flat-square) | 23 signals in 200 ticker-years (2005–2026). The pattern's textbook "reliability" is a subjective-recognition artefact. |

> **In one sentence:** the head-and-shoulders pattern, detected rigorously with a five-point structure and confirmed neckline break, fires almost never on major US equities and the signals that do appear carry no statistically real directional edge over an unconditional random-entry baseline.

## What we tested

The H&S top is the most famous chart pattern: a central price peak (the *head*) flanked by two lower peaks (the *shoulders*), confirmed when the close breaks below the *neckline* connecting the two troughs.  We implement a strict algorithmic detector using `scipy.signal.find_peaks`, requiring shoulder symmetry (heights within 10%), an approximately horizontal neckline (tilt < 15% of the head-to-neckline amplitude), and a confirmed neckline break on the close.  Forward returns at 1, 5, 20, and 60 days after the break bar are then pinned against an unconditional random-day baseline across ten liquid US names (SPY, QQQ, AAPL, MSFT, AMZN, GOOGL, META, NVDA, TSLA, JPM) over ~20 years of daily bars.  A Bonferroni correction covers the eight tests (two pattern types × four horizons).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what the pattern is, why it looks reliable (selection bias), the honest count and forward-return test in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-horizon HAC t-stats, Bonferroni correction, the rarity–selection-bias trade-off, placebo arm, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`head_shoulders/`](head_shoulders/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
