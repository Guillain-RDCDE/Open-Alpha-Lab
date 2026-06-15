# Study 169 — Fluent-Tickers

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | L-S mean **+66 bps/yr**, HAC *t* = **+1.85** (< 2.0); inside the random-label noise band (random mean −20 ± 98 bps, 20 seeds); no monotone quintile gradient. Survivorship-biased panel — an upper bound. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | No detectable gross edge; annual rebalancing costs are trivial (labels fixed by ticker string) but irrelevant — there is nothing to compound. |
| **Survivorship-biased** | ![Upper_bound](https://img.shields.io/badge/Upper_bound-8b949e?style=flat-square) | Universe is current S&P 500 survivors only; IPO-window effect (the original paper's finding) cannot be tested on a survivor panel. |

> **In one sentence:** Alter & Oppenheimer's 2006 PNAS finding — that easily-pronounced tickers like "KAR" beat hard-to-say tickers like "RDO" — does not generalise to a sustained large-cap long-short on S&P 500 survivors: the fluency signal is indistinguishable from a random ticker-label assignment.

## What we tested

Alter & Oppenheimer (2006) documented that IPOs with fluent, easily-pronounced ticker symbols earned roughly 2.1% more on their first trading day (n=89 NYSE listings, 1990–2004).  The mechanism is **processing-fluency bias**: things that feel cognitively easy to process feel better, truer, and safer.  We score all S&P 500 constituents by a simple vowel/consonant-cluster heuristic (high score = pronounceable like a word, e.g. "KAR"; low score = tongue-twisting, e.g. "RDO"), sort into FLUENT / NON-FLUENT at the median, and run an equal-weight long-short portfolio (annual rebalance) against a **random-label control** (20 seeds).

**Survivorship caveat named:** the S&P 500 survivor panel biases absolute return levels upward; the *gap* between the two groups is directionally unbiased but statistically noisy over 6 years.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | ticker fluency visualised, the L-S return vs the random label baseline, why the effect doesn't survive a large-cap generalisation |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC t-stats, quintile sort, power analysis (14y needed to certify the paper's claimed effect), synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`fluent_tickers/`](fluent_tickers/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
