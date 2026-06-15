# Study 160 — Skyscraper-Curse

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Post-completion 1-yr S&P mean **+13.4%** vs +10.9% unconditional (events *outperform*); random-day p = **0.614**; HAC *t* unreliable at n = 6 (power ~5%). |
| **Tradability** — does it survive the real world? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | 6 events over 50 years = one trade per decade; no actionable entry/exit rule; impossible to backtest meaningfully. |
| **n too small?** | ![Busted](https://img.shields.io/badge/n_too_small-Busted-8b949e?style=flat-square) | Structural power floor ~5% at n=6; need ~30+ events to reach the inference bar — the world will never build that many record skyscrapers. |

> **In one sentence:** the Skyscraper Index is one of finance's most beautiful narratives — world-record towers completing near market peaks — but on all six S&P-era completions, post-event returns average *above* the unconditional baseline, the random-day p-value is 0.61, and with n=6 events in 50 years the test structurally cannot distinguish signal from noise.

## What we tested

Andrew Lawrence (1999) proposed the Skyscraper Index: record-height building completions cluster at the peak of credit cycles and should precede market downturns. We test this as a forward-return event study on all **6 world-record completions since 1957** (WTC 1972, Sears Tower 1974, Petronas 1998, Taipei 101 2004, Burj Khalifa 2010, Merdeka 118 2023), measuring compound S&P 500 total returns in the 1, 2, and 3 years after each event against the **unconditional Shiller baseline** (1871-2026) and a **random-day Monte-Carlo control**. The fun teardown: one event goes the "right" way (WTC/1973 bear), the other five do not — and a random bundle of 6 years looks identical.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the legend, the per-event bar chart, the random-day bootstrap, why the curse is a narrative trick |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | HAC inference on n=6, random-day p-value, power analysis (why we need ~30 events), synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`skyscraper_curse/`](skyscraper_curse/). Event table hardcoded from Lawrence (1999) + CTBUH. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
