# Study 344 — Backtest-Overfitting 📉

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is there a real edge to find? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | On a tape we *built* to have zero timing edge, the best of 1,000 rules posts an in-sample Sharpe of **+0.88** that goes to **−0.49** out-of-sample (OOS HAC *t* = −0.77). Nothing real to detect. |
| **Tradability** — could you trade it? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The gorgeous backtest is a selection artefact — gone the instant you stop peeking. There is, by construction, nothing to harvest. |
| **Does grid-searching manufacture a false edge?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | The Deflated Sharpe Ratio (≈0) and PBO (0.68) catch the trap cold; the expected-max-Sharpe bar climbs 1.6→3.3 as trials go 10→1,000; the positive control spares an honest single hypothesis (buy-and-hold DSR 0.98). |

> **In one sentence:** every backtest looks gorgeous because the search *itself* manufactures the beauty — try enough strategy settings on one dataset and the luckiest is guaranteed to dazzle, even when there is provably nothing there, which is exactly why a Sharpe is meaningless without its trial count and why the Deflated Sharpe Ratio and PBO exist.

## What we tested

Bailey, Borwein & López de Prado's *Pseudo-Mathematics and Financial Charlatanism* (2014) argues that most published backtests are statistically empty: the researcher tried many configurations and reported only the best, without disclosing how many — and the expected maximum Sharpe grows without bound in the number of trials. We make the claim un-deniable by running the demonstration on a tape we **know** is empty: a pure random walk on which no timing rule can work. We grid-search 1,000 long/flat moving-average crossovers, keep the in-sample champion, run it out-of-sample, and put two numbers on the wreckage — the **Deflated Sharpe Ratio** (haircut for the trial count and the return moments) and the **Probability of Backtest Overfitting** (PBO via CSCV). A positive-control tape with a real-but-un-timeable drift confirms the diagnostics punish *searching*, not *having an edge*; one real worked example uses SPY 2000–2026.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the trap in plain language: a Sharpe-0.88 'strategy' built from nothing, why the luck bar rises with every rule, and the catch that spares an honest idea |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the full machinery: expected-max-Sharpe asymptotics, the Deflated Sharpe Ratio with the Lo skew/kurtosis correction, PBO via CSCV, HAC inference, and the real SPY worked example |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`backtest_overfitting/`](backtest_overfitting/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
