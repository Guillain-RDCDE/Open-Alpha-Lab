# Study 833 — Deflated Sharpe Ratio 🎏

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is there a real edge to find? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | On a tape we *built* to have **zero** true edge, the best of **1,000** independent strategies posts an annualised Sharpe of **+1.25** (naive *t* = **+2.80**) and is, with certainty, nothing. A synthetic method demo — capped at NONE by construction. |
| **Tradability** — could you trade it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The in-sample champion (Sharpe **+1.77**) collapses out-of-sample to **−0.28** (Newey-West *t* = −0.47) and bleeds on any friction (**−3.65 bps/day** net at 1 bp one-way). By construction there is nothing to harvest. |
| **Does the trial count inflate the best Sharpe?** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | The observed max Sharpe tracks the expected-maximum-Sharpe formula from N=2→1,000 (E[max] ≈ **+1.46** at N=1000). The **DSR** shrinks the winner to a coin flip (**0.32**; mean **0.509** over 40 nulls, deflated excess ≈ 0), firing on **0/40** nulls vs **40/40** for the naive screen — while sparing an honest single strategy (mean DSR **0.965**). |

> **In one sentence:** try enough strategies on one dataset and the luckiest is *guaranteed* to dazzle — the best of 1,000 empty strategies here shows a Sharpe of 1.25 and a *t* of 2.80 with provably zero real edge — because the expected maximum Sharpe grows with the trial count, which is exactly why a Sharpe is meaningless without its `N` and why the Deflated Sharpe Ratio (which shrinks the winner back to a coin flip while sparing genuine skill) exists.

## What we tested

Bailey & López de Prado (2014), **"The Deflated Sharpe Ratio"**: run `N` **independent**
strategies on a tape with **zero** true edge and the *best* sample Sharpe is not zero — under the
null it inflates with `N` per the expected-maximum-Sharpe formula
`E[max] ≈ √V·[(1−γ)Z⁻¹(1−1/N) + γ·Z⁻¹(1−1/(N·e))]`. We build the certified-null world (1,000 iid
zero-drift strategies × 1,260 days, seed 833), watch the winner's Sharpe climb with `N` exactly as
predicted, then let the **Deflated Sharpe Ratio** re-score it against the expected maximum — it
shrinks to a coin flip (DSR 0.32; deflated excess −0.21) while the naive |*t*|>2 screen fires on
every pool. A positive control — an honestly-good *single* strategy (true Sharpe 1.0) — keeps a
high DSR (0.965), proving the correction punishes *searching*, not *skill*. **Dedup:**
[344-backtest-overfitting](../344-backtest-overfitting/) runs the same DSR **plus PBO** on a
**correlated** moving-average crossover grid; [590-sharpe-hacking](../590-sharpe-hacking/) inflates
the Sharpe by **transforming one series' returns**, not by selecting the best of many;
[346-multiple-testing](../346-multiple-testing/) corrects a family of ***p*-values** (Bonferroni /
Holm / Benjamini-Hochberg), not a single selected Sharpe. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the trap in plain language: a Sharpe-1.25 "strategy" built from nothing, why the luck bar rises with every rule, and the catch that spares an honest idea |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the expected-maximum-Sharpe asymptotics, the observed-vs-formula inflation curve, the moment-aware Deflated Sharpe Ratio, the IS→OOS collapse, the costed timer, and the 40-seed null calibration + honest control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`deflated_sharpe/`](deflated_sharpe/). Synthetic-only, offline, deterministic (seed 833) — a research-method demo, so the Signal axis is NONE by construction. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
