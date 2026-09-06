# Study 990 — Counting the Breaks 🚨

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do standard VaR models produce the breach rate they promise? | ![Confirmed](https://img.shields.io/badge/Confirmed-2ea44f?style=flat-square) | At 99% confidence over 6 assets, the textbook **normal** VaR model breached on **1.95%** of days against a promised 1.00% — 1.9× too often — and Kupiec's coverage test rejected it on **100%** of assets. Worse, and this is the part standard practice never checks: the independence test rejected it on **83%**, with runs of up to 4 consecutive breaches. Those are two different failures — the distribution's shape is wrong *and* it does not know today's volatility — and a breach count alone cannot tell them apart. The best model here is **ewma** at 1.79%, passing the joint test on 17% of assets. |
| **Tradability** — does a better-calibrated model change what you would have done? | ![Partial](https://img.shields.io/badge/Partial-dab617?style=flat-square) | Before switching models on this evidence, note how weak the evidence can be. With 5,282 sessions at 99%, a model that breaches **50% too often** (1.5% instead of 1.0%) is caught by Kupiec only **91%** of the time; detecting that reliably needs about **3,536 sessions** — 14 years. On the days the models were wrong they were wrong by a lot: the normal model's average breach overshot its own forecast by 40%, and its worst day lost 37.2% against a forecast of 8.3%. Breach counting says nothing about that, which is the argument for expected shortfall. |

> **In one sentence:** The normal VaR model breaches 1.9× too often and its breaches cluster, while ewma passes the joint test on 17% of assets — but with only 53 expected breaches to count, the test that says so would miss a 50%-too-loose model 9% of the time.

## What we tested

Value-at-Risk is the rare risk measure that makes a falsifiable promise: at 99%
confidence, the loss should exceed the forecast on one day in a hundred. You can check that by
counting. This study counts — five standard models (historical simulation, normal, fitted
Student-*t*, RiskMetrics EWMA, and filtered historical simulation) across six assets at two
confidence levels, with every forecast strictly out of sample.

But the counting has to be done properly, and standard practice does half of it. **Kupiec's**
coverage test asks whether the breach *rate* is right; it is the test everybody runs and it is
blind to *when* the breaches happened. A model that breaches exactly 2.5 times a year, all in
the same fortnight, passes Kupiec and is worthless. **Christoffersen's** independence test asks
whether a breach today predicts a breach tomorrow — and separates two different diseases that
produce the same symptom: a distribution of the wrong *shape* (fat tails) versus a model that
does not know today's *volatility*. The synthetic control turns those knobs independently so the
diagnosis can be checked.

The last section is the one that reframes the rest. With 6,000 sessions at 99% you expect 60
breaches, and the sampling noise on 60 events is large: a model breaching **50% too often** is
caught barely half the time, and detecting that reliably takes decades of data. "The model
passed its backtest" usually means "we did not have enough data to catch it" — and that number
belongs beside every VaR report.
**Dedup:** distinct from **214-value-at-risk-basics** and **507-expected-shortfall** (computing
the measures rather than backtesting them), **256-volatility-clustering** and
**966-garch-vs-har** (volatility forecasting, which appears here only as an input),
**311-fat-tails** (the unconditional distribution) and **988-bitcoin-volatility-decay** (the
level of volatility over time).

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a 99% VaR actually promises, whether five standard models keep it, and why counting the breaks is harder than it sounds |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | five out-of-sample VaR models, Kupiec and Christoffersen and the joint test, breach clustering and overshoot, a simulated power curve, and a synthetic world where the right answer is known |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`breaks/`](breaks/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
