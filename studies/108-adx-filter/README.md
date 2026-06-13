# Study 108 -- ADX-Filter

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) -- see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** -- is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Ungated MA(20/50) cross: **+15.91 bps/trade**, HAC *t* = **+0.39**; gated (ADX > 25): **-25.50 bps/trade**, *t* = **-0.22**; every per-instrument |*t*| < 1 in both arms. |
| **Tradability** -- does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The gated arm is negative before costs; the ungated arm is noise. Neither arm survives any cost level with a positive t-stat. |
| **ADX filter adds value?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | The ADX gate removes **80% of signals** and delivers **-41 bps/trade vs the ungated rule** -- it makes the rule worse, not better. |

> **In one sentence:** the ADX(14) > 25 gate is the most widely taught trend-filter in technical analysis, and on a daily MA(20/50) cross across six liquid instruments it removes four-fifths of signals without adding a single basis point of edge -- the folk rule "only trade strong trends" fails a direct test of its primary promise.

## What we tested

A meta-study on the universal retail rule: *"Never take a moving-average signal unless the ADX(14) is above 25 -- that is when the trend is strong enough to trade."*  We take a standard MA(20/50) crossover on six daily tapes (SPY, QQQ, IWM, AAPL, TSLA, NVDA, 2010-2026), measure 20-day forward returns for every signal **with and without** the ADX filter, and pin both against a **random-direction control** on identical entry dates.  Three comparisons decide the verdict: ungated vs coin (does the cross have any signal?), gated vs ungated (does ADX improve it?), gated vs coin (does the filtered arm beat a fair die?).  All three fail.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the claim, the filter-count reveal, the three-way bar chart, the cost erosion in plain English |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-instrument HAC *t* breakdown, pooled inference, synthetic positive control, cost/turnover sweep |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`adx_filter/`](adx_filter/). **Not investment advice** -- research & education. See [LICENSE](../../LICENSE).*
