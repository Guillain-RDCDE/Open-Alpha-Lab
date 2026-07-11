# Study 682 — Inverse-Fisher-RSI 📈🔀

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | No horizon clears *t* ≥ 2 in support: 10-day NW *t* = **-2.05** — the closest call — runs the *wrong* direction (oversold entries earn **-39.9 bps** less than unconditional); a random-signal placebo beats the real signal on **97%** of draws. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Long-flat timer nets Sharpe **+0.48** (5 bps) vs buy-and-hold **+0.71** over 2010-2026, and only weakly beats a random-exposure control matched on time-in-market (z = +1.27, uncertified). |
| **Sharper than plain RSI(2/14)?** | ![Busted](https://img.shields.io/badge/Sharper_than_plain_RSI%3F-Busted-8b949e?style=flat-square) | IFT-RSI's own gap (-39.9 bps, *t*=-1.89) is worse than plain RSI(2)'s (+20.8 bps, *t*=+1.36) and no better than plain RSI(14)'s (-54.0 bps, *t*=-1.08) — the transform looks crisper on a chart and adds nothing measurable. |

> **In one sentence:** Ehlers' Inverse Fisher Transform squeezes RSI into a bounded, snappy-looking -1/+1 oscillator, but on SPY plus a five-name liquid basket (2010→2026) its ±0.5 crossovers carry **no certified forward-return edge** over an unconditional entry — if anything the point estimate runs the wrong way (10-day NW *t* = -2.05) — it doesn't beat plain RSI(2) or RSI(14) reversal on the identical test, and its costed timer trails buy-and-hold: a mathematically elegant transform of an oscillator that never earns a signal, only a friendlier-looking chart.

## What we tested

Ehlers' exact recipe — `IFT-RSI = tanh(WMA(0.1*(RSI(5)-50), 9))`, bounded in [-1, +1] — tested
on daily total-return closes for **SPY + QQQ, IWM, AAPL, MSFT, NVDA** (2010→2026): forward
returns (5/10/20 days, one execution lag) conditional on the ±0.5 crossover, vs the
**unconditional** distribution, a **random-signal placebo** (20 seeds × 200 draws), and the
identical event-study machinery run on **plain RSI(14)** and **plain RSI(2)** crossovers — the
fairest test of whether the transform *adds* anything. A long-flat **timer with costs** (5/10
bps) checks the tradable side against buy-and-hold and a random-exposure control matched on
time-in-market. A deterministic AR(1) synthetic world with a tunable reversion knob proves the
machinery finds a genuine planted effect and stays silent on a random walk. **Dedup:**
[183-fisher-transform](../183-fisher-transform/) (the *plain* Fisher on price, proven
monotone/redundant — a different mechanism entirely), [75-knee-jerk](../75-knee-jerk/) (the
full RSI(2) treatment — real, *t*=+10.70 — used here only as a baseline),
[428-stochastic-rsi](../428-stochastic-rsi/) (a *different* second transform, Stochastic-of-RSI)
and [669-rsi-divergence](../669-rsi-divergence/) (a structural swing-low pattern, not a
threshold crossover) never test the **Inverse Fisher Transform's own crossover** against plain
RSI — this study does. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why a "crisper-looking" chart isn't the same as a better signal, the head-to-head vs plain RSI, the timer that trails buy-and-hold |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the Welch/NW splits per horizon, the random-signal placebo, the RSI(2)/RSI(14) comparison, the costed timer vs a random-exposure control, the AR(1) synthetic power check |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`inverse_fisher_rsi/`](inverse_fisher_rsi/). SPY/QQQ/IWM are baskets, not survivor
panels; the single names are current mega-caps (a selection, not a survivorship, caveat — named
in [docs/results.md](docs/results.md)). **Not investment advice** — research & education.
See [LICENSE](../../LICENSE).*
