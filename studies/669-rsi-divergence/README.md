# Study 669 — RSI-Divergence 📉📈

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does price making a lower low while RSI(14) makes a higher low mark a reversal? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Divergence-flagged trades earn **less** than the unconditional basket at every horizon (gap **−29.7 / −11.3 / −61.2 bps** at 5/10/20d), Welch *t* **−0.62 / −0.20 / −0.84** (NW cross-check matches); a random signal of the identical size **beats** the real pattern on **61-83%** of placebo draws. |
| **Tradability** — can you trade it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Even gross the edge is a coin flip against the basket's own drift; 5-10 bps one-way costs strip real money from an already-negative edge; worst single trade **−14.3%** dwarfs the average win. |
| **Beats a random signal?** | ![Busted](https://img.shields.io/badge/Beats_a_random_signal%3F-Busted-8b949e?style=flat-square) | The fairest control — same signal count, same tickers, random dates — outperforms real RSI-divergence trades on the clear majority of draws (p = 0.61-0.83) across all three horizons. |

> **In one sentence:** across 109 algorithmically confirmed bullish price/RSI(14) divergences on SPY + a five-name liquid basket (2010→2026), the pattern is not just "no better than random" — a same-sized random signal on the identical tickers beats it most of the time, and the classic technical-analysis story (exhausted sellers, a fading RSI presaging a bounce) is a **Mirage**.

## What we tested

We algorithmically detect **confirmed swing lows** (an 11-bar centred fractal, confirmable only
`order`=5 trading days after it prints — that's how a swing low is *defined*, not a look-ahead)
and flag a **bullish divergence** whenever a confirmed swing low sits at a *lower* price than
the previous confirmed swing low but a *higher* RSI(14) reading — the textbook pattern chartists
draw as two trendlines sloping opposite ways. Entry is the single documented execution lag: the
next session's open after confirmation (zero look-ahead); exit is the close 5/10/20 sessions
later. We compare divergence-conditional forward returns against **(a)** the unconditional
forward-return distribution of the same six tickers and **(b)** a **random-signal placebo** —
the fairest bar, since this basket's own 2010-2026 bull drift already beats a coin on its own.
A timer with one-way costs (5/10 bps) prices the third axis; a synthetic random walk with a
TUNABLE planted post-signal bounce, injected exactly on the pipeline's *own* flagged dates,
proves the machinery is unbiased. **Dedup:** siblings
[109-obv-divergence](../109-obv-divergence/) (volume, not RSI, divergence),
[75-knee-jerk](../75-knee-jerk/) (RSI(2) mean reversion, no divergence structure),
[301-triple-rsi](../301-triple-rsi/) (multi-timeframe RSI alignment),
[428-stochastic-rsi](../428-stochastic-rsi/) (Stochastic-on-RSI long-flat timer) and
[178-cci](../178-cci/) (a different oscillator's breach rule) never test **two confirmed swing
lows where price and RSI disagree in direction** — this study does. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a divergence actually looks like on a chart, why traders swear by it, and why the tape says otherwise |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the Welch/HAC splits, the random-signal placebo, the era contrast, the cost-timer and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`rsi_divergence/`](rsi_divergence/). No hardcoded calendar — the pattern is detected
algorithmically from price + RSI(14). No survivorship (SPY/QQQ/IWM are index ETFs; AAPL/MSFT/
NVDA are still-listed mega-caps — the basket cannot manufacture a fake edge from delisted
losers, though it does skew bullish). **Not investment advice** — research & education.
See [LICENSE](../../LICENSE).*
