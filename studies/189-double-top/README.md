# Study 189 — Double-Top / Double-Bottom

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Zero of six pattern/horizon combinations survive a Bonferroni-corrected excess t-stat test (max |t| = 1.39); double-bottom's positive signal return reflects unconditional market drift, not pattern edge. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | No excess over the random-day placebo at any horizon; even 1 bps round-trip cost pushes the best excess return firmly negative. |
| **Random-day placebo beats pattern?** | ![Yes](https://img.shields.io/badge/Placebo_wins-8b949e?style=flat-square) | At 5-day and 20-day horizons the random-day arm outperforms the confirmed pattern on both double-top and double-bottom. |

> **In one sentence:** across 10 tickers and 15 years of daily data, neither the M-shaped double-top nor the W-shaped double-bottom produces a forward return that is statistically distinguishable from an unconditional random-day bet in the same direction — the patterns describe what has already happened, not what comes next.

## What we tested

The classic chart-analysis claim: *two peaks (or troughs) at a similar price level, separated by a trough (or peak), predict a reversal once the neckline is broken.* We detect both the **double-top** (bearish: break below the intervening trough) and the **double-bottom** (bullish: break above the intervening peak) on daily close data using `scipy.signal.find_peaks` with a 4% height-similarity tolerance and a trailing-ATR prominence filter. The confirmed signal is entered the next bar. Forward returns at 1-day, 5-day, and 20-day horizons are measured against a **random-day placebo** (same tape, same claimed direction, same count) to isolate pattern-specific forecasting power from unconditional market drift. Ten tickers (SPY, QQQ, AAPL, MSFT, AMZN, GOOGL, JPM, XOM, GLD, IWM), 2010–2026, ~2,200 confirmed patterns total. Bonferroni threshold |t| ≥ 3.0 for six tests.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the M & W shapes in plain English, why the neckline break looks like a signal, the random-day test, why the longer-term drift fools the pattern's believers |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | peak-detection sensitivity sweep, per-horizon HAC t-stats, Bonferroni correction, excess vs unconditional drift decomposition, cost sweep, synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`double_top/`](double_top/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
