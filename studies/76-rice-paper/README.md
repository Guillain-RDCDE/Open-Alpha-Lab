# Study 76 — Rice-Paper

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | No pattern's excess return over a random-day baseline survives Bonferroni correction (12 tests); highest HAC *t*(excess) = **+2.16** (doji_bullish, 1-day), threshold 2.64 — fails. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Patterns fire ~6–15 times per ticker per year; at ~180 annual trades across 15 tickers, **one bid-ask spread consumes the entire nominal excess**. |
| **Beats random days?** | ![Not--Supported](https://img.shields.io/badge/Not--Supported-8b949e?style=flat-square) | Every excess t-stat falls inside the random-day noise band under multiple-comparisons correction; patterns merely correlate with unconditional drift direction. |

> **In one sentence:** six Japanese candlestick reversal patterns detected systematically across 15 US equities and ~4,100 daily bars each produce no excess return over a random-day baseline that survives a Bonferroni correction for 12 simultaneous tests — the 18th-century rice-chart patterns are fair coins dressed in colourful names.

## What we tested

A staple of technical analysis lore tracing to 18th-century Japanese rice trading: *"A bullish engulfing / hammer / doji appearing after a down-move signals a reversal — buy the next day."* We take that literally: six patterns (bullish engulfing, bearish engulfing, hammer, shooting star, doji-bullish, doji-bearish) are detected programmatically on daily OHLCV bars across SPY and 14 S&P 500 names (15 tickers, ~4,100 daily bars each, 2010–2026), and their 1-day and 5-day forward returns are pinned against a **random-day control** (same number of random days, same claimed direction). The excess return attributable to the pattern — not the unconditional market drift — is what matters, and it is tested with a Bonferroni correction for 12 simultaneous hypotheses.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the candlestick lore, the random-day baseline in plain language, the multiple-comparisons trap, why the doji "result" evaporates |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-pattern HAC *t* on excess returns, Bonferroni correction, the synthetic positive control (planted mean-reversion), pattern frequency and overlap |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`rice_paper/`](rice_paper/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
