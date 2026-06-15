# Study 186 — Morning-Star

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Morning-star *underperforms* random days by **−28 bps** (1-day, HAC *t* = −3.14) — significant but in the wrong direction; evening-star is noise (t = −0.18). Zero patterns show a positive excess that clears the Bonferroni bar (|*t*| ≥ 2.50 for 4 tests). |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | ~69 morning-star events and ~75 evening-star events per ticker per year; the direction is wrong (negative excess) so there is no claimed edge to trade, and the reverse trade is not robust either. |
| **Beats random days?** | ![Not--Supported](https://img.shields.io/badge/Not--Supported-8b949e?style=flat-square) | The pattern fires after a large bearish move; random days drawn from the same volatile down-window already capture the mean-reversion bounce better than the day *after* the star, making the forward excess negative. |

> **In one sentence:** the morning-star and evening-star three-candle reversal patterns, detected programmatically across 15 US equities over 16 years, produce no positive excess return over a random-day baseline — the morning-star actually *underperforms* random days (−28 bps, t = −3.14) because its third candle has already consumed the mean-reversion bounce the pattern is designed to trade.

## What we tested

A classic from Japanese candlestick charting lore: *"Three candles — a large bearish, a small indecision star gapping below, then a large bullish recovery closing deep into the first candle's body — signal a reversal. Buy the next day."* We take that literally: morning-star and evening-star patterns are detected programmatically on daily OHLCV bars across SPY and 14 S&P 500 names (15 tickers, ~4,137 daily bars each, 2010–2026), and their 1-day and 5-day forward returns are pinned against a **random-day control** (same number of random days, same claimed direction). The excess return attributable to the pattern — not the unconditional market drift — is what matters, and it is tested with a Bonferroni correction for 4 simultaneous hypotheses.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the three-candle lore in plain language, the random-day baseline, why the morning-star fires too late in the reversion cycle |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-pattern HAC *t* on excess returns, Bonferroni correction, the synthetic positive control (planted mean-reversion), pattern rarity analysis |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`morning_star/`](morning_star/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
