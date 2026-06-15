# Study 187 — Three-Soldiers

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — is the effect statistically real? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Zero of six Bonferroni-corrected t-stats survive; the 1-day signal for 3WS is actually *negative* (t = −1.75) — both patterns fire near exhaustion, not at the start of a continuation move. |
| **Tradability** — does it survive costs, capacity, scale? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | Patterns are rare (~1-2 triggers per ticker-year); 1-day signal returns are already negative at 1 bp cost; capacity is negligible. |
| **Continuation or Reversal?** | ![Neither](https://img.shields.io/badge/Neither-8b949e?style=flat-square) | Both patterns underperform a random-day baseline across all three horizons; directional evidence is mixed and non-robust. |

> **In one sentence:** Three White Soldiers and Three Black Crows fire near short-term exhaustion points, not at the start of continuation moves — their claimed direction adds nothing beyond a random day, and after costs both patterns are unprofitable at every horizon.

## What we tested

The two most famous multi-bar Japanese candlestick continuation patterns: **Three White Soldiers** (three consecutive strong bullish candles with ascending closes and opens, each closing near the high) and **Three Black Crows** (the mirror-image bearish pattern). Textbooks from Nison (1991) through Bulkowski (2008) claim both signal powerful directional continuation. We detect them programmatically on 15 US equities (SPY, AAPL, MSFT, AMZN, GOOGL, META, NVDA, TSLA, JPM, BAC, XOM, JNJ, PG, KO, WMT) from 2010 to 2026, measure 1/5/10-day forward returns against a **random-day baseline** in the claimed direction, apply a Bonferroni correction for six comparisons, and confirm the engine recovers an edge on a synthetic tape with planted momentum.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | the pattern recipe, why they fire near exhaustion, the claimed signal vs a random day in plain English |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | per-pattern HAC t-stats, Bonferroni table, the synthetic positive control, cost analysis |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`quantlab/`](../../quantlab/) + [`three_soldiers/`](three_soldiers/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
