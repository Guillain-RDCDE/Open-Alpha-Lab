# Study 698 — ABCD-Harmonic

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does price reverse at a completed AB=CD point D? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | Fading at D (5-day hold) returns **−45.3 bps/event, HAC *t* = −1.18** — negative, and nowhere near *t* ≥ 2 at any horizon (1/5/10-day *t* = −0.35 / −1.18 / −1.49). The lone individually significant number on the board (TSLA, *t* = −2.12) points *against* the pattern. |
| **Tradability** — does it survive costs? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The fade loses money **before** costs and loses more after: −45.3 → −65.3 bps/event as one-way cost rises 0 → 10 bps. No cost level makes it attractive. |
| **Beats a random equal-legged reversal projection?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | A placebo arm reruns the identical pivot-detection pipeline with off-Fibonacci retrace/extension targets. Fibonacci **underperforms** the placebo at every horizon (Welch *t* from −0.24 to −1.68) and beats it on only 2 of 6 tickers. |

> **In one sentence:** the AB=CD harmonic — BC retraces AB by 0.618, CD projects an
> equal-length leg from C, price is supposed to turn at D — shows no reversal edge on
> six liquid daily tapes (2001/2010→2026): the fade is negative on average, never
> statistically distinguishable from zero, and does not beat a placebo control built
> from the identical pivots with arbitrary, non-Fibonacci ratios.

## What we tested

The AB=CD harmonic, the simplest member of the retail "harmonic pattern" family
(Gartley/Bat/Butterfly all elaborate on it with an added X point — see the dedup map
below): three confirmed swing pivots A, B, C where BC retraces AB by **61.8%**,
projecting **D = C + AB** ("AB=CD"), with a reversal expected the moment price
touches D. On SPY, QQQ, AAPL, MSFT, TSLA and NVDA daily bars (the identical basket as
sibling [77-golden-mean](../77-golden-mean/)), we detect every such candidate off a
percentage-threshold zigzag (confirmed pivots only — no look-ahead), scan forward for
the first touch of the projected D, and measure the forward return of fading there —
against a placebo arm that reruns the *identical* pipeline with randomized,
off-Fibonacci retrace/extension targets. Only a Fibonacci-specific advantage over that
placebo would constitute evidence.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what an AB=CD pattern actually looks like on a chart, why the "reversal" is supposed to happen, and what the real tape shows when you actually fade it |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the pivot-confirmation mechanics, the HAC/Welch splits, the per-instrument and per-horizon breakdown, the cost sweep, and the 20-seed synthetic positive control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`abcd_harmonic/`](abcd_harmonic/). No survivorship — six currently-listed,
individually named large-cap/ETF tickers, not a membership-conditioned panel.
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
