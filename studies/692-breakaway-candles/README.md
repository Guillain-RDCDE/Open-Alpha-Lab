# Study 692 — Breakaway Candles 🕳️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the 5-candle gap/run/reversal shape mark a real trend reversal? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The pre-registered, bidirectional test (bullish + bearish pooled) never clears the Bonferroni bar (\|*t*\| ≥ 2.50) at any of 4 horizons — best is **+2.39** at 20 days, and day 1 is a *negative*, near-significant whipsaw (*t* = **−2.16**). The literature-closer strict cut fires only **5** times basket-wide — too few to test. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | **44** confirmed events over 61 names and 25 years — about once per 35 ticker-years. Where the point estimate is positive it survives a 10 bps round trip, so costs aren't the killer; the near-absent event frequency and the missing certification are. |
| **Works both directions equally?** | ![Busted](https://img.shields.io/badge/Works_both_directions_equally%3F-Busted-8b949e?style=flat-square) | Split by side, the bullish breakaway alone nominally clears the bar at 20d (*t* = +3.33) — but its own day-1 reaction is sharply negative (*t* = −3.43) and its best outcomes cluster on famous market-wide crash bottoms, not a name-specific signal. The bearish mirror shows **nothing** at any horizon (3 of 4 point estimates negative). A real reversal figure shouldn't need the market to be crashing in its favor to work on only one side. |

> **In one sentence:** the breakaway — a gap that holds, a two-day run, then a long candle erasing it — fires so rarely (44 times basket-wide in 25 years) that even its most flattering cut (bullish-only, 20 days) can't survive being asked to also explain why it loses money on day one and why the bearish mirror does nothing at all, so the honest read is a shape the eye notices, not a force the tape obeys.

## What we tested

We encode the closest **objective** breakaway rule we can write down — a genuine
downtrend/uptrend context, a long-bodied first candle, a clean gap that stays open, two
candles running further in the trend's direction, then a long reversal candle closing
back through the gap-day's own high/low — on **SPY + 60 long-listed US large-caps**
(yfinance daily OHLCV, ~25 years, cache-first, the same fixed basket as siblings
685/687). For each confirmed breakaway we enter **one session after** the reversal
candle (no look-ahead) and measure forward 1/5/10/20-day returns, both directions
pooled into one **pre-registered, bidirectional** test (already sign-adjusted to trade
P&L) and compared against each name's own **trend-matched base rate** — the same
directional bet on every bar sharing the context, whether or not the specific shape
fired. Four horizons carry a **Bonferroni** correction; a **strict, literature-closer**
cut (bigger gap, genuinely long candles, a full gap fill) is reported alongside the loose
cut; a 2,000-draw label-shuffle placebo and one-way costs (5/10 bps) round out the
arbiters. The two sides are also reported **separately** as the desk's own symmetry
myth-check — a real bidirectional figure should work on both. A deterministic synthetic
control with a *planted* post-reversal drift confirms the engine can bank a real edge
(Welch *t* = +4.43 to +7.96) against a mostly-clean null (mean *t* = +0.58 over 20 seeds,
disclosed honestly rather than smoothed over). **Dedup:**
[417-island-reversal](../417-island-reversal/) (a two-gap bracket-and-strand figure, no
run in between), [74-mind-the-gap](../74-mind-the-gap/) (any single gap filling, no
candle-count structure), [455-three-methods](../455-three-methods/) (a five-candle
*continuation* pause with no gap at all) and
[687-ladder-bottom](../687-ladder-bottom/) (this study's closest structural cousin — the
loose/strict, base-rate-vs-Bonferroni idiom reused directly, but a no-gap, single-
direction, four-declining-candle figure) never test the specific **gap-then-run-then-
reversal-through-the-gap**, bidirectional shape — this study's own axis. As-of
**2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a breakaway looks like on a chart, why the gap-then-run-then-reversal story sounds so convincing, and why the tape says otherwise — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the loose-vs-strict detector split, the trend-matched base rate, Bonferroni across 4 horizons, the label-shuffle placebo, the bullish/bearish symmetry check, costs, and a synthetic planted-drift control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`breakaway_candles/`](breakaway_candles/). Loose cut = downtrend/uptrend
context + long candle 1 + a clean gap + a 2-day monotone run + a reversal candle closing
back through the gap-day's own high/low; strict adds a bigger gap, long bodies on
candles 1 & 5, and a full gap fill. Reversal trades enter the next session's open.
Basket is **survivors** — named on the Signal axis. **Not investment advice** — research
& education. See [LICENSE](../../LICENSE).*
