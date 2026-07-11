# Study 690 — Three Stars in the South ⭐⭐⭐

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the shrinking, rising-low three-star block mark a real bottom? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Neither cut clears the Bonferroni-corrected bar (\|*t*\| ≥ 2.50, 4 horizons). The loose cut (n = 363) comes closest at 1 day (Welch *t* = **2.26**, placebo *p* = 0.015) but falls short — and its own sign **reverses** by 20 days (delta −29.1 bps). The literature-closer strict cut fires only **5** times in 25 years across 61 names, below the desk's minimum for any *t*-statistic at all — its raw deltas flip sign at every horizon. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The literature-closer three-star fires about **once per 305 ticker-years** — the rarest pattern this desk has measured, a near-impossibility to trade, not a curiosity. Costs aren't the binding constraint (most cells stay net-positive of a 10 bps round trip); the missing certification and the near-absent event frequency are. |
| **Beats a downtrend base rate?** | ![Mixed](https://img.shields.io/badge/Beats_a_downtrend_base_rate%3F-Mixed-8b949e?style=flat-square) | The loose cut points the right way at 1/5/10 days (small positive deltas, one nominally significant) before reversing at 20 days; the strict cut's five events split 2-for-3 between real bounces (AXP +624 bps, COST +811 bps) and continued declines (TXN, CAT, BDX) — not backwards like a busted pattern, but n = 5 is far too thin to resolve. |

> **In one sentence:** across 61 large-caps and 25 years the shrinking-body,
> rising-low "three stars in the south" fires often enough on the loose reading
> (363 times) to test, but does nothing statistically that survives multiple horizons
> or a multiple-testing correction (Welch *t* flips from +2.26 at 1 day to −0.80 at
> 20 days), while the literature-faithful version — a hammer first star, no gap on the
> second, a marubozu third that never breaks the second star's low — is so honestly
> rare (**5** occurrences, once per 305 ticker-years) that Bulkowski's own published
> caution about insufficient sample size is exactly what the real tape delivers, so the
> honest call is **None × Mirage**: candlestick lore's rarest bullish claim, measured
> as rigorously as five data points allow, and unable to prove itself either way.

## What we tested

We encode three stars in the south two ways, both against a fixed **prior-downtrend**
context (close 2 bars back below its level 10 sessions earlier): the **loose** cut —
three consecutive bearish candles with strictly **shrinking** intrabar ranges and
strictly **rising** lows — and the **strict**, literature-closer cut — the loose shape
plus a real lower shadow on the first star (a hammer-like candle, selling met by some
buying), the second star opening **inside** the first star's real body (no gap down),
and a **near-marubozu** third star (small shadows both sides) that never breaks the
second star's low — across **SPY + 60 long-listed US large-caps** (yfinance daily,
~25 years, cache-first). For each event we take the reversal trade **long** at the next
session's open (one execution lag), reading forward 1/5/10/20-day returns pinned against
the **downtrend-matched base rate** — the same long bet on every bar already sitting in a
matching downtrend, whether or not the specific three-candle shape fired. Four horizons
mean a **Bonferroni** correction (critical \|*t*\| ≥ 2.50), plus a 2,000-draw
label-shuffle placebo. A deterministic synthetic panel plants three-star blocks *only*
where the underlying random walk is already in a downtrend on its own (so star events
and the base rate are drawn from the same population) with a *tunable* planted
post-block bounce, confirming the harness detects a real effect when one exists (planted
edge 0.02 → *t* = +7.44; 0.04 → *t* = +16.16), while the null stays small (mean
*t* = −0.34 over 20 seeds, 2/20 fire — in the same ballpark as sibling 685's own null).
**Dedup:** [187-three-soldiers](../187-three-soldiers/) (the bullish *continuation*
mirror — ascending white candles in an uptrend, the opposite context and claim),
[408-three-black-crows](../408-three-black-crows/) (three falling candles read as a
**bearish continuation**, shorted, with no shrinking/rising-low requirement — the near-
opposite claim dressed in the same "three black candles" costume) and
[687-ladder-bottom](../687-ladder-bottom/) (the desk's other black-candle-into-reversal
bottom, but **five** bars with a separate confirming bullish candle, vs this study's
**three** bars where the shrinking/rising-low shape *is* the signal) never test the
specific **three-shrinking-black-candles-with-rising-lows** shape — this study's own
axis. As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a three-star reversal is, how rare the "real" one actually is, the five real occurrences in full, and why "nominally significant" isn't the same as significant — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the loose-vs-strict detector split, the downtrend-matched base rate, Bonferroni across 4 horizons, the label-shuffle placebo, costs, and a synthetic planted-bounce control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`three_stars_in_the_south/`](three_stars_in_the_south/). Loose = 3 bearish
candles, strictly shrinking range, strictly rising lows, in a downtrend; strict adds a
hammer first star, a no-gap second star, and a marubozu third star that never breaks the
second star's low. Reversal bet is long-only, entered at the next open. Basket is
**survivors** — a single-pattern event study, so this affects which names contribute
events, not the reversal-vs-base-rate direction. **Not investment advice** — research &
education. See [LICENSE](../../LICENSE).*
