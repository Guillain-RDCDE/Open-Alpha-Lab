# Study 687 — Ladder Bottom 🪜

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the 5-candle ladder mark a real bottom? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Neither cut clears the Bonferroni-corrected bar (\|*t*\| ≥ 2.50, 4 horizons). The common "loose" cut (n = 2,543) is *negative* at 1 day, flat at 10, and only nominally significant at 5/20. The literature-closer "strict" cut (n = 81) comes closest — 20-day Welch *t* = **2.30**, placebo *p* = **0.006** — genuinely interesting, but still short, and its own 1-day reaction is negative. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The literature-closer ladder fires about **once per 19 ticker-years** — a curiosity, not a strategy. Where the point estimate is positive it survives a 10 bps round trip, so costs aren't the killer; the missing certification and the near-absent event frequency are. |
| **Beats a downtrend base rate?** | ![Mixed](https://img.shields.io/badge/Beats_a_downtrend_base_rate%3F-Mixed-8b949e?style=flat-square) | The strict cut points the right way at 3 of 4 horizons with a large net return (+330 bps at 20d) and a suggestive placebo — not backwards like a busted bearish pattern — but doesn't clear the bar, and 3 of its 5 best/worst events land on famous *market-wide* bottoms (2002, 2009, 2020), not obviously the candle shape itself. |

> **In one sentence:** across 61 large-caps and 25 years the ladder bottom's four-declining-
> candles setup fires often (2,543 times) but does nothing statistically (Welch *t* flips sign
> across horizons), while the literature-faithful version — small-bodied committed rungs, a
> warning wick, a true gap-up reversal — is honestly rare (81 times, ~once per 19 ticker-years)
> and comes tantalizingly close to certifying a real bounce (*t* = 2.30 at 20 days) without
> quite clearing the desk's multiple-testing bar, so the honest call is **None × Mirage**: a
> pattern that might be onto something and simply cannot prove it on the sample size the real
> world offers.

## What we tested

We encode the ladder bottom two ways, both against a fixed **prior-downtrend** context
(close 4 bars back below its level 10 sessions earlier): the **loose** cut — four consecutive
bearish candles with strictly descending closes, then a bullish candle closing above the
fourth rung — and the **strict**, literature-closer cut — the loose shape plus committed
(near-low) selling on the first three rungs, a "warning" upper shadow on the fourth, and a
true gap-up on the fifth — across **SPY + 60 long-listed US large-caps** (yfinance daily,
~25 years, cache-first). For each event we take the reversal trade **long** at the next
session's open (one execution lag), reading forward 1/5/10/20-day returns pinned against the
**downtrend-matched base rate** — the same long bet on every bar already sitting in a matching
downtrend, whether or not the specific five-candle shape fired. Four horizons mean a
**Bonferroni** correction (critical \|*t*\| ≥ 2.50), plus a 2,000-draw label-shuffle placebo. A
deterministic synthetic panel with a *planted* post-ladder bounce confirms the harness detects
a real effect when one exists (planted edge 0.02 → *t* = +2.85; 0.04 → *t* = +4.02), and the
null stays small (mean *t* = +0.02 over 20 seeds, 0/20 fire) next to that. **Dedup:**
[455-three-methods](../455-three-methods/) (a different 5-candle shape — a *continuation*
pause, not a reversal, on ETFs), [408-three-black-crows](../408-three-black-crows/) (the same
four-falling-candles setup read *bearish*, shorted, not the fifth reversing candle this study
trades long), [186-morning-star](../186-morning-star/) (a 3-candle bullish reversal, same
random-baseline/Bonferroni idiom) and [685-tri-star-doji](../685-tri-star-doji/) (3 dojis, the
strict/loose + `MIN_N_FOR_TEST` discipline this study reuses directly) never test the specific
**four-declining-then-one-reversing** five-bar shape — this study's own axis. As-of
**2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a ladder bottom is, how rare the "real" one actually is, the best and worst real occurrences, and why "almost significant" isn't the same as significant — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the loose-vs-strict detector split, the downtrend-matched base rate, Bonferroni across 4 horizons, the label-shuffle placebo, costs, and a synthetic planted-bounce control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`ladder_bottom/`](ladder_bottom/). Loose = 4 declining bearish candles in a
downtrend then a bullish break; strict adds near-low rungs 1-3, a warning wick on rung 4, and
a gap-up on rung 5. Reversal bet is long-only, entered at the next open. Basket is
**survivors** — a single-pattern event study, so this affects which names contribute events,
not the reversal-vs-base-rate direction. **Not investment advice** — research & education.
See [LICENSE](../../LICENSE).*
