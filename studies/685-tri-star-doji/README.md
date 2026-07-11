# Study 685 — Tri-Star Doji ✨

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do three dojis in a row mark a reversal? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The literature-faithful **gapped** tri-star fires **twice** in 61 names × 25 years — both times the "major reversal" *lost* (mean −611 to −1,355 bps by horizon, 0% win rate), too few to test. The **loose** cut (any 3 dojis in a row, n = **410**) shows the reversal bet **losing to the base rate at every horizon** (−16 to −94 bps); no horizon clears the Bonferroni bar (\|*t*\| ≥ 2.50) — closest is 20d at *t* = −2.47, and it's the wrong sign. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The real pattern occurs about **once per 30 ticker-years** — a curiosity, not a strategy — and there is no gross edge to defend: net of a 2 bps round-trip + short borrow, every horizon loses *more* than it did gross. |
| **"Does the tri-star mark a major reversal?"** | ![Busted](https://img.shields.io/badge/Tri--star_forecasts%3F-Busted-8b949e?style=flat-square) | Both real strict occurrences lost money; the loose cut's delta is negative at every horizon; a label-shuffle placebo shows a random same-size sample **beats** the tri-star 81–99% of the time. Stacking three dojis doesn't manufacture a reversal that a single doji ([study 405](../../405-doji-reversal/)) already failed to show. |

> **In one sentence:** across 61 large-caps and 25 years the *real* tri-star — three dojis with
> the middle one truly gapped away from its neighbours, Nison's own definition — occurred
> **twice**, and both times the market kept going the way it had been going (JPM −2.2%, CSCO
> −25.0% over the next 60 days); even the loose "three dojis, no gap" reading that's common
> enough to test shows the reversal bet losing to a random-day baseline at every horizon, so
> the desk's rarest chart pattern is an honest **None × Mirage**: too rare to certify, and what
> little evidence exists points the wrong way.

## What we tested

We detect the **tri-star** two ways, both against a fixed doji cut (real body ≤ 10% of the
day's high-low range, the same cut as siblings 405/458): the **strict**, literature-faithful
version — three consecutive dojis where the middle one's range gaps clear of *both*
neighbours (a true island star) — and a **loose** version — three dojis back to back, no gap
requirement — across **SPY + 60 long-listed US large-caps** (yfinance daily, ~25 years,
cache-first). For each tri-star we tag the 10-day trend into the block and take the textbook
**reversal** bet against it at the next session's open, reading forward 5/10/20/60-day
returns pinned against the **unconditional base rate** (the same against-the-trend bet on
every eligible bar). Below 8 pooled events we do not compute a *t*-stat at all — "too few to
test" is the honest answer, not a decorated non-result — and where a test *does* run, four
horizons mean a **Bonferroni** correction (critical \|*t*\| ≥ 2.50), plus a 2,000-draw
label-shuffle placebo. A deterministic synthetic panel with a *planted* tri-star reversal
confirms the harness detects a real effect when one exists (planted edge 0.02 → *t* = +2.36;
0.04 → *t* = +5.14), and the null stays small (mean *t* = −0.60 over 20 seeds) next to that.
**Dedup:** siblings [405-doji-reversal](../405-doji-reversal/) (one doji, same base-rate
idiom), [186-morning-star](../186-morning-star/) (a different three-candle shape, random-day
baseline) and [458-abandoned-baby](../458-abandoned-baby/) (one gapped doji between
directional candles) never test **three dojis in a row** — this study's own axis. As-of
**2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a tri-star is, why "rare" turns out to mean *two in 25 years*, the two real occurrences and what happened next, and why even the loose reading doesn't rescue the claim — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the strict-vs-loose detector split, the base-rate comparison, Bonferroni across 4 horizons, the label-shuffle placebo, costs, and a synthetic planted-reversal control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`tri_star_doji/`](tri_star_doji/). Doji = real body ≤ 10% of range; strict tri-star
requires the middle doji's range to clear both neighbours, loose requires only three in a row;
reversal bet is against the 10-day prior trend, entered at the next open. Basket is
**survivors** — a single-pattern event study, so this affects which names contribute events,
not the reversal-vs-base-rate direction. **Not investment advice** — research & education.
See [LICENSE](../../LICENSE).*
