# Study 691 — Homing Pigeon 🐦

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does a shrinking down-day inside a bigger down-day mark a floor? | ![Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square) | Across **2,885** homing-pigeon occurrences (a smaller down day fully inside a larger prior down day, after a downtrend) on **26** large-caps + SPY (1962→2026), the pooled edge clears **HAC *t* ≥ 2 at every horizon** (best +3.37 at 3 days) — the strongest raw reading of this desk's five candlestick studies. But the decisive **alpha-vs-beta cut** (excess over buying *any* dip in the same downtrend, no pattern required) clears *t* ≥ 2 only at 3/5 days (2.37 / 2.07), not 1 or 10 days, and only **3 of 26 names** individually clear \|*t*\| > 2 — chance level. Events aren't a repackaged crash (busiest 10 weeks = 3.9% of the sample). The harness detects a *planted* floor at *t* = 13.6. (Survivorship tilts *toward* a bounce → conservative.) |
| **Tradability** — can you deploy it? | ![Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square) | Net of a 5-bps round trip the 3-day rule nets **+12.4 bps** and stays positive even at 10 bps (+2.4 bps) — unlike its sibling candlestick studies, costs don't kill this one. But only 3/26 names carry it individually, and part of the edge is a simpler "buy the downtrend dip" rule in a costume. |
| **Beats plain buying any downtrend dip?** | ![Mixed](https://img.shields.io/badge/Beats_a_downtrend_dip%3F-Mixed-8b949e?style=flat-square) | The shape's excess over the no-pattern-required downtrend base rate clears *t* ≥ 2 at 3 and 5 days but **not** at 1 or 10 days — a genuine split, not a clean confirm or bust. |

> **In one sentence:** the homing pigeon is the *one* candlestick pattern on this desk whose pooled edge clears HAC *t* ≥ 2 at every horizon and survives realistic costs — but a real share of that edge is just generic "buy the dip in a downtrend" beta (the excess over that baseline misses *t* = 2 at 1 and 10 days), and only 3 of 26 names individually carry a reproducible edge, so the honest stamp is **Weak × Fragile**, not the clean Real the raw pooled *t* would suggest.

## What we tested

We detect the homing-pigeon candle by precise OHLC rules — a smaller **down** day whose
real body sits entirely inside a larger prior **down** day's body (the *same-colour*
cousin of the [harami](../406-harami-pattern/)) — across the same fixed **26-name** liquid
large-cap basket + SPY (yfinance **un-adjusted, price-only** daily bars, cache-first, the
panel shared with the sibling candlestick studies). For each occurrence after a
**downtrend** (the bullish claim) we measure the **forward 1/3/5/10-day return**, entered
at the **next close** (one documented execution lag), against that name's own
**unconditional base rate**, and test it with a HAC one-sample *t*, a label-shuffle
placebo, and a **Bonferroni** correction across the four simultaneous horizons. The
decisive extra cut asks whether the shape beats **plain "buy any dip in the same
downtrend"** (no pattern required) — the honest alpha-vs-beta test — and a per-name
breakdown and an event-clustering check (are the "many" events really a handful of shared
crash weeks?) probe whether the pooled significance is robust or an averaging artefact. A
deterministic synthetic control with a *planted* bounce proves the harness can detect a
floor when one exists, and we cross-check the same geometry traded long after an
**uptrend** — the "wrong side" of the trend split — to confirm the downtrend condition is
doing real work. Survivorship (surviving-names basket, biased *toward* a bounce for this
bullish claim) is named on the Signal axis. **Dedup:** siblings
[406-harami-pattern](../406-harami-pattern/) (the opposite-colour containment rule this
pattern specializes), [403-hammer-hanging-man](../403-hammer-hanging-man/) (a one-bar
wick shape) and [186-morning-star](../186-morning-star/) (a three-bar sequence) test
different geometries under the same protocol; [684-inverted-hammer](../684-inverted-hammer/)
runs the identical basket and protocol for a different one-bar bullish-floor claim, and
came back a clean None × Mirage — this study's own cuts test whether the two-bar shape
does any better.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a homing pigeon *is*, why a shrinking down-day *feels* like sellers giving up, what "beating the base rate" (and beating the dip-buying baseline) means, and why the floor is real-ish but thin — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the OHLC detector, forward 1/3/5/10-day edge vs base rate, HAC *t* + label-shuffle placebo + Bonferroni, the alpha-vs-beta cut, the trend-split contrast, event clustering, the per-name breakdown, costs, and a synthetic planted-floor power control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`homing_pigeon/`](homing_pigeon/). Un-adjusted OHLC (the candle shape needs printed levels) → forward returns are **price-only**. Basket is **survivors** — named on the Signal axis. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
