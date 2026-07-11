# Study 686 — Stick Sandwich 🥪

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the double-tested close call a bottom? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The bullish stick sandwich does **not** beat a drift-matched **base-rate** baseline: sandwich − base = **+11.5 / −2.9 / −7.3 / +5.2 bps** at 5/10/20/60 days, and the Welch *t* **never clears 2**, let alone the **Bonferroni-corrected 2.50** bar for four simultaneous horizon tests (max \|*t*\| = 1.01). The respectable one-sample *t*'s (up to +7.55 at 60d) are **pure beta** — the basket's own upward drift, which any long-only entry inherits for free. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No residual edge once the free drift is removed; at two of four horizons the pattern is already net-negative before costs, and 10 bps of round-trip cost only deepens the hole. You'd capture the same drift more cheaply by holding the basket. |
| **"Does the equal-close 'meeting' add anything beyond a plain failed rally?"** | ![Busted](https://img.shields.io/badge/Meeting_adds_info%3F-Busted-8b949e?style=flat-square) | Drop the equal-close condition but keep the down-leg + bearish/bullish-rally/bearish context, and the result barely moves: **~11%** of context-matched, no-equal-close draws match or beat the real sandwich (*p* = **0.110**, above the 0.05 bar). The defining "two matching closes" carry no detectable extra information. |

> **In one sentence:** encode the stick sandwich mechanically (bearish, bullish-rally, bearish, outer closes within 15 bps, after a confirmed 10-day decline), fire it **873** times across 30 large-cap names over ~25 years, and it **fails to beat buying on any random day of the same tape** (Bonferroni-corrected Welch *t* never above 1.01) while a placebo that keeps the failed-rally shape but ignores the matching closes does about as well (*p* = 0.11): the "double-tested support" story is decoration on top of ordinary mean-reverting noise.

## What we tested

We encode the tightest mechanical version a proponent (Nison, *Japanese Candlestick Charting
Techniques*) would accept. A **bullish stick sandwich** completes at bar *t* when there is a
confirmed 10-day down leg into the pattern, candle *t-2* is **bearish**, candle *t-1* is
**bullish** and closes *above* candle *t-2* (a failed rally), candle *t* is **bearish**, and the
two bearish closes **meet** (within 15 bps) — two matching-close "bread" candles sandwiching an
up "filling" candle. A long fires on the close of *t*, entered at the **next close** (one
documented lag), and we measure the forward 5/10/20/60-day return on SPY plus 29 long-listed
US large-cap names (yfinance daily total-return, 2001→2026). The Signal axis is
**sandwich vs the unconditional base rate** (a Welch *t*, Bonferroni-corrected across the four
horizons) — the only honest test on an upward-drifting basket — plus a **geometry placebo**
that keeps the down-leg/failed-rally context but drops the equal-close test. Tradability charges
5 bps one-way per leg. A deterministic synthetic control with a *planted* post-sandwich bounce
proves the detector is live (edge 0 → *t* ≈ 0 across 20 seeds; planted bounce → *t* = +23.08), so
the flat real-tape result is a genuine "nothing there." **Dedup:** siblings
[460-counterattack-lines](../460-counterattack-lines/) (the two-candle equal-close cousin, same
verdict), [186-morning-star](../186-morning-star/) (a different three-candle geometry, no
equal-close condition), [452-spinning-top](../452-spinning-top/) (single-candle indecision) and
[459-hikkake-pattern](../459-hikkake-pattern/) (a false-breakout reversal, unrelated geometry)
never test the stick sandwich's defining claim — this study does.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a stick sandwich is, why a dip-buy on a rising market always looks good, the sandwich-vs-base-rate race, and the equal-close placebo — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | mechanical sandwich detection, one-sample HAC *t* vs the beta trap, the Welch base-rate test, the Bonferroni correction, the equal-close geometry placebo, costs, and a synthetic planted-bounce control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`stick_sandwich/`](stick_sandwich/). Pattern is read on the close of *t* (down leg,
bearish/bullish-rally/bearish, equal close within 15 bps); entry is the next close (one lag).
Basket is 30 long-listed, surviving large-caps — but this is a single-instrument pattern study,
so the base-rate baseline (measured on the same panel) neutralizes the drift/survivorship.
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
