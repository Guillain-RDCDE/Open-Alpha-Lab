# Study 462 — Rising Wedge 📐

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the wedge break-down pay? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The "short the lower-line break" rule does **not** beat a drift-matched **random-short** baseline: break − random = **−6.6 / −17.4 / −34.3 / −85.0 bps** at 5/10/20/60 days (the wedge short is *worse* than a random short at **every** horizon), and the break-vs-random Welch *t* **never clears 2** (max **−1.48** at 60d, *p* = 0.140). The big *negative* one-sample *t*'s (20d **−3.47**, 60d **−6.03**) are **pure beta against a short** — the upward drift every short inherits as a loss. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The short **loses outright** and underperforms even a random short; costs only deepen the hole. The "bearish" break is, on the real tape, followed by a **relief rally**. Nothing to scale. |
| **"Does the wedge forecast a downside break?"** | ![Busted](https://img.shields.io/badge/Forecasts_a_downside_break%3F-Busted-8b949e?style=flat-square) | Scramble the support line into nonsense (slope-scramble placebo) and the result barely moves: **25%** of nonsense lines match or beat the real one (*p* = **0.248**). The specific rising-wedge geometry carries no information. |

> **In one sentence:** The rising wedge is sold as a textbook *bearish* pattern — encode it mechanically (confirmed-fractal pivots, both lines rising, narrowing) and fire the "short the support break" rule 340 times across 5 indices over 21 years, and the short **loses money at every horizon, loses more than a random short, and survives a geometry scramble untouched** (*p* = 0.61… *p* = 0.25): the wedge doesn't forecast a break-down — it precedes a relief rally, all tide, no tool.

## What we tested

We encode the tightest mechanical version a proponent would accept. Swing pivots are **confirmed fractals** (a local extremum with *k* = 10 strictly-beaten bars each side, usable only 10 bars later — no look-ahead); at every bar we fit a support line through the recent confirmed swing lows and a resistance line through the recent swing highs, and qualify a **rising wedge** when both lines rise, support rises faster (converging) and the lines have not yet crossed; a **short** fires on the first close **below the rising support**, entered at the **next close** (one documented lag), and we measure the forward 5/10/20/60-day **short** return on SPY, QQQ, IWM, DIA and GLD (yfinance daily total-return, 2005→2026). The Signal axis is **break-down short vs a drift-matched random-short baseline** (a Welch *t*) — the only honest test on an upward-drifting tape, since a short loses regardless of the pattern — plus a **slope-scramble geometry placebo** that destroys the support line while keeping the price marginal. Tradability charges costs on every break. A deterministic synthetic control with a *planted* rising-wedge break-down proves the detector is live (edge 0 → *t* ≈ 0, a fair coin; planted break-down → *t* = +4.88, 100% win), so the flat real-tape result is a genuine "nothing there".

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a rising wedge is, why shorting a rising market always looks bad, the break-vs-random-short race, and the geometry scramble — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | mechanical wedges, one-sample HAC *t* vs the beta trap, the random-short Welch test, the slope-scramble placebo, per-ticker deltas, costs, and a synthetic planted-break-down control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`rising_wedge/`](rising_wedge/). Pivots are confirmed fractals (k = 10) with a 10-bar confirmation lag; the wedge requires both lines rising and converging; entry is the next close (one lag), shorted. Basket is surviving liquid ETFs — but this is a single-instrument pattern study, so the random-short baseline neutralizes the drift/survivorship. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
