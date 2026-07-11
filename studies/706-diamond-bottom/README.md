# Study 706 — Diamond Bottom 💎🔻

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the diamond call the turn? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | The "buy the breakout" rule does **not** beat a drift-matched **random-long** baseline: breakout − random = **−13.9 / +35.6 / −25.5 / +33.3 bps** at 5/10/20/60 days, the sign flips across horizons, and the Welch *t* **never exceeds \|1.16\|** (*p* ≥ 0.25 everywhere). The one-sample *t* against zero looks "significant" at every horizon (+2.22 to +4.11) — but that's just the market's upward drift flattering any long, not the diamond. |
| **Tradability** — can you deploy it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | No residual edge once the drift is subtracted out — the breakout is beta with a diamond drawn on top. No ticker beats its own random-long baseline by a meaningful margin (best: QQQ +4.4 bps at 20d). Costs only shave a little more off nothing. |
| **"Does the diamond shape forecast a reversal?"** | ![Busted](https://img.shields.io/badge/Forecasts_a_reversal%3F-Busted-8b949e?style=flat-square) | Scramble the diamond's geometry into nonsense (shuffled-pivot placebo) and the result barely moves: **84%** of nonsense diamonds match or beat the real one (*p* = **0.842**) — even more decisive than the diamond-top study. The broaden-then-narrow shape carries no information. |

> **In one sentence:** the diamond bottom — range broadens then narrows around a low, the
> textbook "rare reversal" — looks like it works because it's a **long on a market that
> drifts up**: encode it mechanically (confirmed-fractal pivots, no eyeballing) and fire the
> "buy the breakout" rule 197 times across 5 indices over 21.5 years, and the breakout never
> beats a drift-matched random long (Welch *t* never clears \|1.16\|, sign flips across
> horizons) while the geometry placebo leaves the result untouched (*p* = 0.84): the diamond
> marks a volatile pause during a decline, not a bottom — the exact mirror-image failure of
> its bearish twin, [466-diamond-top](../466-diamond-top/).

## What we tested

We encode the tightest mechanical version a proponent would accept — the direct bullish
mirror of [study 466's diamond-top engine](../466-diamond-top/). Swing pivots are
**confirmed fractals** (a local extremum with *k* = 5 strictly-beaten bars each side, usable
only 5 bars later — no look-ahead); over the 6 most-recent alternating pivots we require the
swing amplitudes to **broaden** (rise to a peak) then **narrow** (fall) — a diamond — formed
after a **decline**; a **long** fires on the first close **above** the narrowing-apex
ceiling, entered at the **next close** (one documented lag), and we measure the forward
5/10/20/60-day return of the long on SPY, QQQ, IWM, DIA and GLD (yfinance daily
total-return, 2005 → 2026-06). The Signal axis is **breakout vs a drift-matched random-long
baseline** (a Welch *t*) — the only honest test for a long on an upward-drifting tape, since
a naive one-sample *t* against zero is flattered by drift regardless of the pattern — plus a
**shuffled-pivot geometry placebo** that destroys the diamond while keeping the price
marginal. Tradability charges costs on every breakout. A deterministic synthetic control
with a *planted* diamond-bottom reversal proves the detector is live (null, 20 seeds: mean
*t* ≈ 0.03, never fires; planted reversal → *t* = +4.42, win 59%), so the flat real-tape
result is a genuine "nothing there". **Dedup:**
[466-diamond-top](../466-diamond-top/) (the bearish mirror — same geometry, opposite
context and side), [465-broadening-formation](../465-broadening-formation/) (the
broadening leg alone, no narrowing apex), [695-inverse-head-shoulders](../695-inverse-head-shoulders/)
(a different bullish reversal figure, three troughs and a neckline) and
[705-rounding-top](../705-rounding-top/) (a smooth-curvature top, not a swing pattern) —
none of them run this exact broaden-then-narrow-after-a-decline geometry as a long.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a diamond bottom is, why a long on a rising market looks good for free, the breakout-vs-random-long race, and the geometry scramble — in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | mechanical diamonds, one-sample HAC *t* vs the drift trap, the random-long Welch test, the shuffled-pivot placebo, per-ticker deltas, costs, and a synthetic planted-reversal control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`diamond_bottom/`](diamond_bottom/). Pivots are confirmed fractals (k = 5) with a
5-bar confirmation lag; diamonds span the 6 latest alternating pivots; entry is the next
close (one lag), traded as a long. Basket is surviving liquid ETFs — but this is a
single-instrument pattern study, so the random-long baseline neutralizes the drift/
survivorship. **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
