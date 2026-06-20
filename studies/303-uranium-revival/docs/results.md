# Results — Study 303 (Uranium-Revival), synthetic regime controls

*Generated from the deterministic offline tapes (`uranium_revival.data.synthetic_daily`,
seed=303). The trend rule: **long when close > SMA(200), else in cash**; one execution lag
(signal at the close of *t* earns *t+1*'s return); costs one-way × NAV on every change of
position. The race is **excess-of-cash vs excess-of-cash** — the **timing edge** = timed
book minus buy-and-hold, annualised. HAC (Newey-West) *t* on the daily edge; circular
block-bootstrap (block = 21d) CI on the annualised edge.*

> **No live URA/URNM tape ships with this study.** Every number below is a *machinery*
> result on synthetic regimes whose ground truth we control — a synthetic control is a
> proof that the harness can detect a planted effect, **never** market evidence. By the
> desk's inference bar, that means the Signal stamp is **WEAK**, not REAL: a real-tape
> HAC *t* ≥ 2 we did not measure cannot be claimed.
>
> As-of **2026-06-19**. Reproduce by running `uranium_revival.data.synthetic_daily` at
> seed 303 and matching the per-tape fingerprint below.

## Data stamp (synthetic tapes, seed=303)

| Regime | Role | n_days | Fingerprint |
|---|---|--:|---|
| `trend`  | positive control (persistent bull/bear runs) | 3,000 | see `data.fingerprint` |
| `random` | null (a coin / random walk) | 3,000 | see `data.fingerprint` |
| `hype`   | the realist's case (boom-bust rocket) | 2,500 | see `data.fingerprint` |

*(Fingerprints are content hashes of each tape's close; regenerate locally to confirm you
hold the same deterministic tape.)*

## The headline — trend-timing vs buy-and-hold, by regime (10 bps one-way)

| Regime | timed (%/yr) | buy & hold (%/yr) | timing edge (%/yr) | HAC *t* | Sharpe diff |
|---|--:|--:|--:|--:|--:|
| **`trend` (control)** | +1.7 | −26.8 | **+28.5** | **+2.85** | +0.78 |
| `random` (null) | −13.2 | −11.4 | −1.8 | **−0.75** | −0.31 |
| `hype` (rocket) | +65.1 | +7.3 | +57.7 | **+3.68** | +1.44 |

- **Detection works.** On the planted-trend control the rule banks a +28.5%/yr timing edge
  at HAC *t* = +2.85 (mostly by sitting out the bear runs) — the harness can detect and
  bank a genuine trend.
- **Null is null.** On the coin the edge is −1.8%/yr at *t* = −0.75 — indistinguishable
  from zero, with heavy whipsaw (82 round-trips). The test is not rigged to find trends.
- **The trap.** On the boom-bust rocket the rule posts a spectacular +57.7%/yr edge at
  *t* = +3.68 — and that is the whole problem: a single boom-bust is statistically
  indistinguishable, in-sample, from a real trend.

## The block-bootstrap CIs — how wide a single tape really is

| Regime | annualised timing edge | 95% block-bootstrap CI |
|---|--:|--:|
| `trend` (control) | +28.5%/yr | **[+7.9, +48.0]%/yr** |
| `random` (null) | −1.8%/yr | [−23.5, +11.5]%/yr (straddles 0) |
| `hype` (rocket) | +57.7%/yr | wide, positive (regime-specific) |

Even the planted trend's edge spans a ~40-point interval; the coin straddles zero. On a
*single* asset the uncertainty is enormous — which is exactly why a real-tape conclusion
needs breadth (many markets), not one volatile ETF.

## The single-asset trend trap — same rule, twelve hype-rocket draws

Holding the rule fixed and redrawing the boom-bust (seeds 303–314), the HAC *t* of the
timing edge swings widely from draw to draw (roughly +1 to +5). A *real* edge would be
stable across draws; this one is a coin flip on which rocket you got. **The flattering
backtest is regime luck, not a durable signal.**

## Could you trade it? — cost sweep on the rocket (one-way bps)

| one-way cost | timing edge (%/yr) | HAC *t* |
|---|--:|--:|
| 0 bps  | +58.3 | +3.72 |
| 10 bps | +57.7 | +3.68 |
| 20 bps | +57.2 | +3.64 |
| 50 bps | +55.6 | +3.53 |

A 200-day rule trades a handful of times a year, so costs barely register — the edge
survives 50 bps almost untouched. **This is the seduction, not the safety:** cost-robust,
low-drawdown, gorgeous in-sample — and entirely a function of the one boom-bust you fitted.
The binding constraint is concentration and regime-dependence, neither of which a cost
sweep can show.

## Verdict

- **Signal — WEAK.** The harness banks a *planted* trend (HAC *t* = +2.85) and is null on a
  coin (*t* = −0.75), so the machinery is faithful. But **no live URA/URNM tape** ships here
  to certify a real-tape *t* ≥ 2 — and the desk's rule is that REAL is earned by the tape,
  not by the literature or a synthetic control. Hence WEAK.
- **Tradability — MIRAGE.** On the realistic boom-bust regime the +57.7%/yr "edge" is one
  crash-dodge on a single thin theme — concentration risk dressed as timing, not a scalable,
  repeatable edge. It survives every cost test and fails the one that matters (out-of-sample
  uranium, which we cannot run here).
- **Durable trend or hype rocket? — HYPE ROCKET.** Cost-robust (*t* = +3.53 at 50 bps) yet
  entirely regime-specific: redrawing the boom scrambles the *t*. On a single thematic ETF,
  the great backtest comes from dodging *one* crash — a lucky regime, not a durable edge.

*To upgrade the Signal stamp honestly: populate `_cache/` with
`data.fetch_daily('URA', fetch=True)` (and `'URNM'`) and re-run; if the live tape's timing
edge clears HAC t = 2 out-of-sample, the stamp can move toward REAL. Until then, WEAK.*
