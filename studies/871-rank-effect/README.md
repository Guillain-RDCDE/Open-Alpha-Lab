# Study 871 — The Rank Effect 🏅

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do the best- and worst-ranked names under-earn the middle (Hartzmark)? | ![None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) | Hartzmark's rank effect leaves **no** cross-sectional return footprint on 50 liquid US mega-caps. The specified long-middle / short-extremes spread is **−1.65 bps/day** (Newey-West *t* = **−1.86**) — *wrong-signed* (the extreme-ranked names if anything *out*-earned the middle) and **insignificant**. And once you **control for the raw return level** — the whole point of the rank effect — the spread collapses to **+0.11 bps/day** (*t* = **+0.18**), a flat zero: the tiny raw tilt is a momentum artefact of the tails, not a rank-position effect. Not robust across eras (*t* = −0.50 / −1.99); the observed value sits ~2σ into the *wrong* tail of a 1,000-permutation placebo. A 20-seed synthetic control recovers a *planted* rank-extremity relation cleanly, so this is a real absence, not machinery. *Survivorship: current-membership mega-caps — magnitudes are an upper bound.* |
| **Tradability** — can you get paid for it? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | The specified book loses money gross and net (**−3.79 bps/day** at 1 bp one-way, −11.79 at 5 bps). Even the data-mined *sign-flip* earns only +1.65 bps/day gross, which the **2.14 bps/day** round-trip friction at a mere 1 bp already eats — a Mirage in either direction. |

> **In one sentence:** the celebrated rank effect — investors dump their best- and
> worst-ranked positions, so the extremes should under-earn — **leaves no tradable
> cross-sectional signal on liquid US mega-caps**: the long-middle / short-extremes spread
> is wrong-signed and insignificant, and **vanishes entirely once the raw return level is
> controlled for** (+0.11 bps/day, *t* = +0.18), so the honest read is **claimed signal
> absent, paycheck a mirage**.

## What we tested

Hartzmark (2015), **"The Worst, the Best, Ignoring All the Rest: The Rank Effect and Trading
Behavior"**: investors disproportionately **sell the best- and worst-ranked positions** in
their portfolio (salience of the extremes), putting predictable selling pressure on those
names — so the extreme-ranked names should **under-earn the middle** next period. We take a
self-contained cross-sectional proxy on a **liquid 50-name US cross-section (yfinance daily
OHLC, total-return, 2010-01-04 → 2026-06-30)**: each day **rank the names by trailing-42-day
return**, **long the middle 40% / short both 20% tails** (rank-1 winners + rank-N losers),
sorted point-in-time (signal known at the close of `t−1`, one shift, zero look-ahead), with a
Newey-West *t* on the daily spread, an explicit **level-controlled** cut (residualise the
forward return on a quadratic in the raw trailing-return level, then re-measure), a
1,000-permutation placebo, a two-era robustness cut, a costed long-short timer, and a 20-seed
synthetic positive control. The universe is a **current-membership** survivor set
(`quantlab.universe` opt-in guard) — named on the **Signal** axis. **Dedup:**
[327-disposition-effect](../327-disposition-effect/) is a **purchase-price** reference effect
(sell winners, ride losers), not a reference-free *rank position*;
[365-lottery-max-effect](../365-lottery-max-effect/) sorts on the single **MAX** daily return,
one right tail, not symmetric rank extremity; [806-prospect-theory-value](../806-prospect-theory-value/)
values the whole gain/loss path, not the **ordinal rank**; [202-fifty-two-week-low](../202-fifty-two-week-low/)
anchors on a name's **own 52-week extreme**, not a **cross-sectional** rank among peers.
As-of **2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why dumping your best- and worst-ranked names *should* leave the extremes under-earning — and why on mega-caps there is nothing there |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the spread Newey-West *t*, the level-controlled residual spread, the pooled Welch book test, the 1,000-permutation placebo, the two-era cut, the cost math, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`rank_effect/`](rank_effect/). Cross-section pulled through the
`quantlab.universe` survivorship guard (current membership → magnitudes are an upper bound).
**Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
