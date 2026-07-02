# Study 555 — OpenTable-Reservations 🍽️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

**Do restaurant-reservation trends nowcast restaurant and consumer-discretionary stocks?**

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does the reservations surprise predict next-week returns? | ![Weak](https://img.shields.io/badge/Weak-dab617?style=flat-square) | The alt-data nowcast is **plausible and literature-backed**, and the engine is a faithful detector — on a *planted* world the surprise predicts next-week basket returns at Newey-West *t* **+4.49**, placebo *p* **0.0005**, control flat at the null (mean slope-*t* **−0.01**). But **there is no free real seated-diners tape** to confirm it, so it cannot clear the `REAL` bar (which needs *t* ≥ 2 on a **real** tape). Literature-plausible, machinery-proven, never tape-confirmed → `WEAK`. |
| **Tradability** — could you actually run it? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | OpenTable's seated-diners feed was a transient, restated, discontinued pandemic-era HTML dashboard — never a cache-able multi-cycle panel a retail desk can license and reproduce. The overlay's +9.3% net-of-cost edge lives **only in the synthetic demonstrator**; there is no real signal to trade. |

> **In one sentence:** dining-reservation recovery as a consumer nowcast for restaurant stocks / XLY is a plausible, literature-backed idea and our engine would catch it if it were real (on a planted world the reservations surprise predicts next-week returns at *t* +4.49 with the control flat at the null) — but the free OpenTable seated-diners feed was a transient, restated, discontinued dashboard, so there is **no real tape** to confirm the claim, and it is capped at `WEAK` × `MIRAGE`.

## What we tested

The claim is an **alt-data nowcast**: an OpenTable-style *seated-diners* index (year-over-year
people seated via online reservations) should lead the tape, so this week's reservations *surprise*
predicts next week's return of a restaurant basket (and, loosely, consumer-discretionary / XLY). We
build the tradable **surprise** — reservations YoY with its own short trend and the contemporaneous
market factor removed (so it isn't just market beta) — and run a **predictive regression** of
next-week basket return on this week's surprise with a market control and Newey-West HAC *t*, a
**label-shuffle placebo** null, a sign-of-surprise **timing overlay** with one-way costs and a
short borrow (gross AND net), a four-window sub-period sweep, and a seed-robust (25-seed) synthetic
positive control that plants the nowcast and proves the engine catches it and stays flat at the
null. **Data caveat, named on the Signal axis:** there is no free, stable, machine-readable
seated-diners history to fetch and cache, so this is a **synthetic-only** study — `fetch_series`
returns empty by design, `HAVE_REAL` is always False, and the Signal axis is capped below `REAL`.
*Same synthetic-only pattern as the desk's [273 Lego-Returns](../273-lego-returns/),
[275 Whisky-Cask](../275-whisky-cask/) and [276 Sneaker-Resale](../276-sneaker-resale/) — here it's
the alt-data-**nowcast** instance (a predictive weekly regression, not a collectible index).*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what a reservations nowcast is, why "diners come back → stocks follow" is intuitive, and why there's no free data to actually check it |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the surprise construction, the predictive regression with a market control + Newey-West *t*, the placebo null, the timing overlay with costs & borrow, the four-window sweep, and the seed-robust synthetic positive control |

The fingerprinted machinery run (synthetic demonstrator, `seed = 555`, 312 weeks, nowcast planted
at `nowcast_beta = 0.35`, panel fp `2491c27d3bc8`) is in [docs/results.md](docs/results.md); the
offline core lives in [`opentable_reservations/data.py`](opentable_reservations/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`opentable_reservations/`](opentable_reservations/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
