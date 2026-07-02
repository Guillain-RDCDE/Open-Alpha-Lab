# Study 547 — Blue-Monday 🫐

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does market mood sag on Blue Monday? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The literal day is untestable — **30 of 34** Blue Mondays are **market holidays** (MLK Day), leaving **n=4** (diff +7.2 bps, *t* +0.41). The tradable next-open proxy leans the right way (**−26.8 bps**, blue −22.0 vs other +4.8) but the Welch *t* is only **−1.24**, placebo *p* **0.184**, the vol 'spike' is **+1.1 pts** (none), and the sign **flips in all four sub-periods**. No |t| ≥ 2. |
| **Tradability** — could you trade it? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | You cannot trade a **closed market**. The literal-day timer is indistinguishable from buy-and-hold (**10.74%** vs **10.76%** CAGR, 4 tradable days in 33 years); shorting the sign-flipping proxy day is a coin toss with a borrow fee. |
| **Is Blue Monday tradable at all?** | ![Busted](https://img.shields.io/badge/Busted-8b949e?style=flat-square) | Blue Monday is a 2005 **PR stunt** (Cliff Arnall, Sky Travel) that happens to be a **market holiday**. It is not a market event. |

> **In one sentence:** 'Blue Monday' — the third Monday of January, marketed as the most depressing day of the year — turns out to be a day the US stock market is *literally closed* (it collides with MLK Day in 30 of 34 years), and the one tradable proxy (the Tuesday after) leans faintly negative (−26.8 bps) but never clears |*t*| ≥ 2, shows no vol spike, and flips sign in every sub-period — a pseudo-scientific mood myth with no market echo.

## What we tested

The pseudo-science claim: **'Blue Monday'** (Cliff Arnall, 2005) — the *third Monday of January* —
is the most depressing day of the year, and if seasonal mood drives markets (the SAD thesis,
[Study 150](../../150-sad-effect/)) then equity returns should **dip** and realised **volatility**
should **spike** relative to an ordinary Monday. We test this on **total-return SPY daily closes
(1993-2026)**: the literal Blue-Monday-vs-other-Monday return and vol split (which the **MLK-Day
holiday collision** empties to n=4), a **Blue-adjacent** proxy (the first open session after Blue
Monday, restoring a full 33-obs sample) with a **Welch two-sample *t***, a **random-day placebo**
null, a **realised-vol comparison**, a **four-window sign-stability sweep**, a cost-honest timer
(shorts pay borrow), and a **seed-robust synthetic positive control** that plants a mood dip and
proves the engine catches it. Day-of-week is calendar-known → **no execution lag**; the thin
one-per-year sample is named on the Signal axis. *A mood-anomaly cousin of the
[SAD-Effect (150)](../../150-sad-effect/) and a January special case of the
[Monday-Effect (224)](../../224-monday-effect/).*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what Blue Monday is, why it's a PR stunt, the punchline that the market is *closed* that day, and why the Tuesday-after 'dip' is just noise |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the literal n=4 dead-end, the Blue-adjacent Welch *t*, the random-day placebo, the vol comparison, the four-window sign-flip, costs & borrow, and the seed-robust synthetic control |

The fingerprinted real-data run (SPY 1993-2026, tape fp `d37d9f8a7153`, as-of 2026-06-30) is in
[docs/results.md](docs/results.md); the offline machinery proof runs on the deterministic synthetic
world in [`blue_monday/data.py`](blue_monday/data.py).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`blue_monday/`](blue_monday/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
