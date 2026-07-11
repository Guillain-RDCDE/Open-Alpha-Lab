# Study 645 — ECB Announcement Effect 🇪🇺

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — do euro-area equities react on ECB decision days? | ![Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square) | *Real on the vol · None on the drift.* FEZ's realized high-low range runs **1.19×** the normal-day baseline (Welch *t* = **+2.66**, Newey-West *t* = **+2.83**, placebo *p* = **0.0008**) — but the **signed** decision-day return is indistinguishable from noise (Welch *t* = **−0.82**, NW *t* = **−0.83**, placebo *p* = **0.345**), and EURUSD's absolute move doesn't differ from an ordinary day either (*t* = +0.65). |
| **Tradability** — can you trade the announcement? | ![Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) | Holding FEZ for the decision day only loses money **gross** (−10.3 bps/event, Welch *t* = −0.82) with a **−13.3%** worst day — there is nothing to harvest before a single basis point of cost is even charged. |
| **"Louder, not more directional"?** | ![Confirmed](https://img.shields.io/badge/Louder_not_more_directional%3F-Confirmed-8b949e?style=flat-square) | Decision days are measurably noisier (range 1.19×, *t* ≥ 2.66 two ways, placebo *p* < 0.001) while direction (FEZ, *t* = −0.82) and the FX reaction (EURUSD, *t* = +0.65) are both coin flips. |

> **In one sentence:** across **207 scheduled ECB Governing Council decisions 2005–2026**
> euro-area equities (FEZ) genuinely trade **~19% louder** on decision day (Welch *t* ≥ 2.66,
> placebo *p* = 0.0008) but show **zero** directional edge (*t* = −0.82) and no Lucca-Moench-style
> pre-meeting run-up (*t* = +0.54) — the ECB moves the market's *pulse*, not its *sign*, and a
> timer strategy that tries to ride the day loses money before costs — **Mixed, and a Mirage
> to trade.**

## What we tested

We hardcode the full calendar of scheduled ECB Governing Council monetary-policy decisions —
**208 dates 2005–2026** (monthly era through 2014, the Governing Council's own switch to a
6-week cycle from 2015), sourced from the ECB's year-ahead schedule press releases and
cross-checked against the effective-date jumps in its own key-rate series — and test FEZ (the
tradable Euro STOXX 50 ETF) decision-day return and realized range against all other days:
Welch and Newey-West *t*, a two-sided random-calendar placebo, a Lucca-Moench-style pre-meeting
event window, a justified era split at the 2015 cadence change, and EURUSD's own reaction as a
cross-check. **Tradability** is a "costs on a timer" rule — hold FEZ for the decision day only,
entered at the prior close (the calendar is public months ahead — zero look-ahead) — swept
across a one-way-cost ladder. A 20-seed synthetic control proves the machinery is unbiased. As-of
**2026-06-30**.

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | why the ECB should move euro-area markets, what actually happens on decision day — a louder tape, a coin-flip direction — and why "trade the ECB" loses money before costs, in plain language |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the Welch/HAC splits, the realized-range robustness (with its era-split caveat), the EURUSD cross-check, the pre-meeting event window, the timer strategy with costs and tails, and the 20-seed synthetic control |

Sources & literature map: [docs/references.md](docs/references.md). Reproducible headline run: [docs/results.md](docs/results.md).

---

*Engine: [`ecb_announcement_effect/`](ecb_announcement_effect/). The decision calendar is
hardcoded from the ECB's own schedule press releases and statement archive; FEZ and EURUSD=X
carry no survivorship (baskets/rates, not a hand-picked panel). Siblings:
[637-fomc-vol-crush](../637-fomc-vol-crush/), [517-pre-fomc-drift](../517-pre-fomc-drift/),
[135-fomc-cycle](../135-fomc-cycle/), [322-fomc-blackout](../322-fomc-blackout/) test the
**Fed's** version of this question; [606-opec-announcement-effect](../606-opec-announcement-effect/)
is the closest sibling in shape (a non-Fed scheduled decision, same Real-vol/None-drift
verdict); [314-jackson-hole](../314-jackson-hole/) is an unscheduled-content central-bank
*speech*, not a rate decision. **Not investment advice** — research & education.
See [LICENSE](../../LICENSE).*
