# Study 554 — Airline-Bookings ✈️

[![tests](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Guillain-RDCDE/Open-Alpha-Lab/actions/workflows/tests.yml)

> *Part of [Open-Alpha-Lab](../../README.md) — see the [desk](../../README.md) and its [house style](../../METHODOLOGY.md).*

**Do rising flight bookings lead airline stocks — or is the signal already in the price by the time you see it?**

## Verdict

| Axis | Stamp | Why |
|---|---|---|
| **Signal** — does booking momentum *lead* airline returns? | ![None](https://img.shields.io/badge/None-c0392b?style=flat-square) | The **predictive** regression (next-month return on this-month bookings) is **noise**: HAC *t* **−0.07**, corr **−0.005**, placebo *p* **0.94** — and the sign wanders across every sub-sample. No forward edge. (Synthetic-only tape ⇒ capped at `WEAK` regardless — no free booking index exists.) |
| **Tradability** — does the timing rule pay? | ![Mirage](https://img.shields.io/badge/Mirage-c0392b?style=flat-square) | The sign-of-momentum rule **loses to doing nothing**: long/flat nets **−0.5%/yr** vs buy-and-hold **+3.3%/yr** (−3.8 pp); the long/short version bleeds **−5.6%/yr** net after a 300 bps short borrow. Nothing to harvest. |
| **"Already in the price?"** | ![Confirmed](https://img.shields.io/badge/Confirmed-8b949e?style=flat-square) | The **contemporaneous** regression is huge — same-month return on bookings has HAC *t* **+12.6**, corr **+0.66** — while the *forward* one is dead. The lead is contemporaneous, not predictive: the demand is already discounted by the time the booking print is public. |

> **In one sentence:** flight bookings and airline stocks move together beautifully in the *same* month (HAC *t* +12.6) — but that co-move does **not** lead: on the forward return you could actually trade, booking momentum is noise (HAC *t* −0.07, placebo *p* 0.94), the timing rule loses to buy-and-hold, and by the time you see the booking signal it is already in the price.

## What we tested

The alt-data pitch: a proprietary **flight-bookings momentum** index turns up *before* airline
earnings and therefore *before* airline stock returns, so buying the sector when bookings are
strong should harvest a predictable edge. Because **no free, point-in-time flight-bookings index is
retail-reachable** (the real card-panel / GDS feeds are paywalled), this is a **synthetic-only**
study: a deterministic monthly generator (seed 554) with a single knob — `lead_beta` — that plants
a genuine *forward* lead, and a second, `contemp_beta`, that sets how much of the signal is *already
in the price this month*. The headline world is the realistic **efficient-market** case
(`contemp_beta = 0.9`, `lead_beta = 0`): bookings co-move with the sector but carry no forward
information. We run the honest test — a predictive regression with a **Newey-West** HAC *t*, a
**contemporaneous** check, a **block-shuffle placebo**, a timing rule with costs + short borrow
(gross *and* net), a six-window robustness sweep, and a **seed-robust synthetic positive control**
(25 seeds) proving the engine catches a *planted* lead and stays flat at the null. The
data-availability limit is named on the Signal axis (synthetic-only ⇒ `WEAK` ceiling). *Distinct
from the sentiment leads — [257 AAII](../257-aaii-sentiment/), [335 Buzz-ETF](../335-buzz-sentiment-etf/),
[392 Glassdoor](../392-glassdoor-sentiment/); this is the **travel-booking** instance and its
contemporaneous-vs-predictive split.*

## The full teardown lives in the notebooks

| | For whom | Inside |
|---|---|---|
| **[01_for_the_curious](notebooks/01_for_the_curious.ipynb)** | the curious | what "booking momentum leads stocks" means, why the same-month chart looks so convincing, and why the *forward* edge is gone |
| **[02_for_the_quants](notebooks/02_for_the_quants.ipynb)** | quants | the predictive vs contemporaneous HAC-*t* split, the block-shuffle placebo, the timing rule net of costs + borrow, the six-window sweep, and the seed-robust synthetic positive control |

The fingerprinted headline run (synthetic efficient-market world, seed 554, frame fp `fb00515e49fb`,
as-of 2026-06-30) is in [docs/results.md](docs/results.md); the offline machinery lives in
[`airline_bookings/`](airline_bookings/).

---

*Sources & literature map: [docs/references.md](docs/references.md). Engine: [`airline_bookings/`](airline_bookings/). **Not investment advice** — research & education. See [LICENSE](../../LICENSE).*
